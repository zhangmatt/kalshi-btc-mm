from __future__ import annotations

import base64
import json
import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Iterable, Mapping, Optional

import requests


PROD_REST_URL = "https://external-api.kalshi.com/trade-api/v2"
PROD_WS_URL = "wss://external-api-ws.kalshi.com/trade-api/ws/v2"
DEMO_REST_URL = "https://demo-api.kalshi.co/trade-api/v2"
DEMO_WS_URL = "wss://external-api-ws.demo.kalshi.co/trade-api/ws/v2"


def _parse_time(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())


@dataclass(frozen=True)
class PriceRange:
    start: float
    end: float
    step: float


@dataclass(frozen=True)
class KalshiMarket:
    ticker: str
    event_ticker: str
    title: str
    status: str
    open_ts: int
    close_ts: int
    expected_expiration_ts: int
    strike_average: float
    price_ranges: tuple[PriceRange, ...]
    rules_primary: str
    rules_secondary: str
    result: str = ""

    @classmethod
    def from_api(cls, raw: Mapping[str, Any]) -> "KalshiMarket":
        ranges = tuple(
            PriceRange(float(item["start"]), float(item["end"]), float(item["step"]))
            for item in raw.get("price_ranges", [])
        )
        return cls(
            ticker=str(raw["ticker"]),
            event_ticker=str(raw.get("event_ticker", "")),
            title=str(raw.get("title", "")),
            status=str(raw.get("status", "")),
            open_ts=_parse_time(str(raw["open_time"])),
            close_ts=_parse_time(str(raw["close_time"])),
            expected_expiration_ts=_parse_time(str(raw.get("expected_expiration_time") or raw["close_time"])),
            strike_average=float(raw["floor_strike"]),
            price_ranges=ranges or (PriceRange(0.0, 1.0, 0.01),),
            rules_primary=str(raw.get("rules_primary", "")),
            rules_secondary=str(raw.get("rules_secondary", "")),
            result=str(raw.get("result", "")),
        )


class PriceGrid:
    def __init__(self, ranges: Iterable[PriceRange]):
        self.ranges = tuple(ranges)
        if not self.ranges:
            raise ValueError("at least one price range is required")

    def step(self, price: float) -> float:
        bounded = min(1.0, max(0.0, price))
        for index, item in enumerate(self.ranges):
            if item.start <= bounded < item.end or (index == len(self.ranges) - 1 and bounded <= item.end):
                return item.step
        return self.ranges[-1].step

    def floor(self, price: float) -> float:
        return self._round(price, up=False)

    def ceil(self, price: float) -> float:
        return self._round(price, up=True)

    def _round(self, price: float, *, up: bool) -> float:
        bounded = min(0.999, max(0.001, price))
        for index, item in enumerate(self.ranges):
            if item.start <= bounded < item.end or (index == len(self.ranges) - 1 and bounded <= item.end):
                units = (bounded - item.start) / item.step
                rounded = math.ceil(units - 1e-10) if up else math.floor(units + 1e-10)
                return round(item.start + rounded * item.step, 4)
        raise ValueError(f"price {price} is outside configured ranges")


@dataclass(frozen=True)
class KalshiCredentials:
    key_id: str = field(default="", repr=False)
    private_key_path: str = field(default="", repr=False)
    private_key_pem: str = field(default="", repr=False)

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "KalshiCredentials":
        return cls(
            key_id=env.get("KALSHI_API_KEY_ID", ""),
            private_key_path=env.get("KALSHI_PRIVATE_KEY_PATH", ""),
            private_key_pem=env.get("KALSHI_PRIVATE_KEY_PEM", ""),
        )

    def validate(self) -> None:
        if not self.key_id:
            raise ValueError("KALSHI_API_KEY_ID is required")
        if not self.private_key_path and not self.private_key_pem:
            raise ValueError("KALSHI_PRIVATE_KEY_PATH or KALSHI_PRIVATE_KEY_PEM is required")


class KalshiSigner:
    def __init__(self, credentials: KalshiCredentials):
        credentials.validate()
        try:
            from cryptography.hazmat.primitives import serialization
        except ImportError as exc:  # pragma: no cover - dependency error
            raise RuntimeError("install the 'kalshi' extra to enable request signing") from exc
        pem = credentials.private_key_pem.encode()
        if not pem:
            pem = Path(credentials.private_key_path).expanduser().read_bytes()
        self.key_id = credentials.key_id
        self._private_key = serialization.load_pem_private_key(pem, password=None)

    def headers(self, method: str, path: str, *, timestamp_ms: Optional[int] = None) -> dict[str, str]:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        timestamp = str(timestamp_ms if timestamp_ms is not None else int(time.time() * 1000))
        clean_path = path.split("?", 1)[0]
        message = f"{timestamp}{method.upper()}{clean_path}".encode()
        signature = self._private_key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
        }


class KalshiApiError(RuntimeError):
    def __init__(self, status_code: int, payload: Any):
        super().__init__(f"Kalshi API returned {status_code}: {payload}")
        self.status_code = status_code
        self.payload = payload


class KalshiRestClient:
    def __init__(
        self,
        *,
        base_url: str = PROD_REST_URL,
        signer: Optional[KalshiSigner] = None,
        session: Optional[requests.Session] = None,
        timeout_s: float = 5.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.signer = signer
        self.session = session or requests.Session()
        self.timeout_s = timeout_s

    def request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        json_body: Optional[dict[str, Any]] = None,
        authenticated: bool = True,
    ) -> dict[str, Any]:
        if not path.startswith("/"):
            path = "/" + path
        headers = {"Content-Type": "application/json"}
        if authenticated:
            if self.signer is None:
                raise ValueError("authenticated request requires a signer")
            headers.update(self.signer.headers(method, "/trade-api/v2" + path))
        response = self.session.request(
            method,
            self.base_url + path,
            params=params,
            json=json_body,
            headers=headers,
            timeout=self.timeout_s,
        )
        try:
            payload = response.json()
        except ValueError:
            payload = response.text
        if response.status_code >= 400:
            raise KalshiApiError(response.status_code, payload)
        return payload

    def discover_btc15m(self, *, status: str = "open") -> list[KalshiMarket]:
        payload = self.request(
            "GET",
            "/markets",
            params={"series_ticker": "KXBTC15M", "status": status, "limit": 100},
            authenticated=False,
        )
        return [KalshiMarket.from_api(raw) for raw in payload.get("markets", [])]

    def get_series(self, series_ticker: str) -> dict[str, Any]:
        payload = self.request("GET", f"/series/{series_ticker}", authenticated=False)
        return dict(payload.get("series", {}))

    def get_series_fee_changes(self, series_ticker: str) -> list[dict[str, Any]]:
        payload = self.request(
            "GET",
            "/series/fee_changes",
            params={"series_ticker": series_ticker, "show_historical": False},
            authenticated=False,
        )
        return [dict(value) for value in payload.get("series_fee_change_arr", [])]

    def get_incentives(self, *, status: str = "active", incentive_type: str = "liquidity") -> list[dict[str, Any]]:
        payload = self.request(
            "GET",
            "/incentive_programs",
            params={"status": status, "type": incentive_type, "limit": 10_000},
            authenticated=False,
        )
        return [dict(value) for value in payload.get("incentive_programs", [])]

    def get_market(self, ticker: str) -> KalshiMarket:
        payload = self.request("GET", f"/markets/{ticker}", authenticated=False)
        return KalshiMarket.from_api(payload["market"])

    def get_balance(self, *, subaccount: int = 0) -> dict[str, Any]:
        return self.request("GET", "/portfolio/balance", params={"subaccount": subaccount})

    def get_positions(self, ticker: str, *, subaccount: int = 0) -> list[dict[str, Any]]:
        payload = self.request(
            "GET", "/portfolio/positions", params={"ticker": ticker, "subaccount": subaccount, "limit": 100}
        )
        return list(payload.get("market_positions", []))

    def get_fills(self, ticker: str, *, min_ts: Optional[int] = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"ticker": ticker, "limit": 1000}
        if min_ts is not None:
            params["min_ts"] = min_ts
        payload = self.request("GET", "/portfolio/fills", params=params)
        return list(payload.get("fills", []))

    def get_orders(self, ticker: str, *, status: str = "resting") -> list[dict[str, Any]]:
        payload = self.request(
            "GET",
            "/portfolio/orders",
            params={"ticker": ticker, "status": status, "limit": 1000},
        )
        return list(payload.get("orders", []))

    def create_orders(self, orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not orders:
            return []
        payload = self.request("POST", "/portfolio/events/orders/batched", json_body={"orders": orders})
        return list(payload.get("orders", []))

    def cancel_orders(self, order_ids: list[str], *, subaccount: int = 0) -> list[dict[str, Any]]:
        if not order_ids:
            return []
        body = {"orders": [{"order_id": order_id, "subaccount": subaccount, "exchange_index": 0} for order_id in order_ids]}
        payload = self.request("DELETE", "/portfolio/events/orders/batched", json_body=body)
        return list(payload.get("orders", []))


def new_client_order_id(prefix: str = "btc15m") -> str:
    return f"{prefix}-{uuid.uuid4()}"


def build_order(
    *,
    ticker: str,
    side: str,
    price: float,
    count: float,
    expiration_time: int,
    client_order_id: Optional[str] = None,
    subaccount: int = 0,
    order_group_id: Optional[str] = None,
) -> dict[str, Any]:
    if side not in {"bid", "ask"}:
        raise ValueError("side must be bid or ask")
    order: dict[str, Any] = {
        "ticker": ticker,
        "client_order_id": client_order_id or new_client_order_id(),
        "side": side,
        "count": f"{count:.2f}",
        "price": f"{price:.4f}",
        "time_in_force": "good_till_canceled",
        "self_trade_prevention_type": "taker_at_cross",
        "expiration_time": int(expiration_time),
        "post_only": True,
        "cancel_order_on_pause": True,
        "reduce_only": False,
        "subaccount": subaccount,
        "exchange_index": 0,
    }
    if order_group_id:
        order["order_group_id"] = order_group_id
    return order


MAKER_FEE_RATE = Decimal("0.0175")
TAKER_FEE_RATE = Decimal("0.07")


def event_contract_fee(
    *,
    contracts: float,
    price: float,
    maker: bool,
    maker_multiplier: float = 1.0,
    taker_multiplier: float = 1.0,
) -> float:
    """Kalshi quadratic fee, rounded up to the next cent per the published schedule.

    The maker rate for CRYPTO15M is assumed at the standard 0.0175 multiplier until
    verified against the fee schedule or a real fill; override with maker_multiplier=0
    if the series turns out to be maker-fee-free.
    """
    multiplier = Decimal(str(maker_multiplier if maker else taker_multiplier))
    rate = MAKER_FEE_RATE if maker else TAKER_FEE_RATE
    probability = Decimal(str(price))
    raw = multiplier * rate * Decimal(str(contracts)) * probability * (Decimal("1") - probability)
    return float(max(Decimal("0"), raw).quantize(Decimal("0.01"), rounding=ROUND_CEILING))


def maker_fee_rate(*, price: float, multiplier: float = 1.0) -> float:
    """Unrounded per-contract maker fee for edge estimation.

    The charged fee rounds up to the cent per order (event_contract_fee); using
    that rounding on a single contract would overstate the marginal cost.
    """
    probability = Decimal(str(price))
    raw = Decimal(str(multiplier)) * MAKER_FEE_RATE * probability * (Decimal("1") - probability)
    return float(max(Decimal("0"), raw))


@dataclass(frozen=True)
class KalshiBbo:
    bid: Optional[float]
    ask: Optional[float]
    bid_size: float = 0.0
    ask_size: float = 0.0
    valid: bool = False


class KalshiOrderBook:
    """Maintain Kalshi's YES book from snapshot and delta messages."""

    def __init__(self, ticker: str):
        self.ticker = ticker
        self._bids: dict[int, float] = {}
        self._asks: dict[int, float] = {}
        self._seq: Optional[int] = None
        self._valid = False
        self._last_update_ms = 0
        self._lock = RLock()

    @staticmethod
    def _key(price: float) -> int:
        return int(round(price * 10_000))

    def apply(self, envelope: Mapping[str, Any], *, use_yes_price: bool = True) -> bool:
        msg_type = envelope.get("type")
        msg = envelope.get("msg") or {}
        if msg.get("market_ticker") != self.ticker:
            return False
        seq = int(envelope.get("seq", 0))
        with self._lock:
            if msg_type == "orderbook_snapshot":
                self._bids = self._levels(msg.get("yes_dollars_fp", []), convert_no=False)
                self._asks = self._levels(msg.get("no_dollars_fp", []), convert_no=not use_yes_price)
                self._seq = seq
                self._valid = True
                self._last_update_ms = int(msg.get("ts_ms") or time.time_ns() // 1_000_000)
                return True
            if msg_type != "orderbook_delta":
                return False
            if self._seq is None or seq != self._seq + 1:
                self._valid = False
                self._seq = seq
                return False
            self._seq = seq
            side = str(msg.get("side", ""))
            price = float(msg["price_dollars"])
            if side == "no" and not use_yes_price:
                price = 1.0 - price
            levels = self._bids if side == "yes" else self._asks
            key = self._key(price)
            size = levels.get(key, 0.0) + float(msg["delta_fp"])
            if size <= 1e-9:
                levels.pop(key, None)
            else:
                levels[key] = size
            self._last_update_ms = int(msg.get("ts_ms") or time.time_ns() // 1_000_000)
            return True

    @property
    def last_update_ms(self) -> int:
        with self._lock:
            return self._last_update_ms

    @property
    def sequence(self) -> Optional[int]:
        with self._lock:
            return self._seq

    def invalidate(self) -> None:
        """Require a new snapshot before the book can be used for quoting."""
        with self._lock:
            self._valid = False
            self._seq = None

    def _levels(self, values: Iterable[Iterable[Any]], *, convert_no: bool) -> dict[int, float]:
        result: dict[int, float] = {}
        for price_raw, size_raw in values:
            price = float(price_raw)
            if convert_no:
                price = 1.0 - price
            size = float(size_raw)
            if size > 0:
                result[self._key(price)] = size
        return result

    def snapshot(self) -> KalshiBbo:
        with self._lock:
            bid_key = max(self._bids) if self._bids else None
            ask_key = min(self._asks) if self._asks else None
            return KalshiBbo(
                bid=bid_key / 10_000 if bid_key is not None else None,
                ask=ask_key / 10_000 if ask_key is not None else None,
                bid_size=self._bids.get(bid_key, 0.0) if bid_key is not None else 0.0,
                ask_size=self._asks.get(ask_key, 0.0) if ask_key is not None else 0.0,
                valid=self._valid,
            )

    def level_size(self, side: str, price: float) -> float:
        if side not in {"bid", "ask"}:
            raise ValueError("side must be bid or ask")
        with self._lock:
            levels = self._bids if side == "bid" else self._asks
            return levels.get(self._key(price), 0.0)

    def top_levels(self, side: str, count: int) -> list[tuple[float, float]]:
        """Best `count` (price, size) levels, best first."""
        if side not in {"bid", "ask"}:
            raise ValueError("side must be bid or ask")
        with self._lock:
            levels = self._bids if side == "bid" else self._asks
            keys = sorted(levels, reverse=side == "bid")[:count]
            return [(key / 10_000, levels[key]) for key in keys]


@dataclass
class BrtiState:
    value: Optional[float] = None
    source_ts_ms: Optional[int] = None
    trailing_60s_average: Optional[float] = None
    final_minute_values: list[float] = field(default_factory=list)
    final_minute_average: Optional[float] = None
    final_minute_count: int = 0
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def apply(self, envelope: Mapping[str, Any]) -> bool:
        if envelope.get("type") != "cfbenchmarks_value":
            return False
        msg = envelope.get("msg") or {}
        if msg.get("index_id") != "BRTI":
            return False
        raw = json.loads(msg.get("data") or "{}")
        with self._lock:
            self.value = float(raw["value"])
            self.source_ts_ms = int(raw["time"])
            average = msg.get("avg_60s_data") or {}
            self.trailing_60s_average = float(average["value"]) if average.get("value") is not None else None
            final = msg.get("last_60s_windowed_average_15min")
            if final:
                count = int(final.get("window_size", 0))
                self.final_minute_average = float(final["value"])
                self.final_minute_count = count
                self.final_minute_values = [self.final_minute_average] * count
            else:
                self.final_minute_values.clear()
                self.final_minute_average = None
                self.final_minute_count = 0
        return True

    def snapshot(self) -> "BrtiSnapshot":
        with self._lock:
            return BrtiSnapshot(
                value=self.value,
                source_ts_ms=self.source_ts_ms,
                trailing_60s_average=self.trailing_60s_average,
                final_minute_average=self.final_minute_average,
                final_minute_count=self.final_minute_count,
            )


@dataclass(frozen=True)
class BrtiSnapshot:
    value: Optional[float]
    source_ts_ms: Optional[int]
    trailing_60s_average: Optional[float]
    final_minute_average: Optional[float]
    final_minute_count: int

    @property
    def final_minute_values(self) -> tuple[float, ...]:
        if self.final_minute_average is None:
            return ()
        return (self.final_minute_average,) * self.final_minute_count
