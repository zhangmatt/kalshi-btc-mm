from __future__ import annotations

import json
import ssl
import statistics
import threading
import time
import zlib
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Callable, Optional

import websocket
import certifi

from .recorder import EventRecorder
from .volatility import MultiHorizonVolatility, VolatilityForecast


@dataclass(frozen=True)
class ReferencePrice:
    ts_ms: int
    price: float
    venues: tuple[str, ...]
    dispersion_bps: float


@dataclass(frozen=True)
class ExternalBookUpdate:
    kind: str
    ts_ms: int
    bids: tuple[tuple[Decimal, Decimal], ...]
    asks: tuple[tuple[Decimal, Decimal], ...]
    sequence: Optional[int] = None
    checksum: Optional[int] = None


@dataclass(frozen=True)
class ExternalBookSnapshot:
    bids: tuple[tuple[Decimal, Decimal], ...]
    asks: tuple[tuple[Decimal, Decimal], ...]

    @property
    def valid(self) -> bool:
        return bool(self.bids and self.asks and self.bids[0][0] < self.asks[0][0])


class ExternalBookState:
    """Local L2 state used to emit bounded, replayable depth snapshots."""

    def __init__(self, depth: int, *, truncate_to_depth: bool = False):
        if depth < 1:
            raise ValueError("depth must be positive")
        self.depth = depth
        self.truncate_to_depth = truncate_to_depth
        self.bids: dict[Decimal, Decimal] = {}
        self.asks: dict[Decimal, Decimal] = {}
        self.initialized = False

    def reset(self) -> None:
        self.bids.clear()
        self.asks.clear()
        self.initialized = False

    def apply(self, update: ExternalBookUpdate) -> None:
        if update.kind == "snapshot":
            self.bids.clear()
            self.asks.clear()
            self.initialized = True
        elif update.kind != "update":
            raise ValueError(f"unsupported external book update kind: {update.kind}")
        if not self.initialized:
            return
        self._apply_side(self.bids, update.bids)
        self._apply_side(self.asks, update.asks)
        if self.truncate_to_depth:
            self.bids = dict(sorted(self.bids.items(), reverse=True)[: self.depth])
            self.asks = dict(sorted(self.asks.items())[: self.depth])

    @staticmethod
    def _apply_side(book: dict[Decimal, Decimal], updates: tuple[tuple[Decimal, Decimal], ...]) -> None:
        for price, quantity in updates:
            if price <= 0:
                continue
            if quantity <= 0:
                book.pop(price, None)
            else:
                book[price] = quantity

    def snapshot(self) -> ExternalBookSnapshot:
        return ExternalBookSnapshot(
            bids=tuple(sorted(self.bids.items(), reverse=True)[: self.depth]),
            asks=tuple(sorted(self.asks.items())[: self.depth]),
        )

    def kraken_checksum(self) -> int:
        snapshot = self.snapshot()
        encoded = "".join(
            _kraken_checksum_component(value)
            for levels in (snapshot.asks[:10], snapshot.bids[:10])
            for level in levels
            for value in level
        )
        return zlib.crc32(encoded.encode("ascii")) & 0xFFFFFFFF


class CompositeReference:
    def __init__(self, *, max_age_ms: int = 2_000):
        self.max_age_ms = max_age_ms
        self._latest: dict[str, tuple[int, float]] = {}
        self._lock = threading.Lock()
        self.volatility = MultiHorizonVolatility()

    def update(self, venue: str, ts_ms: int, price: float) -> Optional[ReferencePrice]:
        with self._lock:
            self._latest[venue] = (ts_ms, price)
            composite = self._snapshot_locked(ts_ms)
            if composite:
                self.volatility.update(ts_ms, composite.price)
            return composite

    def snapshot(self, now_ms: Optional[int] = None) -> Optional[ReferencePrice]:
        with self._lock:
            return self._snapshot_locked(now_ms or time.time_ns() // 1_000_000)

    def forecast(self) -> Optional[VolatilityForecast]:
        return self.volatility.forecast()

    def _snapshot_locked(self, now_ms: int) -> Optional[ReferencePrice]:
        fresh = {venue: value for venue, value in self._latest.items() if now_ms - value[0] <= self.max_age_ms}
        if not fresh:
            return None
        prices = [value[1] for value in fresh.values()]
        price = statistics.median(prices)
        dispersion = (max(prices) - min(prices)) / price * 10_000 if len(prices) > 1 else 0.0
        return ReferencePrice(now_ms, price, tuple(sorted(fresh)), dispersion)


class VenueTradeStream:
    def __init__(
        self,
        *,
        venue: str,
        url: str,
        subscribe: Optional[dict[str, Any]],
        parser: Callable[[dict[str, Any]], Optional[tuple[int, float]]],
        composite: CompositeReference,
        recorder: Optional[EventRecorder] = None,
        insecure_ssl: bool = False,
        verbose: bool = False,
    ):
        self.venue = venue
        self.url = url
        self.subscribe = subscribe
        self.parser = parser
        self.composite = composite
        self.recorder = recorder
        self.insecure_ssl = insecure_ssl
        self.verbose = verbose
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ws: Any = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"{self.venue}-ws", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._ws:
            self._ws.close()
        if self._thread:
            self._thread.join(timeout=3.0)

    def _on_open(self, ws: Any) -> None:
        self._record_status("connected")
        if self.subscribe:
            ws.send(json.dumps(self.subscribe, separators=(",", ":")))

    def _on_error(self, _ws: Any, error: Any) -> None:
        self._record_status("error", error=str(error))
        if self.verbose:
            print(f"[{self.venue}-ws] error: {error}", flush=True)

    def _on_close(self, _ws: Any, code: Any, reason: Any) -> None:
        self._record_status("closed", code=code, reason=reason)

    def _record_status(self, status: str, **details: Any) -> None:
        if not self.recorder:
            return
        try:
            self.recorder.write(
                source=self.venue,
                event_type="feed_status",
                payload={"status": status, **details},
            )
        except RuntimeError:
            pass

    def _on_message(self, _ws: Any, message: str) -> None:
        try:
            payload = json.loads(message)
            parsed = self.parser(payload)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            return
        if parsed is None:
            return
        ts_ms, price = parsed
        self.composite.update(self.venue, ts_ms, price)
        if self.recorder:
            self.recorder.write(
                source=self.venue,
                event_type="spot",
                payload={"price": price, "raw": payload},
                exchange_ts_ms=ts_ms,
            )

    def _run(self) -> None:
        backoff = 0.25
        while not self._stop.is_set():
            self._ws = websocket.WebSocketApp(
                self.url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )
            sslopt = (
                {"cert_reqs": ssl.CERT_NONE, "check_hostname": False}
                if self.insecure_ssl
                else {"ca_certs": certifi.where()}
            )
            self._ws.run_forever(ping_interval=30, ping_timeout=20, sslopt=sslopt)
            if self._stop.wait(backoff):
                break
            backoff = min(10.0, backoff * 2.0)


class ExternalBookStream:
    """Reconnect-safe public L2 stream with bounded, rate-limited recording."""

    def __init__(
        self,
        *,
        venue: str,
        symbol: str,
        url: str,
        subscriptions: tuple[dict[str, Any], ...],
        parser: Callable[[dict[str, Any]], tuple[ExternalBookUpdate, ...]],
        recorder: EventRecorder,
        depth: int,
        sample_hz: float,
        truncate_to_depth: bool = False,
        validate_kraken_checksum: bool = False,
        enforce_sequence: bool = False,
        insecure_ssl: bool = False,
        verbose: bool = False,
    ):
        if sample_hz <= 0:
            raise ValueError("sample_hz must be positive")
        self.venue = venue
        self.symbol = symbol
        self.url = url
        self.subscriptions = subscriptions
        self.parser = parser
        self.recorder = recorder
        self.state = ExternalBookState(depth, truncate_to_depth=truncate_to_depth)
        self.sample_interval_ms = max(1, round(1_000 / sample_hz))
        self.validate_kraken_checksum = validate_kraken_checksum
        self.enforce_sequence = enforce_sequence
        self.insecure_ssl = insecure_ssl
        self.verbose = verbose
        self.last_event_ms = 0
        self.last_record_ms = 0
        self._last_sequence: Optional[int] = None
        self._last_fingerprint: Optional[tuple[Any, ...]] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ws: Any = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"{self.venue}-l2-ws", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._ws:
            self._ws.close()
        if self._thread:
            self._thread.join(timeout=3.0)

    def _on_open(self, ws: Any) -> None:
        self.state.reset()
        self._last_sequence = None
        self._last_fingerprint = None
        self._record_status("connected")
        for subscription in self.subscriptions:
            ws.send(json.dumps(subscription, separators=(",", ":")))

    def _on_error(self, _ws: Any, error: Any) -> None:
        self._record_status("error", error=str(error))
        if self.verbose:
            print(f"[{self.venue}-l2] error: {error}", flush=True)

    def _on_close(self, _ws: Any, code: Any, reason: Any) -> None:
        self._record_status("closed", code=code, reason=reason)

    def _record_status(self, status: str, **details: Any) -> None:
        try:
            self.recorder.write(
                source=self.venue,
                event_type="external_book_status",
                payload={"status": status, "symbol": self.symbol, **details},
            )
        except RuntimeError:
            pass

    def _on_message(self, _ws: Any, message: str) -> None:
        receive_ts_ms = time.time_ns() // 1_000_000
        try:
            payload = json.loads(message, parse_float=Decimal)
            updates = self.parser(payload)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            self._record_status("parse_error", error=str(exc))
            return
        if not updates:
            return

        sequence = updates[0].sequence
        if sequence is not None and self.enforce_sequence:
            if self._last_sequence is not None and sequence > self._last_sequence + 1:
                self._record_status(
                    "sequence_gap",
                    expected=self._last_sequence + 1,
                    received=sequence,
                )
                self.state.reset()
                self._ws.close()
                return
            if self._last_sequence is not None and sequence <= self._last_sequence:
                return
            self._last_sequence = sequence

        for update in updates:
            self.state.apply(update)
            if not self.state.initialized:
                continue
            if self.validate_kraken_checksum and update.checksum is not None:
                local_checksum = self.state.kraken_checksum()
                if local_checksum != update.checksum:
                    self._record_status(
                        "checksum_mismatch",
                        expected=update.checksum,
                        calculated=local_checksum,
                    )
                    self.state.reset()
                    self._ws.close()
                    return
            self.last_event_ms = receive_ts_ms
            self._record_book(update, receive_ts_ms, force=update.kind == "snapshot")

    def _record_book(self, update: ExternalBookUpdate, receive_ts_ms: int, *, force: bool) -> None:
        snapshot = self.state.snapshot()
        if not snapshot.valid:
            return
        fingerprint = (snapshot.bids, snapshot.asks)
        if not force:
            if fingerprint == self._last_fingerprint:
                return
            if receive_ts_ms - self.last_record_ms < self.sample_interval_ms:
                return
        best_bid, best_bid_qty = snapshot.bids[0]
        best_ask, best_ask_qty = snapshot.asks[0]
        midpoint = (best_bid + best_ask) / Decimal(2)
        top_total = best_bid_qty + best_ask_qty
        microprice = (
            (best_ask * best_bid_qty + best_bid * best_ask_qty) / top_total
            if top_total > 0
            else midpoint
        )
        bid_depth = sum(quantity for _, quantity in snapshot.bids)
        ask_depth = sum(quantity for _, quantity in snapshot.asks)
        depth_total = bid_depth + ask_depth
        imbalance = (bid_depth - ask_depth) / depth_total if depth_total > 0 else Decimal(0)
        payload = {
            "symbol": self.symbol,
            "depth": self.state.depth,
            "update_kind": update.kind,
            "bids": [[str(price), str(quantity)] for price, quantity in snapshot.bids],
            "asks": [[str(price), str(quantity)] for price, quantity in snapshot.asks],
            "best_bid": float(best_bid),
            "best_ask": float(best_ask),
            "mid": float(midpoint),
            "microprice": float(microprice),
            "depth_imbalance": float(imbalance),
            "sequence_num": update.sequence,
            "exchange_checksum": update.checksum,
        }
        self.recorder.write(
            source=self.venue,
            event_type="external_orderbook",
            payload=payload,
            exchange_ts_ms=update.ts_ms,
            receive_ts_ms=receive_ts_ms,
        )
        self.last_record_ms = receive_ts_ms
        self._last_fingerprint = fingerprint

    def _run(self) -> None:
        backoff = 0.25
        while not self._stop.is_set():
            self._ws = websocket.WebSocketApp(
                self.url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )
            sslopt = (
                {"cert_reqs": ssl.CERT_NONE, "check_hostname": False}
                if self.insecure_ssl
                else {"ca_certs": certifi.where()}
            )
            self._ws.run_forever(ping_interval=30, ping_timeout=20, sslopt=sslopt)
            if self._stop.wait(backoff):
                break
            backoff = min(10.0, backoff * 2.0)


def _iso_ms(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)


def _decimal(value: Any) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


def _kraken_checksum_component(value: Decimal) -> str:
    normalized = format(value, "f").replace(".", "").lstrip("0")
    return normalized or "0"


def _coinbase_depth(row: dict[str, Any]) -> tuple[ExternalBookUpdate, ...]:
    if row.get("channel") != "l2_data":
        return ()
    default_ts = _iso_ms(str(row["timestamp"]))
    sequence = int(row["sequence_num"]) if row.get("sequence_num") is not None else None
    parsed: list[ExternalBookUpdate] = []
    for event in row.get("events") or ():
        kind = str(event.get("type", ""))
        if kind not in {"snapshot", "update"}:
            continue
        bids: list[tuple[Decimal, Decimal]] = []
        asks: list[tuple[Decimal, Decimal]] = []
        event_ts = default_ts
        for level in event.get("updates") or ():
            side = str(level.get("side", "")).lower()
            price = _decimal(level["price_level"])
            quantity = _decimal(level["new_quantity"])
            target = bids if side in {"bid", "buy"} else asks if side in {"ask", "offer", "sell"} else None
            if target is None:
                continue
            target.append((price, quantity))
            level_ts = str(level.get("event_time") or "")
            if level_ts and not level_ts.startswith("1970-"):
                event_ts = max(event_ts, _iso_ms(level_ts))
        parsed.append(
            ExternalBookUpdate(
                kind=kind,
                ts_ms=event_ts,
                bids=tuple(bids),
                asks=tuple(asks),
                sequence=sequence,
            )
        )
    return tuple(parsed)


def _kraken_depth(row: dict[str, Any]) -> tuple[ExternalBookUpdate, ...]:
    if row.get("channel") != "book" or row.get("type") not in {"snapshot", "update"}:
        return ()
    kind = str(row["type"])
    parsed: list[ExternalBookUpdate] = []
    for data in row.get("data") or ():
        bids = tuple((_decimal(level["price"]), _decimal(level["qty"])) for level in data.get("bids") or ())
        asks = tuple((_decimal(level["price"]), _decimal(level["qty"])) for level in data.get("asks") or ())
        parsed.append(
            ExternalBookUpdate(
                kind=kind,
                ts_ms=_iso_ms(str(data["timestamp"])),
                bids=bids,
                asks=asks,
                checksum=int(data["checksum"]) if data.get("checksum") is not None else None,
            )
        )
    return tuple(parsed)


def _binance_depth(row: dict[str, Any]) -> tuple[ExternalBookUpdate, ...]:
    bids_raw = row.get("bids") if row.get("bids") is not None else row.get("b")
    asks_raw = row.get("asks") if row.get("asks") is not None else row.get("a")
    if not isinstance(bids_raw, list) or not isinstance(asks_raw, list):
        return ()
    bids = tuple((_decimal(level[0]), _decimal(level[1])) for level in bids_raw)
    asks = tuple((_decimal(level[0]), _decimal(level[1])) for level in asks_raw)
    update_id = row.get("lastUpdateId") if row.get("lastUpdateId") is not None else row.get("u")
    return (
        ExternalBookUpdate(
            kind="snapshot",
            ts_ms=int(row.get("E") or time.time_ns() // 1_000_000),
            bids=bids,
            asks=asks,
            sequence=int(update_id) if update_id is not None else None,
        ),
    )


def _binance_price(row: dict[str, Any]) -> Optional[tuple[int, float]]:
    if row.get("e") in {"trade", "aggTrade"} and row.get("p") is not None:
        return int(row.get("T") or row.get("E") or time.time_ns() // 1_000_000), float(row["p"])
    if row.get("b") is not None and row.get("a") is not None:
        bid = float(row["b"])
        ask = float(row["a"])
        if 0 < bid <= ask:
            return time.time_ns() // 1_000_000, (bid + ask) / 2.0
    return None


def reference_streams(
    composite: CompositeReference,
    recorder: Optional[EventRecorder] = None,
    *,
    insecure_ssl: bool = False,
    binance_url: str = "wss://stream.binance.com:9443/ws/btcusdt@trade",
) -> list[VenueTradeStream]:
    return [
        VenueTradeStream(
            venue="binance",
            url=binance_url,
            subscribe=None,
            parser=_binance_price,
            composite=composite,
            recorder=recorder,
            insecure_ssl=insecure_ssl,
        ),
        VenueTradeStream(
            venue="coinbase",
            url="wss://advanced-trade-ws.coinbase.com",
            subscribe={"type": "subscribe", "product_ids": ["BTC-USD"], "channel": "ticker"},
            parser=lambda row: (
                (_iso_ms(row["timestamp"]), float(row["events"][0]["tickers"][0]["price"]))
                if row.get("channel") == "ticker" and row.get("events")
                else None
            ),
            composite=composite,
            recorder=recorder,
            insecure_ssl=insecure_ssl,
        ),
        VenueTradeStream(
            venue="kraken",
            url="wss://ws.kraken.com/v2",
            subscribe={"method": "subscribe", "params": {"channel": "ticker", "symbol": ["BTC/USD"], "snapshot": True}},
            parser=lambda row: (
                (time.time_ns() // 1_000_000, float(row["data"][0]["last"]))
                if row.get("channel") == "ticker" and row.get("data")
                else None
            ),
            composite=composite,
            recorder=recorder,
            insecure_ssl=insecure_ssl,
        ),
    ]


def external_depth_streams(
    recorder: EventRecorder,
    *,
    depth: int = 10,
    sample_hz: float = 10.0,
    insecure_ssl: bool = False,
    verbose: bool = False,
    binance_url: str = "wss://stream.binance.com:9443/ws/btcusdt@depth20@100ms",
    coinbase_url: str = "wss://advanced-trade-ws.coinbase.com",
    kraken_url: str = "wss://ws.kraken.com/v2",
) -> list[ExternalBookStream]:
    if depth not in {10, 25, 100, 500, 1_000}:
        raise ValueError("external depth must be one of 10, 25, 100, 500, or 1000")
    # Binance partial-depth supports 5/10/20 levels. Request 20 and emit the
    # same bounded depth used for the other venues.
    return [
        ExternalBookStream(
            venue="binance",
            symbol="BTCUSDT",
            url=binance_url,
            subscriptions=(),
            parser=_binance_depth,
            recorder=recorder,
            depth=min(depth, 20),
            sample_hz=sample_hz,
            truncate_to_depth=True,
            insecure_ssl=insecure_ssl,
            verbose=verbose,
        ),
        ExternalBookStream(
            venue="coinbase",
            symbol="BTC-USD",
            url=coinbase_url,
            subscriptions=(
                {"type": "subscribe", "product_ids": ["BTC-USD"], "channel": "level2"},
            ),
            parser=_coinbase_depth,
            recorder=recorder,
            depth=depth,
            sample_hz=sample_hz,
            insecure_ssl=insecure_ssl,
            verbose=verbose,
        ),
        ExternalBookStream(
            venue="kraken",
            symbol="BTC/USD",
            url=kraken_url,
            subscriptions=(
                {
                    "method": "subscribe",
                    "params": {"channel": "book", "symbol": ["BTC/USD"], "depth": depth, "snapshot": True},
                },
            ),
            parser=_kraken_depth,
            recorder=recorder,
            depth=depth,
            sample_hz=sample_hz,
            truncate_to_depth=True,
            validate_kraken_checksum=True,
            insecure_ssl=insecure_ssl,
            verbose=verbose,
        ),
    ]
