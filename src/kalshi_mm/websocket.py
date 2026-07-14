from __future__ import annotations

import json
import ssl
import threading
import time
from typing import Any, Callable, Mapping, Optional

import websocket
import certifi

from .exchange import BrtiState, KalshiOrderBook, KalshiSigner, PROD_WS_URL
from .recorder import EventRecorder


class KalshiWebSocket:
    def __init__(
        self,
        *,
        signer: KalshiSigner,
        market_ticker: str,
        order_book: KalshiOrderBook,
        brti: BrtiState,
        ws_url: str = PROD_WS_URL,
        recorder: Optional[EventRecorder] = None,
        on_event: Optional[Callable[[Mapping[str, Any]], None]] = None,
        include_private: bool = False,
        insecure_ssl: bool = False,
        verbose: bool = False,
    ):
        self.signer = signer
        self.market_ticker = market_ticker
        self.order_book = order_book
        self.brti = brti
        self.ws_url = ws_url
        self.recorder = recorder
        self.on_event = on_event
        self.include_private = include_private
        self.insecure_ssl = insecure_ssl
        self.verbose = verbose
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ws: Any = None
        self.last_message_ms = 0
        self.fatal_error: Optional[str] = None
        self._connected_at_ms = 0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="kalshi-ws", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=5.0)

    def reconnect(self) -> None:
        """Force a fresh connection (and therefore a fresh book snapshot)."""
        self.order_book.invalidate()
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass

    def _subscriptions(self) -> list[dict[str, Any]]:
        commands = [
            {
                "id": 1,
                "cmd": "subscribe",
                "params": {
                    "channels": ["orderbook_delta"],
                    "market_tickers": [self.market_ticker],
                    "use_yes_price": True,
                },
            },
            {
                "id": 2,
                "cmd": "subscribe",
                "params": {"channels": ["ticker", "trade"], "market_tickers": [self.market_ticker]},
            },
            {
                "id": 3,
                "cmd": "subscribe",
                "params": {"channels": ["cfbenchmarks_value"], "index_ids": ["BRTI"]},
            },
            {
                "id": 4,
                "cmd": "subscribe",
                "params": {
                    "channels": ["market_lifecycle_v2"],
                    "market_tickers": [self.market_ticker],
                },
            },
        ]
        if self.include_private:
            commands.append(
                {
                    "id": 5,
                    "cmd": "subscribe",
                    "params": {
                        "channels": ["fill", "user_orders", "market_positions"],
                        "market_tickers": [self.market_ticker],
                    },
                }
            )
        return commands

    def _on_open(self, ws: Any) -> None:
        self._connected_at_ms = time.time_ns() // 1_000_000
        self.order_book.invalidate()
        self._record_status("connected")
        if self.verbose:
            print("[kalshi-ws] connected; subscribing", flush=True)
        for command in self._subscriptions():
            ws.send(json.dumps(command, separators=(",", ":")))

    def _on_message(self, ws: Any, message: str) -> None:
        receive_ts_ms = time.time_ns() // 1_000_000
        try:
            envelope = json.loads(message)
        except json.JSONDecodeError:
            return
        self.last_message_ms = receive_ts_ms
        msg = envelope.get("msg") or {}
        if envelope.get("type") == "error":
            self.fatal_error = f"websocket subscription failed: {msg}"
        exchange_ts_ms = msg.get("ts_ms") or msg.get("received_at")
        if self.recorder:
            self.recorder.write(
                source="kalshi",
                event_type=str(envelope.get("type", "unknown")),
                payload=envelope,
                exchange_ts_ms=int(exchange_ts_ms) if exchange_ts_ms is not None else None,
                receive_ts_ms=receive_ts_ms,
            )
        previous_sequence = self.order_book.sequence
        updated = self.order_book.apply(envelope, use_yes_price=True)
        if envelope.get("type") == "orderbook_delta" and not updated and not self.order_book.snapshot().valid:
            if self.recorder:
                self.recorder.write(
                    source="system",
                    event_type="orderbook_sequence_gap",
                    payload={
                        "market_ticker": self.market_ticker,
                        "previous_sequence": previous_sequence,
                        "received_sequence": envelope.get("seq"),
                        "subscription_id": envelope.get("sid"),
                    },
                    receive_ts_ms=receive_ts_ms,
                )
            ws.send(
                json.dumps(
                    {
                        "id": 10_000 + int(time.time()) % 10_000,
                        "cmd": "update_subscription",
                        "params": {
                            "sids": [envelope.get("sid")],
                            "action": "get_snapshot",
                            "market_tickers": [self.market_ticker],
                        },
                    },
                    separators=(",", ":"),
                )
            )
        self.brti.apply(envelope)
        if self.on_event:
            self.on_event(envelope)

    def _on_error(self, _ws: Any, error: Any) -> None:
        status_code = getattr(error, "status_code", None)
        if status_code in {401, 403}:
            self.fatal_error = (
                f"websocket authentication rejected with HTTP {status_code}; "
                "verify KALSHI_ENV, API key ID, PEM pairing, key scope, and system clock"
            )
        self._record_status("error", error=str(error), status_code=status_code)
        if self.verbose:
            print(f"[kalshi-ws] error: {error}", flush=True)

    def _on_close(self, _ws: Any, code: Any, reason: Any) -> None:
        self.order_book.invalidate()
        self._record_status("closed", code=code, reason=reason)
        if self.verbose:
            print(f"[kalshi-ws] closed code={code} reason={reason}", flush=True)

    def _record_status(self, status: str, **details: Any) -> None:
        if not self.recorder:
            return
        try:
            self.recorder.write(
                source="kalshi",
                event_type="feed_status",
                payload={"status": status, **details},
            )
        except RuntimeError:
            pass

    def _run(self) -> None:
        backoff = 0.25
        while not self._stop.is_set():
            headers = self.signer.headers("GET", "/trade-api/ws/v2")
            self._ws = websocket.WebSocketApp(
                self.ws_url,
                header=[f"{key}: {value}" for key, value in headers.items()],
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
            self._ws.run_forever(
                ping_interval=30,
                ping_timeout=20,
                sslopt=sslopt,
                suppress_origin=True,
            )
            if self.fatal_error:
                break
            if self._connected_at_ms and time.time_ns() // 1_000_000 - self._connected_at_ms >= 5_000:
                backoff = 0.25
            if self._stop.wait(backoff):
                break
            backoff = min(10.0, backoff * 2.0)
