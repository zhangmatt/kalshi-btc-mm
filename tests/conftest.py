import json

import pytest

from kalshi_mm.recorder import EventRecorder

START_S = 1_800_000_000


def make_metadata(ticker: str = "KXBTC15M-TEST", result: str = "yes") -> dict:
    return {
        "ticker": ticker,
        "event_ticker": "KXBTC15M",
        "title": "test",
        "status": "active",
        "open_time": "2027-01-15T08:00:00+00:00",
        "close_time": "2027-01-15T08:15:00+00:00",
        "expected_expiration_time": "2027-01-15T08:20:00+00:00",
        "floor_strike": 100.0,
        "price_ranges": [{"start": 0, "end": 1, "step": 0.01}],
        "rules_primary": "test",
        "rules_secondary": "test",
        "result": result,
    }


def write_active_recording(path, *, ticker: str = "KXBTC15M-TEST", seconds: int = 60, trades_per_s: int = 2):
    """A synthetic window with book, BRTI drift, trades, and a close marker."""
    metadata = make_metadata(ticker)
    with EventRecorder(path) as recorder:
        recorder.write(
            source="system", event_type="market_metadata", payload=metadata, receive_ts_ms=START_S * 1000
        )
        recorder.write(
            source="kalshi",
            event_type="orderbook_snapshot",
            payload={
                "type": "orderbook_snapshot",
                "seq": 1,
                "msg": {
                    "market_ticker": ticker,
                    "yes_dollars_fp": [["0.49", "10"], ["0.48", "30"]],
                    "no_dollars_fp": [["0.51", "10"], ["0.52", "30"]],
                },
            },
            receive_ts_ms=START_S * 1000,
        )
        for second in range(seconds):
            ts_ms = (START_S + second) * 1000
            recorder.write(
                source="kalshi",
                event_type="cfbenchmarks_value",
                payload={
                    "type": "cfbenchmarks_value",
                    "msg": {
                        "index_id": "BRTI",
                        "data": json.dumps({"time": ts_ms, "value": str(100.0 + 0.003 * second)}),
                    },
                },
                exchange_ts_ms=ts_ms,
                receive_ts_ms=ts_ms,
            )
            for k in range(trades_per_s):
                trade_ts = ts_ms + 300 + 200 * k
                recorder.write(
                    source="kalshi",
                    event_type="trade",
                    payload={
                        "type": "trade",
                        "msg": {
                            "market_ticker": ticker,
                            "yes_price_dollars": "0.48" if (second + k) % 3 else "0.51",
                            "count_fp": "6",
                            "taker_book_side": "ask" if (second + k) % 3 else "bid",
                            "ts_ms": trade_ts,
                        },
                    },
                    exchange_ts_ms=trade_ts,
                    receive_ts_ms=trade_ts,
                )
        recorder.write(
            source="system",
            event_type="market_status_at_close",
            payload=metadata,
            receive_ts_ms=(START_S + 900) * 1000,
        )
    return path


@pytest.fixture
def active_recording(tmp_path):
    return write_active_recording(tmp_path / "events_test.jsonl")
