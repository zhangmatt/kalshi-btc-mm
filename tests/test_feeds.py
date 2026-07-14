import time
from decimal import Decimal

import pytest

from kalshi_mm.feeds import (
    ExternalBookState,
    ExternalBookUpdate,
    _binance_depth,
    _binance_price,
    _coinbase_depth,
    _kraken_depth,
)


def test_binance_trade_price():
    assert _binance_price({"e": "trade", "T": 1234, "p": "100.25"}) == (1234, 100.25)


def test_binance_book_ticker_midpoint():
    before = time.time_ns() // 1_000_000
    ts_ms, price = _binance_price({"b": "100.00", "a": "100.20"})
    after = time.time_ns() // 1_000_000
    assert before <= ts_ms <= after
    assert price == pytest.approx(100.10)


def test_binance_rejects_crossed_book():
    assert _binance_price({"b": "100.20", "a": "100.00"}) is None


def test_external_book_state_applies_absolute_updates():
    state = ExternalBookState(2)
    state.apply(
        ExternalBookUpdate(
            "snapshot",
            1,
            ((Decimal("100"), Decimal("2")), (Decimal("99"), Decimal("3"))),
            ((Decimal("101"), Decimal("4")), (Decimal("102"), Decimal("5"))),
        )
    )
    state.apply(
        ExternalBookUpdate(
            "update",
            2,
            ((Decimal("100"), Decimal("0")), (Decimal("99.5"), Decimal("7"))),
            ((Decimal("101"), Decimal("6")),),
        )
    )
    snapshot = state.snapshot()
    assert snapshot.bids == ((Decimal("99.5"), Decimal("7")), (Decimal("99"), Decimal("3")))
    assert snapshot.asks[0] == (Decimal("101"), Decimal("6"))


def test_binance_partial_depth_is_parsed_as_snapshot():
    update = _binance_depth(
        {
            "lastUpdateId": 160,
            "bids": [["100.0", "2.5"]],
            "asks": [["100.1", "3.5"]],
        }
    )[0]
    assert update.kind == "snapshot"
    assert update.sequence == 160
    assert update.bids == ((Decimal("100.0"), Decimal("2.5")),)


def test_coinbase_level2_parser_preserves_absolute_quantities():
    updates = _coinbase_depth(
        {
            "channel": "l2_data",
            "timestamp": "2026-07-14T01:02:03.456Z",
            "sequence_num": 42,
            "events": [
                {
                    "type": "update",
                    "updates": [
                        {
                            "side": "bid",
                            "event_time": "2026-07-14T01:02:03.455Z",
                            "price_level": "62700.01",
                            "new_quantity": "1.25",
                        },
                        {
                            "side": "offer",
                            "event_time": "2026-07-14T01:02:03.456Z",
                            "price_level": "62700.02",
                            "new_quantity": "0",
                        },
                    ],
                }
            ],
        }
    )
    assert updates[0].sequence == 42
    assert updates[0].bids == ((Decimal("62700.01"), Decimal("1.25")),)
    assert updates[0].asks == ((Decimal("62700.02"), Decimal("0")),)


def test_kraken_parser_and_checksum_match_official_example():
    bids = [
        ("45283.5", "0.10000000"), ("45283.4", "1.54582015"),
        ("45282.1", "0.10000000"), ("45281.0", "0.10000000"),
        ("45280.3", "1.54592586"), ("45279.0", "0.07990000"),
        ("45277.6", "0.03310103"), ("45277.5", "0.30000000"),
        ("45277.3", "1.54602737"), ("45276.6", "0.15445238"),
    ]
    asks = [
        ("45285.2", "0.00100000"), ("45286.4", "1.54571953"),
        ("45286.6", "1.54571109"), ("45289.6", "1.54560911"),
        ("45290.2", "0.15890660"), ("45291.8", "1.54553491"),
        ("45294.7", "0.04454749"), ("45296.1", "0.35380000"),
        ("45297.5", "0.09945542"), ("45299.5", "0.18772827"),
    ]
    payload = {
        "channel": "book",
        "type": "snapshot",
        "data": [
            {
                "symbol": "BTC/USD",
                "bids": [{"price": price, "qty": qty} for price, qty in bids],
                "asks": [{"price": price, "qty": qty} for price, qty in asks],
                "checksum": 3310070434,
                "timestamp": "2026-07-14T01:02:03.456Z",
            }
        ],
    }
    update = _kraken_depth(payload)[0]
    state = ExternalBookState(10, truncate_to_depth=True)
    state.apply(update)
    assert state.kraken_checksum() == 3310070434


class _FakeWs:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_coinbase_sequence_gap_resets_book_and_reconnects(tmp_path):
    import json

    from kalshi_mm.feeds import ExternalBookStream
    from kalshi_mm.recorder import EventRecorder, iter_events

    log = tmp_path / "events.jsonl"
    recorder = EventRecorder(log)
    stream = ExternalBookStream(
        venue="coinbase",
        symbol="BTC-USD",
        url="wss://unused",
        subscriptions=(),
        parser=_coinbase_depth,
        recorder=recorder,
        depth=10,
        sample_hz=10.0,
        enforce_sequence=True,
        sequence_extractor=lambda row: (
            int(row["sequence_num"]) if row.get("sequence_num") is not None else None
        ),
    )
    stream._ws = _FakeWs()
    snapshot = {
        "channel": "l2_data",
        "timestamp": "2026-07-14T01:02:03.456Z",
        "sequence_num": 0,
        "events": [
            {
                "type": "snapshot",
                "updates": [
                    {"side": "bid", "price_level": "100", "new_quantity": "1"},
                    {"side": "offer", "price_level": "101", "new_quantity": "1"},
                ],
            }
        ],
    }
    stream._on_message(None, json.dumps(snapshot))
    assert stream.state.initialized
    stream._on_message(None, json.dumps({"channel": "heartbeats", "sequence_num": 1}))
    assert stream.state.initialized
    # A skipped connection-level sequence number means lost data: reset + reconnect.
    stream._on_message(None, json.dumps({"channel": "heartbeats", "sequence_num": 3}))
    assert stream._ws.closed
    assert not stream.state.initialized
    recorder.close()
    events = list(iter_events(log))
    assert any(
        row["event_type"] == "external_book_status" and row["payload"]["status"] == "sequence_gap"
        for row in events
    )
