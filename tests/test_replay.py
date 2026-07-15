import json

import pytest

from kalshi_mm.recorder import EventRecorder
from kalshi_mm.replay import ConservativeQueueSimulator, replay
from kalshi_mm.strategy import KalshiQuote


class Bbo:
    bid = 0.49
    ask = 0.51
    bid_size = 10
    ask_size = 12


def _write_latency_fixture(path) -> None:
    start_s = 1_800_000_000
    metadata = {
        "ticker": "KXBTC15M-TEST",
        "event_ticker": "KXBTC15M-TEST",
        "title": "BTC price up in next 15 mins?",
        "status": "active",
        "open_time": "2027-01-15T08:00:00+00:00",
        "close_time": "2027-01-15T08:15:00+00:00",
        "expected_expiration_time": "2027-01-15T08:20:00+00:00",
        "floor_strike": 100.0,
        "price_ranges": [{"start": 0, "end": 1, "step": 0.01}],
        "rules_primary": "test",
        "rules_secondary": "test",
        "result": "",
    }
    with EventRecorder(path) as recorder:
        recorder.write(
            source="system", event_type="market_metadata", payload=metadata, receive_ts_ms=start_s * 1000
        )
        recorder.write(
            source="kalshi",
            event_type="orderbook_snapshot",
            payload={
                "type": "orderbook_snapshot",
                "seq": 1,
                "msg": {
                    "market_ticker": "KXBTC15M-TEST",
                    "yes_dollars_fp": [["0.40", "5"]],
                    "no_dollars_fp": [["0.60", "5"]],
                },
            },
            receive_ts_ms=start_s * 1000,
        )
        def brti(ts_ms: int) -> None:
            recorder.write(
                source="kalshi",
                event_type="cfbenchmarks_value",
                payload={
                    "type": "cfbenchmarks_value",
                    "msg": {"index_id": "BRTI", "data": json.dumps({"time": ts_ms, "value": "100.0"})},
                },
                exchange_ts_ms=ts_ms,
                receive_ts_ms=ts_ms,
            )

        def trade(ts_ms: int) -> None:
            recorder.write(
                source="kalshi",
                event_type="trade",
                payload={
                    "type": "trade",
                    "msg": {
                        "market_ticker": "KXBTC15M-TEST",
                        # At the strike the lognormal fair sits a hair under 0.5,
                        # so the strategy bids 0.48; print through that level.
                        "yes_price_dollars": "0.48",
                        "count_fp": "10",
                        "taker_book_side": "ask",
                        "ts_ms": ts_ms,
                    },
                },
                exchange_ts_ms=ts_ms,
                receive_ts_ms=ts_ms,
            )

        # The volatility forecast first resolves at +3000, producing the first
        # decision there (pending activation at +3150 with 150ms latency).
        for second in range(4):
            brti(start_s * 1000 + second * 1000)
        # 100ms after the decision: inside the latency window.
        trade(start_s * 1000 + 3_100)
        # This tick is past the activation time, so pending orders join the book.
        brti(start_s * 1000 + 4_000)
        trade(start_s * 1000 + 4_100)


def test_adjuster_context_carries_external_and_depth_state(tmp_path):
    from kalshi_mm.replay import QuoteAdjuster, replay_rows
    from kalshi_mm.exchange import KalshiMarket
    from kalshi_mm.recorder import iter_events

    path = tmp_path / "events.jsonl"
    _write_latency_fixture(path)
    rows = list(iter_events(path))
    # Splice an external book observation fresh (<2s) at the first decision,
    # which fires at the +3000ms BRTI tick once the volatility forecast resolves.
    external = {
        "receive_ts_ms": 1_800_000_000 * 1000 + 2_500,
        "source": "coinbase",
        "event_type": "external_orderbook",
        "payload": {"microprice": 100.05, "depth_imbalance": 0.4, "mid": 100.04},
    }
    index = next(i for i, row in enumerate(rows) if int(row["receive_ts_ms"]) > external["receive_ts_ms"])
    rows.insert(index, external)
    metadata = next(row["payload"] for row in rows if row["event_type"] == "market_metadata")
    market = KalshiMarket.from_api(metadata)
    captured = []

    class Capture(QuoteAdjuster):
        def adjust(self, ctx):
            captured.append(ctx)
            return ctx.fair_yes, 0.0, 0.0

    replay_rows(market, rows, latency_ms=0, maker_fee_multiplier=0.0, adjuster=Capture())
    assert captured
    first = captured[0]
    # BRTI is 100.0 and coinbase microprice 100.05 -> lead of ~+5bp.
    assert first.ext_micro_lead_bps == pytest.approx(5.0, abs=0.5)
    assert first.ext_imbalance == pytest.approx(0.4)
    # The first trade prints after the first decision; later contexts see it.
    assert any(ctx.trades_5s >= 1 for ctx in captured)


def test_crossed_posts_are_rejected_at_activation(tmp_path):
    """A bid whose price the market crossed during the latency window must not rest."""
    from kalshi_mm.exchange import KalshiMarket
    from kalshi_mm.recorder import iter_events
    from kalshi_mm.replay import replay_rows

    path = tmp_path / "events.jsonl"
    _write_latency_fixture(path)
    rows = list(iter_events(path))
    metadata = next(row["payload"] for row in rows if row["event_type"] == "market_metadata")
    market = KalshiMarket.from_api(metadata)
    # During the 150ms flight after the +3000ms decision (bid 0.48 pending),
    # the ask collapses to 0.45 — live post_only would reject the bid.
    crash_ts = 1_800_000_000 * 1000 + 3_050
    crash = {
        "receive_ts_ms": crash_ts,
        "source": "kalshi",
        "event_type": "orderbook_snapshot",
        "payload": {
            "type": "orderbook_snapshot",
            "seq": 2,
            "msg": {
                "market_ticker": "KXBTC15M-TEST",
                "yes_dollars_fp": [["0.40", "5"]],
                "no_dollars_fp": [["0.45", "5"]],
            },
        },
    }
    index = next(i for i, row in enumerate(rows) if int(row["receive_ts_ms"]) > crash_ts)
    rows.insert(index, crash)
    result = replay_rows(market, rows, latency_ms=150, maker_fee_multiplier=0.0)
    # The 0.48 bid must have been rejected at activation: no fill at 0.48.
    assert all(fill.price != 0.48 for fill in result.fills)


def test_stale_brti_cancels_resting_orders(tmp_path):
    """Orders must not keep filling through a BRTI outage live would cancel on."""
    from kalshi_mm.exchange import KalshiMarket
    from kalshi_mm.recorder import iter_events
    from kalshi_mm.replay import replay_rows

    path = tmp_path / "events.jsonl"
    _write_latency_fixture(path)
    rows = list(iter_events(path))
    metadata = next(row["payload"] for row in rows if row["event_type"] == "market_metadata")
    market = KalshiMarket.from_api(metadata)
    base = 1_800_000_000 * 1000
    # Delta keeps the book alive at +9s (trades no longer needed) while BRTI
    # is silent since +4s -> stale by >2s -> pending cancel-all at +9s...
    rows.append(
        {
            "receive_ts_ms": base + 9_000,
            "source": "kalshi",
            "event_type": "orderbook_delta",
            "payload": {
                "type": "orderbook_delta",
                "seq": 2,
                "msg": {"market_ticker": "KXBTC15M-TEST", "side": "yes", "price_dollars": "0.40", "delta_fp": "1"},
            },
        }
    )
    rows.append(
        {
            "receive_ts_ms": base + 9_300,
            "source": "kalshi",
            "event_type": "orderbook_delta",
            "payload": {
                "type": "orderbook_delta",
                "seq": 3,
                "msg": {"market_ticker": "KXBTC15M-TEST", "side": "yes", "price_dollars": "0.40", "delta_fp": "1"},
            },
        }
    )
    # ...then a trade at +10s that would have filled the (now cancelled) bid.
    rows.append(
        {
            "receive_ts_ms": base + 10_000,
            "source": "kalshi",
            "event_type": "trade",
            "payload": {
                "type": "trade",
                "msg": {
                    "market_ticker": "KXBTC15M-TEST",
                    "yes_price_dollars": "0.40",
                    "count_fp": "1000",
                    "taker_book_side": "ask",
                    "ts_ms": base + 10_000,
                },
            },
        }
    )
    result = replay_rows(market, rows, latency_ms=150, maker_fee_multiplier=0.0)
    assert all(fill.ts_ms < base + 9_000 for fill in result.fills)


def test_replay_latency_delays_order_activation(tmp_path):
    path = tmp_path / "events.jsonl"
    _write_latency_fixture(path)
    delayed = replay(path, latency_ms=150, maker_fee_multiplier=0.0)
    instant = replay(path, latency_ms=0, maker_fee_multiplier=0.0)
    # With latency, the +3.1s trade lands before activation; only the +4.1s trade fills.
    assert delayed.fills and delayed.fills[0].ts_ms == 1_800_000_000 * 1000 + 4_100
    assert instant.fills and instant.fills[0].ts_ms == 1_800_000_000 * 1000 + 3_100
    assert delayed.latency_ms == 150


def test_queue_simulator_requires_trades_to_clear_queue_ahead():
    simulator = ConservativeQueueSimulator()
    simulator.replace(
        cancel_ids=[],
        quotes=[KalshiQuote("bid", 0.49, 5, 0.01)],
        bbo=Bbo(),
    )
    assert simulator.trade(ts_ms=1, price=0.49, count=9, taker_book_side="ask", fair_yes=0.5) == []
    fills = simulator.trade(ts_ms=2, price=0.49, count=4, taker_book_side="ask", fair_yes=0.5)
    assert len(fills) == 1
    assert fills[0].count == 3
