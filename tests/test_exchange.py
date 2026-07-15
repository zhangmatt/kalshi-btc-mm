import json

import pytest

from kalshi_mm.exchange import (
    BrtiState,
    KalshiMarket,
    KalshiOrderBook,
    PriceGrid,
    PriceRange,
    build_order,
    event_contract_fee,
    maker_fee_rate,
)


def test_market_and_tapered_price_grid():
    market = KalshiMarket.from_api(
        {
            "ticker": "KXBTC15M-X",
            "event_ticker": "KXBTC15M",
            "title": "BTC price up in next 15 mins?",
            "status": "active",
            "open_time": "2026-07-13T10:45:00Z",
            "close_time": "2026-07-13T11:00:00Z",
            "expected_expiration_time": "2026-07-13T11:05:00Z",
            "floor_strike": 62880.87,
            "price_ranges": [
                {"start": "0", "end": "0.1", "step": "0.001"},
                {"start": "0.1", "end": "0.9", "step": "0.01"},
                {"start": "0.9", "end": "1", "step": "0.001"},
            ],
            "rules_primary": "rule",
            "rules_secondary": "details",
        }
    )
    grid = PriceGrid(market.price_ranges)
    assert market.strike_average == 62880.87
    assert grid.floor(0.0956) == 0.095
    assert grid.ceil(0.0956) == 0.096
    assert grid.floor(0.506) == 0.50
    assert grid.ceil(0.506) == 0.51
    assert grid.step(0.10) == 0.01
    assert grid.step(0.90) == 0.001
    # Taper boundaries: one tick below 0.01 is 0.009 (fine tail), not 0.001.
    assert grid.tick_below(0.01) == 0.009
    assert grid.tick_below(0.5) == 0.49
    assert grid.tick_above(0.9) == 0.901
    assert grid.tick_above(0.49) == 0.50


def test_order_book_applies_yes_scale_snapshot_and_detects_gap():
    book = KalshiOrderBook("TICKER")
    assert book.apply(
        {
            "type": "orderbook_snapshot",
            "seq": 10,
            "msg": {
                "market_ticker": "TICKER",
                "yes_dollars_fp": [["0.49", "10"]],
                "no_dollars_fp": [["0.51", "12"]],
            },
        }
    )
    assert book.snapshot().bid == 0.49
    assert book.snapshot().ask == 0.51
    assert book.sequence == 10
    assert book.apply(
        {
            "type": "orderbook_delta",
            "seq": 11,
            "msg": {"market_ticker": "TICKER", "side": "yes", "price_dollars": "0.50", "delta_fp": "3"},
        }
    )
    assert book.snapshot().bid == 0.50
    assert book.sequence == 11
    assert not book.apply(
        {
            "type": "orderbook_delta",
            "seq": 13,
            "msg": {"market_ticker": "TICKER", "side": "yes", "price_dollars": "0.50", "delta_fp": "1"},
        }
    )
    assert not book.snapshot().valid


def test_brti_state_retains_partial_average_when_field_is_omitted():
    state = BrtiState()
    state.apply(
        {
            "type": "cfbenchmarks_value",
            "msg": {
                "index_id": "BRTI",
                "data": json.dumps({"time": 1, "value": "100.0"}),
                "last_60s_windowed_average_15min": {"value": "100.2", "window_size": 10},
            },
        }
    )
    # A tick without the field must not wipe observed final-minute state.
    state.apply(
        {
            "type": "cfbenchmarks_value",
            "msg": {"index_id": "BRTI", "data": json.dumps({"time": 2, "value": "100.1"})},
        }
    )
    assert state.final_minute_count == 10
    assert state.final_minute_average == 100.2


def test_brti_state_uses_published_partial_average_and_count():
    state = BrtiState()
    state.apply(
        {
            "type": "cfbenchmarks_value",
            "msg": {
                "index_id": "BRTI",
                "data": json.dumps({"time": 123, "value": "100.25"}),
                "avg_60s_data": {"value": "100.10"},
                "last_60s_windowed_average_15min": {"value": "100.20", "window_size": 14},
            },
        }
    )
    assert state.value == 100.25
    assert state.final_minute_count == 14
    assert sum(state.final_minute_values) == pytest.approx(14 * 100.20)


def test_fee_and_post_only_order_shape():
    assert event_contract_fee(contracts=100, price=0.5, maker=False) == 1.75
    # Taker fee rounds up to a centicent per the 2026-07-07 schedule formula.
    assert event_contract_fee(contracts=1, price=0.35, maker=False) == pytest.approx(0.016)
    # KXBTC15M maker multiplier defaults to 0 (absent from the Non-Standard table).
    assert event_contract_fee(contracts=100, price=0.5, maker=True) == 0.0
    # Series that do charge maker fees use the 0.0175 rate.
    assert event_contract_fee(contracts=100, price=0.5, maker=True, maker_multiplier=1.0) == pytest.approx(0.4375)
    assert maker_fee_rate(price=0.5) == 0.0
    assert maker_fee_rate(price=0.5, multiplier=1.0) == pytest.approx(0.004375)
    order = build_order(ticker="T", side="bid", price=0.5, count=5, expiration_time=123)
    assert order["post_only"] is True
    assert order["cancel_order_on_pause"] is True
    assert order["self_trade_prevention_type"] == "taker_at_cross"
