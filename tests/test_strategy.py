from kalshi_mm.exchange import KalshiBbo, PriceGrid, PriceRange
from kalshi_mm.strategy import (
    KalshiInventory,
    KalshiMakerStrategy,
    KalshiStrategyConfig,
    RestingKalshiOrder,
)


def strategy():
    return KalshiMakerStrategy(
        price_grid=PriceGrid([PriceRange(0, 1, 0.01)]),
        config=KalshiStrategyConfig(base_count=5, max_abs_position=50, min_edge_dollars=0.01),
    )


def test_strategy_quotes_two_sided_post_only_prices():
    decision = strategy().decide(
        fair_yes=0.50,
        bbo=KalshiBbo(0.49, 0.51, valid=True),
        inventory=KalshiInventory(),
        current_orders=[],
        seconds_to_close=600,
        data_age_ms=10,
    )
    assert [(quote.side, quote.price) for quote in decision.plan.post] == [("bid", 0.49), ("ask", 0.51)]


def test_inventory_skews_prices_and_sizes_toward_reduction():
    decision = strategy().decide(
        fair_yes=0.50,
        bbo=KalshiBbo(0.45, 0.55, valid=True),
        inventory=KalshiInventory(position=25),
        current_orders=[],
        seconds_to_close=600,
        data_age_ms=10,
    )
    quotes = {quote.side: quote for quote in decision.plan.post}
    assert decision.reservation_yes < 0.50
    assert quotes["ask"].count > quotes["bid"].count


def test_unchanged_orders_keep_queue_priority():
    current = [
        RestingKalshiOrder("b", "b", "bid", 0.49, 5),
        RestingKalshiOrder("a", "a", "ask", 0.51, 5),
    ]
    decision = strategy().decide(
        fair_yes=0.50,
        bbo=KalshiBbo(0.49, 0.51, valid=True),
        inventory=KalshiInventory(),
        current_orders=current,
        seconds_to_close=600,
        data_age_ms=10,
    )
    assert not decision.plan.changed


def test_close_cutoff_cancels_orders():
    order = RestingKalshiOrder("b", "b", "bid", 0.49, 5)
    decision = strategy().decide(
        fair_yes=0.50,
        bbo=KalshiBbo(0.49, 0.51, valid=True),
        inventory=KalshiInventory(),
        current_orders=[order],
        seconds_to_close=30,
        data_age_ms=10,
    )
    assert decision.plan.cancel_ids == ("b",)
    assert decision.plan.post == ()


def test_available_cash_caps_combined_collateral():
    maker = KalshiMakerStrategy(
        price_grid=PriceGrid([PriceRange(0, 1, 0.01)]),
        config=KalshiStrategyConfig(cash_reserve_dollars=5.0, max_quote_notional_dollars=100.0),
    )
    decision = maker.decide(
        fair_yes=0.5,
        bbo=KalshiBbo(0.49, 0.51, valid=True),
        inventory=KalshiInventory(position=0.0, cash_dollars=7.0),
        current_orders=[],
        seconds_to_close=600,
        data_age_ms=0,
    )
    collateral = sum(
        quote.count * (quote.price if quote.side == "bid" else 1.0 - quote.price)
        for quote in decision.plan.post
    )
    assert collateral <= 2.0 + 1e-9
