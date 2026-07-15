from kalshi_mm.exchange import KalshiBbo, PriceGrid, PriceRange
from kalshi_mm.strategy import (
    KalshiInventory,
    KalshiMakerStrategy,
    KalshiStrategyConfig,
    RestingKalshiOrder,
)


def strategy(**overrides):
    config = dict(base_count=5, max_abs_position=50, min_edge_dollars=0.01, maker_fee_multiplier=0.0)
    config.update(overrides)
    return KalshiMakerStrategy(
        price_grid=PriceGrid([PriceRange(0, 1, 0.01)]),
        config=KalshiStrategyConfig(**config),
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


def test_maker_fee_shrinks_quoted_edge():
    # With the standard maker rate, one tick of edge no longer clears a one-cent
    # minimum edge, so the tight quotes disappear.
    decision = strategy(maker_fee_multiplier=1.0).decide(
        fair_yes=0.50,
        bbo=KalshiBbo(0.49, 0.51, valid=True),
        inventory=KalshiInventory(),
        current_orders=[],
        seconds_to_close=600,
        data_age_ms=10,
    )
    assert decision.plan.post == ()


def test_partial_fill_keeps_queue_priority():
    current = [
        RestingKalshiOrder("b", "b", "bid", 0.49, 3.2),  # partially filled from 5
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


def test_depleted_order_is_topped_up():
    current = [
        RestingKalshiOrder("b", "b", "bid", 0.49, 0.5),  # below half of desired 5
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
    assert decision.plan.cancel_ids == ("b",)
    assert [quote.side for quote in decision.plan.post] == ["bid"]


def test_sub_minimum_inventory_room_is_not_quoted():
    # Long 49.5 of 50: only 0.5 contracts of bid room — below min_order_count,
    # so no bid is posted (and no cancel/repost churn loop can start).
    decision = strategy().decide(
        fair_yes=0.50,
        bbo=KalshiBbo(0.45, 0.55, valid=True),
        inventory=KalshiInventory(position=49.5),
        current_orders=[],
        seconds_to_close=600,
        data_age_ms=10,
    )
    assert all(quote.side != "bid" for quote in decision.plan.post)
    assert all(quote.count >= 1.0 for quote in decision.plan.post)


def test_resting_order_matching_its_own_target_is_kept():
    # A resting order equal to a small (but valid) target must never fail the
    # keep floor, even when the fraction-based floor would exceed the target.
    current = [RestingKalshiOrder("b", "b", "bid", 0.49, 1.0)]
    decision = strategy(base_count=1.0).decide(
        fair_yes=0.50,
        bbo=KalshiBbo(0.49, 0.51, valid=True),
        inventory=KalshiInventory(),
        current_orders=current,
        seconds_to_close=600,
        data_age_ms=10,
    )
    assert "b" not in decision.plan.cancel_ids


def test_duplicate_same_side_orders_are_all_cancelled_but_one():
    # Two resting bids (restart / presumed-failed post that landed): exactly
    # one may survive; the shadowed one must be cancelled, not forgotten.
    current = [
        RestingKalshiOrder("b1", "b1", "bid", 0.49, 5),
        RestingKalshiOrder("b2", "b2", "bid", 0.48, 5),
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
    assert "b2" in decision.plan.cancel_ids  # wrong price -> cancelled
    assert "b1" not in decision.plan.cancel_ids  # matches target -> kept
    # Duplicates at the SAME correct price: keep one, cancel the rest.
    dupes = [
        RestingKalshiOrder("b1", "b1", "bid", 0.49, 5),
        RestingKalshiOrder("b2", "b2", "bid", 0.49, 5),
    ]
    decision = strategy().decide(
        fair_yes=0.50,
        bbo=KalshiBbo(0.49, 0.51, valid=True),
        inventory=KalshiInventory(),
        current_orders=dupes,
        seconds_to_close=600,
        data_age_ms=10,
    )
    assert len(set(decision.plan.cancel_ids) & {"b1", "b2"}) == 1


def test_zero_min_edge_is_rejected():
    import pytest

    with pytest.raises(ValueError, match="min_edge"):
        KalshiStrategyConfig(min_edge_dollars=0.0)


def test_available_cash_caps_combined_collateral():
    maker = KalshiMakerStrategy(
        price_grid=PriceGrid([PriceRange(0, 1, 0.01)]),
        config=KalshiStrategyConfig(
            cash_reserve_dollars=5.0, max_quote_notional_dollars=100.0, maker_fee_multiplier=0.0
        ),
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
