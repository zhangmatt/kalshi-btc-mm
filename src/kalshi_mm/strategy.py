from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

from .exchange import KalshiBbo, PriceGrid, maker_fee_rate


@dataclass(frozen=True)
class KalshiInventory:
    position: float = 0.0
    cash_dollars: Optional[float] = None


@dataclass(frozen=True)
class KalshiQuote:
    side: str
    price: float
    count: float
    expected_edge: float


@dataclass(frozen=True)
class RestingKalshiOrder:
    order_id: str
    client_order_id: str
    side: str
    price: float
    remaining_count: float


@dataclass(frozen=True)
class KalshiQuotePlan:
    cancel_ids: tuple[str, ...]
    post: tuple[KalshiQuote, ...]
    reason: str

    @property
    def changed(self) -> bool:
        return bool(self.cancel_ids or self.post)


@dataclass(frozen=True)
class KalshiStrategyConfig:
    base_count: float = 5.0
    max_abs_position: float = 50.0
    max_open_count: float = 20.0
    min_edge_dollars: float = 0.01
    inventory_skew_dollars: float = 0.02
    stale_after_ms: int = 2_000
    stop_quote_before_close_s: float = 90.0
    # Conservative default: assume the standard 0.0175 maker rate applies to this
    # series until verified against the fee schedule or a real fill.
    maker_fee_multiplier: float = 1.0
    max_quote_notional_dollars: float = 10.0
    cash_reserve_dollars: float = 10.0
    min_order_count: float = 1.0
    # Keep a partially filled order (and its queue priority) while its remaining
    # size is at least this fraction of the desired size.
    requote_remaining_fraction: float = 0.5

    def __post_init__(self) -> None:
        if self.base_count <= 0 or self.max_abs_position <= 0 or self.max_open_count <= 0:
            raise ValueError("order and position limits must be positive")
        if self.max_quote_notional_dollars <= 0 or self.cash_reserve_dollars < 0:
            raise ValueError("cash limits are invalid")
        if not 0.0 < self.requote_remaining_fraction <= 1.0:
            raise ValueError("requote_remaining_fraction must be in (0, 1]")


@dataclass(frozen=True)
class KalshiStrategyDecision:
    fair_yes: float
    reservation_yes: float
    plan: KalshiQuotePlan
    reason: str


class KalshiMakerStrategy:
    def __init__(self, *, price_grid: PriceGrid, config: KalshiStrategyConfig = KalshiStrategyConfig()):
        self.price_grid = price_grid
        self.config = config

    def decide(
        self,
        *,
        fair_yes: float,
        bbo: KalshiBbo,
        inventory: KalshiInventory,
        current_orders: Iterable[RestingKalshiOrder],
        seconds_to_close: float,
        data_age_ms: int,
        bid_toxicity_dollars: float = 0.0,
        ask_toxicity_dollars: float = 0.0,
    ) -> KalshiStrategyDecision:
        current = tuple(current_orders)
        if not bbo.valid or bbo.bid is None or bbo.ask is None:
            return self._cancel_all(fair_yes, current, "invalid book")
        if data_age_ms > self.config.stale_after_ms:
            return self._cancel_all(fair_yes, current, "stale market data")
        if seconds_to_close <= self.config.stop_quote_before_close_s:
            return self._cancel_all(fair_yes, current, "close cutoff")

        utilization = max(-1.0, min(1.0, inventory.position / self.config.max_abs_position))
        reservation = max(0.001, min(0.999, fair_yes - utilization * self.config.inventory_skew_dollars))
        desired: list[KalshiQuote] = []

        bid_room = self.config.max_abs_position - inventory.position
        bid_count = min(self.config.base_count * max(0.0, 1.0 - utilization), bid_room)
        if bid_count > 0:
            bid_edge = self.config.min_edge_dollars + max(0.0, bid_toxicity_dollars)
            bid = self.price_grid.floor(reservation - bid_edge)
            bid = min(bid, self.price_grid.floor(bbo.ask - self.price_grid.step(bbo.ask)))
            fee = maker_fee_rate(price=bid, multiplier=self.config.maker_fee_multiplier)
            edge = reservation - bid - fee - max(0.0, bid_toxicity_dollars)
            bid_count = min(bid_count, self.config.max_quote_notional_dollars / max(0.001, bid))
            if 0.0 < bid < bbo.ask and edge >= self.config.min_edge_dollars - 1e-9:
                desired.append(KalshiQuote("bid", bid, min(bid_count, self.config.max_open_count), edge))

        ask_room = self.config.max_abs_position + inventory.position
        ask_count = min(self.config.base_count * max(0.0, 1.0 + utilization), ask_room)
        if ask_count > 0:
            ask_edge = self.config.min_edge_dollars + max(0.0, ask_toxicity_dollars)
            ask = self.price_grid.ceil(reservation + ask_edge)
            ask = max(ask, self.price_grid.ceil(bbo.bid + self.price_grid.step(bbo.bid)))
            fee = maker_fee_rate(price=ask, multiplier=self.config.maker_fee_multiplier)
            edge = ask - reservation - fee - max(0.0, ask_toxicity_dollars)
            ask_collateral = max(0.001, 1.0 - ask)
            ask_count = min(ask_count, self.config.max_quote_notional_dollars / ask_collateral)
            if bbo.bid < ask < 1.0 and edge >= self.config.min_edge_dollars - 1e-9:
                desired.append(KalshiQuote("ask", ask, min(ask_count, self.config.max_open_count), edge))

        # Sub-minimum quotes (e.g. inventory-room dust near the position cap) are
        # never worth posting and would churn against the keep floor forever.
        desired = [quote for quote in desired if quote.count >= self.config.min_order_count]

        if inventory.cash_dollars is not None and desired:
            available = max(0.0, inventory.cash_dollars - self.config.cash_reserve_dollars)
            required = sum(
                quote.count * (quote.price if quote.side == "bid" else 1.0 - quote.price)
                for quote in desired
            )
            scale = min(1.0, available / required) if required > 0 else 0.0
            desired = [
                KalshiQuote(
                    quote.side,
                    quote.price,
                    math.floor(quote.count * scale * 100.0) / 100.0,
                    quote.expected_edge,
                )
                for quote in desired
            ]
            desired = [quote for quote in desired if quote.count >= self.config.min_order_count]

        plan = self._plan(current, desired)
        return KalshiStrategyDecision(fair_yes, reservation, plan, "quoting")

    def _plan(self, current: tuple[RestingKalshiOrder, ...], desired: list[KalshiQuote]) -> KalshiQuotePlan:
        desired_by_side = {quote.side: quote for quote in desired}
        current_by_side = {order.side: order for order in current}
        cancel: list[str] = []
        post: list[KalshiQuote] = []
        for side, order in current_by_side.items():
            target = desired_by_side.get(side)
            if target is None or abs(target.price - order.price) > 1e-9:
                cancel.append(order.order_id)
                continue
            # Cancel/repost forfeits queue priority, so tolerate a partially
            # filled or slightly oversized resting order at the right price.
            # The floor is clamped by the target so a freshly posted order can
            # always satisfy it (otherwise a sub-floor target churns forever).
            keep_floor = min(
                target.count,
                max(self.config.min_order_count, self.config.requote_remaining_fraction * target.count),
            )
            if order.remaining_count + 1e-9 < keep_floor:
                cancel.append(order.order_id)
        for side, target in desired_by_side.items():
            order = current_by_side.get(side)
            if order is None or order.order_id in cancel:
                post.append(target)
        reason = "quote changed" if cancel or post else "unchanged"
        return KalshiQuotePlan(tuple(cancel), tuple(post), reason)

    @staticmethod
    def _cancel_all(
        fair_yes: float,
        current: tuple[RestingKalshiOrder, ...],
        reason: str,
    ) -> KalshiStrategyDecision:
        return KalshiStrategyDecision(
            fair_yes=fair_yes,
            reservation_yes=fair_yes,
            plan=KalshiQuotePlan(tuple(order.order_id for order in current), (), reason),
            reason=reason,
        )
