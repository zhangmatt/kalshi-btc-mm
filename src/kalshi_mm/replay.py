from __future__ import annotations

import argparse
import bisect
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from .exchange import BrtiState, KalshiMarket, KalshiOrderBook, PriceGrid, event_contract_fee
from .strategy import (
    KalshiInventory,
    KalshiMakerStrategy,
    KalshiQuotePlan,
    KalshiStrategyConfig,
    RestingKalshiOrder,
)

# Mirrors the live runner's stale-data gate and the feature store's sampling
# cadence; keep in sync with KalshiStrategyConfig.stale_after_ms and
# features.EMIT_INTERVAL_MS.
STALE_AFTER_MS = 2_000
DELTA_SAMPLE_MS = 250
# Decisions are re-evaluated at most this often in event time. The live loop
# cannot decide per book delta either (~50Hz ceiling); deciding on all ~200k
# events per window is both unrealistic and ~20x slower (each decision prices
# the O(60^2) settlement model).
DECISION_INTERVAL_MS = 100
# A markout horizon is only meaningful if a mid exists reasonably soon after
# it; past this slack the "5s" label would silently span a data gap.
MARKOUT_MAX_SLACK_MS = 2_000
from .recorder import iter_events
from .fair_value import kalshi_btc15m_fair_value
from .settlement import infer_settlement
from .volatility import MultiHorizonVolatility


@dataclass
class SimulatedOrder:
    order_id: str
    side: str
    price: float
    remaining: float
    queue_ahead: float


@dataclass(frozen=True)
class ReplayFill:
    ts_ms: int
    side: str
    price: float
    count: float
    fair_yes: float


class ConservativeQueueSimulator:
    """Fill only after observed aggressive trades exhaust displayed queue ahead."""

    def __init__(self):
        self.orders: dict[str, SimulatedOrder] = {}
        self._next_id = 1

    def replace(
        self,
        *,
        cancel_ids: Iterable[str],
        quotes: Iterable[Any],
        bbo: Any,
        book: Optional[KalshiOrderBook] = None,
    ) -> None:
        for order_id in cancel_ids:
            self.orders.pop(order_id, None)
        for quote in quotes:
            order_id = f"sim-{self._next_id}"
            self._next_id += 1
            if book is not None:
                queue = book.level_size(quote.side, quote.price)
            else:
                queue = bbo.bid_size if quote.side == "bid" and quote.price == bbo.bid else 0.0
                queue = bbo.ask_size if quote.side == "ask" and quote.price == bbo.ask else queue
            self.orders[order_id] = SimulatedOrder(order_id, quote.side, quote.price, quote.count, queue)

    def resting(self) -> list[RestingKalshiOrder]:
        return [RestingKalshiOrder(value.order_id, value.order_id, value.side, value.price, value.remaining) for value in self.orders.values()]

    def trade(self, *, ts_ms: int, price: float, count: float, taker_book_side: str, fair_yes: float) -> list[ReplayFill]:
        fills: list[ReplayFill] = []
        maker_side = "bid" if taker_book_side == "ask" else "ask"
        remaining_trade = count
        for order in list(self.orders.values()):
            marketable = (maker_side == "bid" and price <= order.price) or (maker_side == "ask" and price >= order.price)
            if order.side != maker_side or not marketable or remaining_trade <= 0:
                continue
            consumed = min(order.queue_ahead, remaining_trade)
            order.queue_ahead -= consumed
            remaining_trade -= consumed
            if order.queue_ahead > 0 or remaining_trade <= 0:
                continue
            filled = min(order.remaining, remaining_trade)
            order.remaining -= filled
            remaining_trade -= filled
            fills.append(ReplayFill(ts_ms, order.side, order.price, filled, fair_yes))
            if order.remaining <= 1e-9:
                self.orders.pop(order.order_id, None)
        return fills


@dataclass(frozen=True)
class ReplayResult:
    decisions: int
    orders_posted: int
    contracts_posted: float
    fills: tuple[ReplayFill, ...]
    contracts_filled: float
    fill_rate: float
    final_position: float
    cash_flow: float
    fees_paid: float
    latency_ms: int
    maker_fee_multiplier: float
    gross_fair_edge: float
    markout_1s: Optional[float]
    markout_5s: Optional[float]
    settlement_pnl: Optional[float]
    settlement_result: Optional[str]
    settlement_source: Optional[str]
    settlement_reference: Optional[float]
    fair_mid_mae: Optional[float]
    mids: tuple[tuple[int, float], ...] = ()


def _markout(fills: list[ReplayFill], mids: list[tuple[int, float]], horizon_ms: int) -> Optional[float]:
    if not fills or not mids:
        return None
    timestamps = [ts_ms for ts_ms, _ in mids]
    total = 0.0
    contracts = 0.0
    for fill in fills:
        index = bisect.bisect_left(timestamps, fill.ts_ms + horizon_ms)
        if index >= len(mids):
            continue
        # Bound the match: without this, a data gap turns a "5s" markout into
        # a much longer one measured at whatever mid appears after the gap.
        if mids[index][0] > fill.ts_ms + horizon_ms + MARKOUT_MAX_SLACK_MS:
            continue
        future_mid = mids[index][1]
        total += (future_mid - fill.price if fill.side == "bid" else fill.price - future_mid) * fill.count
        contracts += fill.count
    return total / contracts if contracts > 0 else None


@dataclass(frozen=True)
class QuoteContext:
    """State handed to a QuoteAdjuster before each strategy decision."""

    ts_ms: int
    seconds_to_close: float
    fair_yes: float
    bbo: Any
    book: KalshiOrderBook
    flow_5s: float
    flow_1s: float
    trades_5s: int = 0
    bid_size_delta: float = 0.0
    ask_size_delta: float = 0.0
    ext_micro_lead_bps: Optional[float] = None
    ext_imbalance: Optional[float] = None


class QuoteAdjuster:
    """Pluggable quoting variant: adjust fair value and per-side toxicity.

    The baseline (identity) adjuster reproduces plain settlement-model quoting.
    Variants used by the A/B harness subclass or configure this.
    """

    name = "baseline"

    def adjust(self, ctx: QuoteContext) -> tuple[float, float, float]:
        return ctx.fair_yes, 0.0, 0.0


def extract_brti_ticks(rows: Iterable[Any]) -> list[tuple[int, float]]:
    """BRTI (ts_ms, value) observations from recorded rows, for volatility prewarm."""
    state = BrtiState()
    ticks: list[tuple[int, float]] = []
    for row in rows:
        payload = row.get("payload") or {}
        if row.get("source") == "kalshi" and state.apply(payload) and state.value is not None:
            ticks.append((state.source_ts_ms or int(row["receive_ts_ms"]), state.value))
    return ticks


def replay(
    path: str | Path,
    *,
    min_edge_dollars: float = 0.01,
    latency_ms: int = 150,
    maker_fee_multiplier: float = 0.0,
    prewarm_paths: tuple[str | Path, ...] = (),
    adjuster: Optional[QuoteAdjuster] = None,
) -> ReplayResult:
    """Queue-aware replay of one recording file. See replay_rows for semantics."""
    rows = list(iter_events(path))
    metadata = next((row["payload"] for row in rows if row["event_type"] == "market_metadata"), None)
    if metadata is None:
        raise ValueError("recording has no market_metadata event")
    market = KalshiMarket.from_api(metadata)
    prewarm: list[tuple[int, float]] = []
    for prewarm_path in prewarm_paths:
        prewarm.extend(extract_brti_ticks(iter_events(prewarm_path)))
    return replay_rows(
        market,
        rows,
        min_edge_dollars=min_edge_dollars,
        latency_ms=latency_ms,
        maker_fee_multiplier=maker_fee_multiplier,
        prewarm_ticks=prewarm,
        adjuster=adjuster,
    )


def replay_rows(
    market: KalshiMarket,
    rows: list[Any],
    *,
    min_edge_dollars: float = 0.01,
    latency_ms: int = 150,
    maker_fee_multiplier: float = 0.0,
    prewarm_ticks: list[tuple[int, float]] = [],
    adjuster: Optional[QuoteAdjuster] = None,
) -> ReplayResult:
    """Queue-aware replay with a feed-to-matching-engine latency model.

    Every plan the strategy produces activates latency_ms after the decision:
    new orders join the queue against the book as of activation, and cancelled
    orders remain fillable until the cancel activates. An optional QuoteAdjuster
    implements quoting variants (skew, toxicity gating) for A/B comparison.
    """
    if latency_ms < 0:
        raise ValueError("latency_ms cannot be negative")
    settlement = infer_settlement(rows, market)
    book = KalshiOrderBook(market.ticker)
    brti = BrtiState()
    vol = MultiHorizonVolatility()
    for ts_ms, value in prewarm_ticks:
        vol.update(ts_ms, value)
    strategy = KalshiMakerStrategy(
        price_grid=PriceGrid(market.price_ranges),
        config=KalshiStrategyConfig(
            min_edge_dollars=min_edge_dollars,
            maker_fee_multiplier=maker_fee_multiplier,
        ),
    )
    trade_flow: deque[tuple[int, float]] = deque()
    external_books: dict[str, tuple[int, Optional[float], Optional[float]]] = {}
    prev_bid_size: Optional[float] = None
    prev_ask_size: Optional[float] = None
    last_delta_sample_ms = 0
    last_decision_ms = 0
    simulator = ConservativeQueueSimulator()
    inventory = KalshiInventory()
    fills: list[ReplayFill] = []
    fair_mid_diffs: list[float] = []
    mids: list[tuple[int, float]] = []
    decisions = 0
    orders_posted = 0
    contracts_posted = 0.0
    last_fair = 0.5
    cash_flow = 0.0
    fees_paid = 0.0
    gross_fair_edge = 0.0
    pending_plan: Optional[tuple[int, Any]] = None

    for row in rows:
        ts_ms = int(row["receive_ts_ms"])
        payload = row["payload"]
        source = row.get("source")
        if source in {"binance", "coinbase", "kraken"}:
            if adjuster is not None and row.get("event_type") == "external_orderbook":
                try:
                    external_books[source] = (
                        ts_ms,
                        float(payload["microprice"]) if payload.get("microprice") is not None else None,
                        float(payload["depth_imbalance"]) if payload.get("depth_imbalance") is not None else None,
                    )
                except (TypeError, ValueError, KeyError):
                    pass
            continue
        if source != "kalshi":
            continue
        book.apply(payload, use_yes_price=True)
        if brti.apply(payload) and brti.value is not None:
            vol.update(brti.source_ts_ms or ts_ms, brti.value)
        if ts_ms / 1000.0 > market.close_ts:
            continue
        bbo = book.snapshot()

        if pending_plan is not None and ts_ms >= pending_plan[0]:
            plan = pending_plan[1]
            # Post-only validation at activation: the engine rejects orders
            # whose price crossed during the latency window, and an invalid
            # book means live could not have safely rested anything.
            accepted = []
            if bbo.valid and bbo.bid is not None and bbo.ask is not None:
                for quote in plan.post:
                    if quote.side == "bid" and quote.price < bbo.ask:
                        accepted.append(quote)
                    elif quote.side == "ask" and quote.price > bbo.bid:
                        accepted.append(quote)
            simulator.replace(
                cancel_ids=plan.cancel_ids,
                quotes=accepted,
                bbo=bbo,
                book=book if bbo.valid else None,
            )
            orders_posted += len(accepted)
            contracts_posted += sum(quote.count for quote in accepted)
            pending_plan = None

        forecast = vol.forecast()
        # Mirror the live runner's gates: an invalid book or stale BRTI makes
        # pricing unavailable, and live safety-cancels all resting orders.
        brti_fresh = (
            brti.value is not None
            and brti.source_ts_ms is not None
            and ts_ms - brti.source_ts_ms <= STALE_AFTER_MS
        )
        # Mids record at full resolution (markout fidelity is cheap)…
        if bbo.valid and bbo.bid is not None and bbo.ask is not None:
            full_mid = (bbo.bid + bbo.ask) / 2.0
            if not mids or mids[-1] != (ts_ms, full_mid):
                mids.append((ts_ms, full_mid))

        pricing_ok = bbo.valid and brti_fresh and forecast is not None
        if not pricing_ok:
            if pending_plan is None and simulator.orders:
                cancel_all = KalshiQuotePlan(
                    tuple(simulator.orders.keys()), (), "pricing unavailable"
                )
                pending_plan = (ts_ms + latency_ms, cancel_all)
        elif ts_ms - last_decision_ms >= DECISION_INTERVAL_MS:
            # …while pricing + decisions run on the live loop's cadence.
            last_decision_ms = ts_ms
            seconds_to_close = max(0.0, market.close_ts - ts_ms / 1000.0)
            observed = brti.final_minute_values if seconds_to_close <= 60.0 else ()
            fair = kalshi_btc15m_fair_value(
                strike_average=market.strike_average,
                spot=brti.value,
                sigma_per_s=forecast.sigma_per_s,
                seconds_to_close=seconds_to_close,
                observed_final_values=observed,
            )
            if bbo.bid is not None and bbo.ask is not None:
                fair_mid_diffs.append(abs(fair.yes - (bbo.bid + bbo.ask) / 2.0))
            # While a cancel/replace is in flight, live execution cannot issue another.
            if pending_plan is None:
                fair_yes = fair.yes
                bid_tox = ask_tox = 0.0
                if adjuster is not None:
                    micro_leads = [
                        (micro - brti.value) / brti.value * 10_000
                        for venue_ts, micro, _ in external_books.values()
                        if micro is not None and ts_ms - venue_ts <= 2_000
                    ]
                    ext_imbs = [
                        imb
                        for venue_ts, _, imb in external_books.values()
                        if imb is not None and ts_ms - venue_ts <= 2_000
                    ]
                    context = QuoteContext(
                        ts_ms=ts_ms,
                        seconds_to_close=seconds_to_close,
                        fair_yes=fair.yes,
                        bbo=bbo,
                        book=book,
                        flow_5s=sum(count for _, count in trade_flow),
                        flow_1s=sum(count for flow_ts, count in trade_flow if flow_ts >= ts_ms - 1_000),
                        trades_5s=len(trade_flow),
                        bid_size_delta=bbo.bid_size - prev_bid_size if prev_bid_size is not None else 0.0,
                        ask_size_delta=bbo.ask_size - prev_ask_size if prev_ask_size is not None else 0.0,
                        ext_micro_lead_bps=sorted(micro_leads)[len(micro_leads) // 2] if micro_leads else None,
                        ext_imbalance=sorted(ext_imbs)[len(ext_imbs) // 2] if ext_imbs else None,
                    )
                    # Deltas sample on the feature store's cadence so training
                    # and consumption see the same timescale.
                    if last_delta_sample_ms == 0 or ts_ms - last_delta_sample_ms >= DELTA_SAMPLE_MS:
                        prev_bid_size, prev_ask_size = bbo.bid_size, bbo.ask_size
                        last_delta_sample_ms = ts_ms
                    fair_yes, bid_tox, ask_tox = adjuster.adjust(context)
                    fair_yes = min(0.999, max(0.001, fair_yes))
                # The fair attached to fills is the one orders were quoted at.
                last_fair = fair_yes
                decision = strategy.decide(
                    fair_yes=fair_yes,
                    bbo=bbo,
                    inventory=inventory,
                    current_orders=simulator.resting(),
                    seconds_to_close=seconds_to_close,
                    data_age_ms=0,
                    bid_toxicity_dollars=bid_tox,
                    ask_toxicity_dollars=ask_tox,
                )
                if decision.plan.changed:
                    pending_plan = (ts_ms + latency_ms, decision.plan)
                decisions += 1

        if payload.get("type") == "trade":
            msg = payload.get("msg") or {}
            if msg.get("market_ticker") != market.ticker:
                continue
            price_raw = msg.get("yes_price_dollars") or msg.get("price_dollars")
            count_raw = msg.get("count_fp") or msg.get("count")
            taker_side = msg.get("taker_book_side") or msg.get("book_side")
            if price_raw is not None and count_raw is not None and taker_side in {"bid", "ask"}:
                trade_flow.append((ts_ms, float(count_raw) if taker_side == "bid" else -float(count_raw)))
                while trade_flow and trade_flow[0][0] < ts_ms - 5_000:
                    trade_flow.popleft()
                new_fills = simulator.trade(
                    ts_ms=ts_ms,
                    price=float(price_raw),
                    count=float(count_raw),
                    taker_book_side=str(taker_side),
                    fair_yes=last_fair,
                )
                fills.extend(new_fills)
                for fill in new_fills:
                    delta = fill.count if fill.side == "bid" else -fill.count
                    inventory = KalshiInventory(position=inventory.position + delta)
                    fee = event_contract_fee(
                        contracts=fill.count,
                        price=fill.price,
                        maker=True,
                        maker_multiplier=maker_fee_multiplier,
                    )
                    fees_paid += fee
                    cash_flow += (-fill.price * fill.count if fill.side == "bid" else fill.price * fill.count) - fee
                    gross_fair_edge += (
                        (fill.fair_yes - fill.price) if fill.side == "bid" else (fill.price - fill.fair_yes)
                    ) * fill.count

    mae = sum(fair_mid_diffs) / len(fair_mid_diffs) if fair_mid_diffs else None
    contracts_filled = sum(fill.count for fill in fills)
    fill_rate = contracts_filled / contracts_posted if contracts_posted > 0 else 0.0
    settlement_pnl = None
    if settlement is not None:
        settlement_pnl = cash_flow + inventory.position * (1.0 if settlement.result == "yes" else 0.0)
    return ReplayResult(
        decisions=decisions,
        orders_posted=orders_posted,
        contracts_posted=contracts_posted,
        fills=tuple(fills),
        contracts_filled=contracts_filled,
        fill_rate=fill_rate,
        final_position=inventory.position,
        cash_flow=cash_flow,
        fees_paid=fees_paid,
        latency_ms=latency_ms,
        maker_fee_multiplier=maker_fee_multiplier,
        gross_fair_edge=gross_fair_edge,
        markout_1s=_markout(fills, mids, 1_000),
        markout_5s=_markout(fills, mids, 5_000),
        settlement_pnl=settlement_pnl,
        settlement_result=settlement.result if settlement else None,
        settlement_source=settlement.source if settlement else None,
        settlement_reference=settlement.reference_value if settlement else None,
        fair_mid_mae=mae,
        mids=tuple(mids),
    )


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Replay a Kalshi BTC 15m recording conservatively.")
    parser.add_argument("recording")
    parser.add_argument("--min-edge", type=float, default=0.01)
    parser.add_argument("--latency-ms", type=int, default=150, help="decision-to-matching-engine latency")
    parser.add_argument(
        "--maker-fee-multiplier",
        type=float,
        default=0.0,
        help="KXBTC15M charges no maker fee (verified 2026-07-07 schedule); set 1 for series that do",
    )
    parser.add_argument(
        "--prewarm",
        action="append",
        default=[],
        help="prior recording(s) whose BRTI warms the volatility estimator",
    )
    args = parser.parse_args(argv)
    result = replay(
        args.recording,
        min_edge_dollars=args.min_edge,
        latency_ms=args.latency_ms,
        maker_fee_multiplier=args.maker_fee_multiplier,
        prewarm_paths=tuple(args.prewarm),
    )
    print(f"latency_ms={result.latency_ms} maker_fee_multiplier={result.maker_fee_multiplier}")
    print(f"decisions={result.decisions}")
    print(f"orders_posted={result.orders_posted}")
    print(f"fills={len(result.fills)}")
    print(f"contracts_posted={result.contracts_posted:.2f}")
    print(f"contracts_filled={result.contracts_filled:.2f}")
    print(f"fill_rate={result.fill_rate:.6f}")
    print(f"final_position={result.final_position:.2f}")
    print(f"cash_flow={result.cash_flow:.4f}")
    print(f"fees_paid={result.fees_paid:.4f}")
    print(f"gross_fair_edge={result.gross_fair_edge:.4f}")
    print(f"markout_1s={result.markout_1s if result.markout_1s is not None else 'n/a'}")
    print(f"markout_5s={result.markout_5s if result.markout_5s is not None else 'n/a'}")
    print(f"settlement_result={result.settlement_result or 'n/a'}")
    print(f"settlement_source={result.settlement_source or 'n/a'}")
    print(
        f"settlement_reference="
        f"{result.settlement_reference if result.settlement_reference is not None else 'n/a'}"
    )
    print(f"settlement_pnl={result.settlement_pnl if result.settlement_pnl is not None else 'n/a'}")
    print(f"fair_mid_mae={result.fair_mid_mae if result.fair_mid_mae is not None else 'n/a'}")


if __name__ == "__main__":
    main()
