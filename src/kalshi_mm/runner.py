from __future__ import annotations

import argparse
import threading
import time

import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from .exchange import (
    BrtiState,
    KalshiApiError,
    KalshiOrderBook,
    KalshiRestClient,
    KalshiSigner,
    PriceGrid,
    build_order,
    event_contract_fee,
)
from .collect import _choose_market, _market_payload, schedule_result_finalizer
from .config import KalshiRuntimeConfig
from .strategy import (
    KalshiInventory,
    KalshiMakerStrategy,
    KalshiQuotePlan,
    RestingKalshiOrder,
)
from .websocket import KalshiWebSocket
from .recorder import EventRecorder
from .replay import ConservativeQueueSimulator
from .live_gate import LiveToxicityGate
from .feeds import CompositeReference, reference_streams
from .fair_value import kalshi_btc15m_fair_value
from .volatility import MultiHorizonVolatility

MAX_CONSECUTIVE_EXECUTION_FAILURES = 10
POSITION_DIVERGENCE_TOLERANCE = 1.0
# This market's book updates hundreds of times per second; a multi-second
# silent book on a live BRTI connection means the subscription died.
BOOK_FROZEN_AFTER_MS = 5_000


class PaperExecutor:
    """Paper trading with queue-aware fill simulation against the live trade tape."""

    def __init__(
        self,
        *,
        book: Optional[KalshiOrderBook] = None,
        recorder: Optional[EventRecorder] = None,
        maker_fee_multiplier: float = 1.0,
    ):
        self.book = book
        self.recorder = recorder
        self.maker_fee_multiplier = maker_fee_multiplier
        self.simulator = ConservativeQueueSimulator()
        self.position = 0.0
        self.cash_flow = 0.0
        self.fees_paid = 0.0
        self.fill_count = 0
        self.last_fair_yes = 0.5
        self._lock = threading.Lock()

    def execute(self, plan: KalshiQuotePlan, *, ticker: str, expiration_time: int) -> None:
        with self._lock:
            bbo = self.book.snapshot() if self.book is not None else None
            self.simulator.replace(cancel_ids=plan.cancel_ids, quotes=plan.post, bbo=bbo, book=self.book)

    def resting(self) -> list[RestingKalshiOrder]:
        with self._lock:
            return self.simulator.resting()

    def cancel_all(self) -> None:
        with self._lock:
            self.simulator.orders.clear()

    def on_trade_envelope(self, envelope: Mapping[str, Any], market_ticker: str) -> None:
        if envelope.get("type") != "trade":
            return
        msg = envelope.get("msg") or {}
        if msg.get("market_ticker") != market_ticker:
            return
        price_raw = msg.get("yes_price_dollars") or msg.get("price_dollars")
        count_raw = msg.get("count_fp") or msg.get("count")
        taker_side = msg.get("taker_book_side") or msg.get("book_side")
        if price_raw is None or count_raw is None or taker_side not in {"bid", "ask"}:
            return
        ts_ms = int(msg.get("ts_ms") or time.time_ns() // 1_000_000)
        with self._lock:
            fills = self.simulator.trade(
                ts_ms=ts_ms,
                price=float(price_raw),
                count=float(count_raw),
                taker_book_side=str(taker_side),
                fair_yes=self.last_fair_yes,
            )
            for fill in fills:
                delta = fill.count if fill.side == "bid" else -fill.count
                self.position += delta
                fee = event_contract_fee(
                    contracts=fill.count,
                    price=fill.price,
                    maker=True,
                    maker_multiplier=self.maker_fee_multiplier,
                )
                self.fees_paid += fee
                self.cash_flow += (-fill.price * fill.count if fill.side == "bid" else fill.price * fill.count) - fee
                self.fill_count += 1
                if self.recorder:
                    self.recorder.write(
                        source="strategy",
                        event_type="paper_fill",
                        payload={
                            "side": fill.side,
                            "price": fill.price,
                            "count": fill.count,
                            "fair_yes": fill.fair_yes,
                            "fee": fee,
                            "position": self.position,
                            "cash_flow": self.cash_flow,
                        },
                        exchange_ts_ms=ts_ms,
                    )


class FillTracker:
    """Consume the private fill channel so inventory does not wait on REST polls."""

    def __init__(self, market_ticker: str):
        self.market_ticker = market_ticker
        self._lock = threading.Lock()
        self._synced_position = 0.0
        self._delta_since_sync = 0.0
        self.unparsed_fills = 0

    def on_envelope(self, envelope: Mapping[str, Any]) -> None:
        if envelope.get("type") != "fill":
            return
        msg = envelope.get("msg") or {}
        if msg.get("market_ticker") != self.market_ticker:
            return
        try:
            count = float(msg.get("count_fp") or msg.get("count") or 0)
        except (TypeError, ValueError):
            count = 0.0
        delta = self._yes_delta(msg, count)
        with self._lock:
            if delta is None:
                self.unparsed_fills += 1
            else:
                self._delta_since_sync += delta

    @staticmethod
    def _yes_delta(msg: Mapping[str, Any], count: float) -> Optional[float]:
        if count <= 0:
            return None
        book_side = msg.get("book_side")
        if book_side not in {"bid", "ask"} and msg.get("is_taker"):
            # taker_book_side describes the aggressor; it is our side only when we took.
            book_side = msg.get("taker_book_side")
        if book_side == "bid":
            return count
        if book_side == "ask":
            return -count
        action = str(msg.get("action", "")).lower()
        side = str(msg.get("side", "")).lower()
        if action in {"buy", "sell"} and side in {"yes", "no"}:
            sign = 1.0 if action == "buy" else -1.0
            if side == "no":
                sign = -sign
            return sign * count
        return None

    def position(self) -> float:
        with self._lock:
            return self._synced_position + self._delta_since_sync

    def sync(self, rest_position: float) -> float:
        """Adopt the REST position; return how far the fill-based estimate had drifted."""
        with self._lock:
            divergence = rest_position - (self._synced_position + self._delta_since_sync)
            self._synced_position = rest_position
            self._delta_since_sync = 0.0
            return divergence


class TickerWatch:
    """Cross-check the delta-built book against the independent ticker channel."""

    def __init__(self, market_ticker: str, *, tolerance_ticks: float = 2.0, sustain_s: float = 5.0):
        self.market_ticker = market_ticker
        self.tolerance_ticks = tolerance_ticks
        self.sustain_s = sustain_s
        self._lock = threading.Lock()
        self._bid: Optional[float] = None
        self._ask: Optional[float] = None
        self._ts_ms = 0
        self._diverged_since_ms: Optional[int] = None

    def on_envelope(self, envelope: Mapping[str, Any]) -> None:
        if envelope.get("type") != "ticker":
            return
        msg = envelope.get("msg") or {}
        if msg.get("market_ticker") != self.market_ticker:
            return
        try:
            bid = float(msg["yes_bid_dollars"]) if msg.get("yes_bid_dollars") is not None else None
            ask = float(msg["yes_ask_dollars"]) if msg.get("yes_ask_dollars") is not None else None
        except (TypeError, ValueError):
            return
        with self._lock:
            self._bid = bid
            self._ask = ask
            self._ts_ms = int(msg.get("ts_ms") or time.time_ns() // 1_000_000)

    def divergence_exceeded(self, *, book_bid: float, book_ask: float, tick: float, now_ms: int) -> bool:
        with self._lock:
            if self._bid is None or self._ask is None or now_ms - self._ts_ms > 5_000:
                self._diverged_since_ms = None
                return False
            tolerance = self.tolerance_ticks * max(tick, 1e-9)
            diverged = abs(book_bid - self._bid) > tolerance or abs(book_ask - self._ask) > tolerance
            if not diverged:
                self._diverged_since_ms = None
                return False
            if self._diverged_since_ms is None:
                self._diverged_since_ms = now_ms
                return False
            return now_ms - self._diverged_since_ms >= self.sustain_s * 1_000

    def reset(self) -> None:
        with self._lock:
            self._diverged_since_ms = None


class LiveExecutor:
    def __init__(
        self,
        client: KalshiRestClient,
        *,
        subaccount: int = 0,
        recorder: Optional[EventRecorder] = None,
    ):
        self.client = client
        self.subaccount = subaccount
        self.recorder = recorder

    def _record_ack(self, event_type: str, started_ms: int, responses: list[dict[str, Any]]) -> None:
        if not self.recorder:
            return
        completed_ms = time.time_ns() // 1_000_000
        engine_times = [int(row["ts_ms"]) for row in responses if row.get("ts_ms") is not None]
        self.recorder.write(
            source="execution",
            event_type=event_type,
            payload={
                "request_start_ms": started_ms,
                "response_receive_ms": completed_ms,
                "round_trip_ms": completed_ms - started_ms,
                "matching_engine_ts_ms": max(engine_times) if engine_times else None,
                "matching_engine_to_receive_ms": completed_ms - max(engine_times) if engine_times else None,
                "responses": responses,
            },
            exchange_ts_ms=max(engine_times) if engine_times else None,
            receive_ts_ms=completed_ms,
        )

    def _record_order_errors(self, event_type: str, errors: list[Any]) -> None:
        # Per-order rejections (already-filled cancels, post-only crosses) are
        # normal races; record them and let the next reconcile converge.
        if self.recorder:
            self.recorder.write(source="execution", event_type=event_type, payload={"errors": errors})

    def execute(self, plan: KalshiQuotePlan, *, ticker: str, expiration_time: int) -> list[RestingKalshiOrder]:
        """Execute the plan; return orders created (parsed from the responses).

        Re-reading GET /portfolio/orders immediately after a create is not
        guaranteed read-your-writes consistent — a missing just-posted order
        would be posted again on the next decision. The create responses are
        the authoritative record of what now rests.
        """
        created: list[RestingKalshiOrder] = []
        if plan.cancel_ids:
            started_ms = time.time_ns() // 1_000_000
            responses = self.client.cancel_orders(list(plan.cancel_ids), subaccount=self.subaccount)
            self._record_ack("cancel_ack", started_ms, responses)
            errors = [row.get("error") for row in responses if row.get("error")]
            if errors:
                self._record_order_errors("cancel_order_errors", errors)
        if plan.post:
            orders = [
                build_order(
                    ticker=ticker,
                    side=quote.side,
                    price=quote.price,
                    count=quote.count,
                    expiration_time=expiration_time,
                    subaccount=self.subaccount,
                )
                for quote in plan.post
            ]
            started_ms = time.time_ns() // 1_000_000
            responses = self.client.create_orders(orders)
            self._record_ack("post_ack", started_ms, responses)
            errors = [row.get("error") for row in responses if row.get("error")]
            if errors:
                self._record_order_errors("post_order_errors", errors)
            for sent, row in zip(orders, responses):
                inner = row.get("order") if isinstance(row.get("order"), dict) else row
                if not isinstance(inner, dict):
                    continue
                order_id = inner.get("order_id")
                if row.get("error") or not order_id:
                    continue
                created.append(
                    RestingKalshiOrder(
                        order_id=str(order_id),
                        client_order_id=str(sent["client_order_id"]),
                        side=str(sent["side"]),
                        price=float(sent["price"]),
                        remaining_count=float(sent["count"]),
                    )
                )
        return created

    def resting(self, ticker: str) -> list[RestingKalshiOrder]:
        result = []
        for row in self.client.get_orders(ticker):
            side = row.get("book_side")
            if side not in {"bid", "ask"}:
                action = row.get("action")
                outcome = row.get("outcome_side") or row.get("side")
                side = "bid" if action == "buy" and outcome == "yes" else "ask"
            result.append(
                RestingKalshiOrder(
                    order_id=str(row["order_id"]),
                    client_order_id=str(row.get("client_order_id", "")),
                    side=str(side),
                    price=float(row.get("yes_price_dollars") or row.get("price_dollars")),
                    remaining_count=float(row.get("remaining_count_fp") or row.get("remaining_count") or 0),
                )
            )
        return result

    def cancel_all(self, ticker: str) -> None:
        # The last safety action before exit: retry, because a single failed
        # REST call here means unattended resting orders until GTD expiry.
        last_error: Optional[Exception] = None
        for _ in range(3):
            try:
                orders = self.resting(ticker)
                if orders:
                    self.client.cancel_orders([order.order_id for order in orders], subaccount=self.subaccount)
                return
            except Exception as exc:
                last_error = exc
                time.sleep(1.0)
        if last_error is not None:
            raise last_error


def _inventory(client: KalshiRestClient, ticker: str, subaccount: int) -> KalshiInventory:
    positions = client.get_positions(ticker, subaccount=subaccount)
    position = float(positions[0].get("position_fp", 0)) if positions else 0.0
    balance = client.get_balance(subaccount=subaccount)
    return KalshiInventory(position=position, cash_dollars=float(balance.get("balance", 0)) / 100.0)


def run_one(
    *,
    config: KalshiRuntimeConfig,
    ticker: Optional[str],
    live: bool,
    output: Optional[str],
    verbose: bool,
) -> Path:
    signer = KalshiSigner(config.credentials)
    client = KalshiRestClient(base_url=config.rest_url, signer=signer)
    try:
        client.get_balance(subaccount=config.subaccount)
    except Exception as exc:
        raise RuntimeError(
            "Kalshi authenticated REST preflight failed. Verify that KALSHI_ENV matches the key, "
            "KALSHI_API_KEY_ID is the key ID (not its name), and the PEM belongs to that key."
        ) from exc
    market = _choose_market(client, ticker, wait=ticker is None, verbose=verbose)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = Path(output) if output else Path(config.data_dir) / market.ticker / f"runner_{run_id}.jsonl.gz"
    recorder = EventRecorder(path)
    recorder.write(source="system", event_type="market_metadata", payload=_market_payload(market))

    book = KalshiOrderBook(market.ticker)
    brti = BrtiState()
    brti_vol = MultiHorizonVolatility()
    composite = CompositeReference()
    refs = reference_streams(
        composite,
        recorder,
        insecure_ssl=config.ws_insecure,
        binance_url=config.binance_ws_url,
    )
    price_grid = PriceGrid(market.price_ranges)
    strategy = KalshiMakerStrategy(price_grid=price_grid, config=config.strategy)
    executor: Any = (
        LiveExecutor(client, subaccount=config.subaccount, recorder=recorder)
        if live
        else PaperExecutor(book=book, recorder=recorder, maker_fee_multiplier=config.strategy.maker_fee_multiplier)
    )
    fill_tracker = FillTracker(market.ticker) if live else None
    ticker_watch = TickerWatch(market.ticker)
    toxicity_gate: Optional[LiveToxicityGate] = None
    if config.toxicity_model_path:
        toxicity_gate = LiveToxicityGate(
            config.toxicity_model_path,
            scale_dollars=config.toxicity_scale_dollars,
        )
        print(
            f"[kalshi] toxicity gate ENABLED model={config.toxicity_model_path} "
            f"scale={config.toxicity_scale_dollars}",
            flush=True,
        )

    def on_ws_event(envelope: Mapping[str, Any]) -> None:
        ticker_watch.on_envelope(envelope)
        if fill_tracker is not None:
            fill_tracker.on_envelope(envelope)
        if not live:
            executor.on_trade_envelope(envelope, market.ticker)
        if toxicity_gate is not None and envelope.get("type") == "trade":
            msg = envelope.get("msg") or {}
            if msg.get("market_ticker") == market.ticker:
                count_raw = msg.get("count_fp") or msg.get("count")
                taker = msg.get("taker_book_side") or msg.get("book_side")
                if count_raw is not None and taker in {"bid", "ask"}:
                    ts = int(msg.get("ts_ms") or time.time_ns() // 1_000_000)
                    toxicity_gate.on_trade(ts, str(taker), float(count_raw))

    stream = KalshiWebSocket(
        signer=signer,
        market_ticker=market.ticker,
        order_book=book,
        brti=brti,
        ws_url=config.ws_url,
        recorder=recorder,
        on_event=on_ws_event,
        include_private=live,
        insecure_ssl=config.ws_insecure,
        verbose=verbose,
    )
    inventory = _inventory(client, market.ticker, config.subaccount) if live else KalshiInventory()
    if live and fill_tracker is not None:
        fill_tracker.sync(inventory.position)
    current_orders: list[RestingKalshiOrder] = executor.resting(market.ticker) if live else executor.resting()
    last_position_poll = 0.0
    last_reconcile = 0.0
    last_brti_ts = 0
    last_execute_s = 0.0
    consecutive_execution_failures = 0
    consecutive_divergences = 0
    last_quote_summary: Optional[tuple[Any, ...]] = None

    def execute_plan(plan: KalshiQuotePlan) -> bool:
        nonlocal consecutive_execution_failures, current_orders
        try:
            created = executor.execute(plan, ticker=market.ticker, expiration_time=market.close_ts - 1)
        except (KalshiApiError, requests.RequestException) as exc:
            # Network timeouts are as transient as API errors; neither may
            # kill the loop before the consecutive-failure cap.
            consecutive_execution_failures += 1
            recorder.write(
                source="execution",
                event_type="execution_error",
                payload={"error": str(exc), "consecutive": consecutive_execution_failures},
            )
            if consecutive_execution_failures >= MAX_CONSECUTIVE_EXECUTION_FAILURES:
                raise RuntimeError(f"{consecutive_execution_failures} consecutive execution failures") from exc
            time.sleep(0.5)
            return False
        consecutive_execution_failures = 0
        if live:
            # Seed local order state from the create responses instead of an
            # immediate REST re-read (not read-your-writes consistent; a
            # missing just-posted order would be double-posted next decision).
            cancelled = set(plan.cancel_ids)
            current_orders = [order for order in current_orders if order.order_id not in cancelled]
            current_orders.extend(created or [])
        return True

    print(
        f"[kalshi] mode={'LIVE' if live else 'PAPER'} ticker={market.ticker} "
        f"strike={market.strike_average:.2f} maker_fee_multiplier={config.strategy.maker_fee_multiplier} "
        f"output={path}",
        flush=True,
    )
    if config.ws_insecure:
        print("[kalshi] warning: websocket TLS certificate verification is disabled", flush=True)
    for ref in refs:
        ref.start()
    stream.start()
    try:
        while time.time() < market.close_ts:
            time.sleep(0.02)
            if stream.fatal_error:
                raise RuntimeError(stream.fatal_error)
            now_ms = time.time_ns() // 1_000_000
            now_s = now_ms / 1000.0
            brti_snapshot = brti.snapshot()
            if (
                brti_snapshot.value is not None
                and brti_snapshot.source_ts_ms
                and brti_snapshot.source_ts_ms != last_brti_ts
            ):
                brti_vol.update(brti_snapshot.source_ts_ms, brti_snapshot.value)
                last_brti_ts = brti_snapshot.source_ts_ms

            if live and now_s - last_position_poll >= config.position_poll_s:
                rest_inventory = _inventory(client, market.ticker, config.subaccount)
                divergence = fill_tracker.sync(rest_inventory.position) if fill_tracker else 0.0
                if abs(divergence) > POSITION_DIVERGENCE_TOLERANCE:
                    consecutive_divergences += 1
                    recorder.write(
                        source="execution",
                        event_type="position_divergence",
                        payload={
                            "rest_position": rest_inventory.position,
                            "divergence": divergence,
                            "consecutive": consecutive_divergences,
                        },
                    )
                    if consecutive_divergences >= 2:
                        raise RuntimeError(
                            f"position reconciliation diverged twice consecutively ({divergence:+.2f})"
                        )
                else:
                    consecutive_divergences = 0
                inventory = rest_inventory
                last_position_poll = now_s
            if live and fill_tracker is not None:
                inventory = KalshiInventory(position=fill_tracker.position(), cash_dollars=inventory.cash_dollars)
            if live and now_s - last_reconcile >= config.order_reconcile_s:
                current_orders = executor.resting(market.ticker)
                last_reconcile = now_s
            elif not live:
                inventory = KalshiInventory(position=executor.position)
                current_orders = executor.resting()

            bbo = book.snapshot()
            proxy = composite.snapshot(now_ms)
            # A dead orderbook subscription on a connection kept alive by 1Hz
            # BRTI leaves the book valid-but-frozen; stream liveness alone
            # cannot detect it. Gate on the book's own last update.
            book_age = now_ms - book.last_update_ms if book.last_update_ms else 10**9
            book_frozen = bbo.valid and book_age > BOOK_FROZEN_AFTER_MS
            brti_age = now_ms - (brti_snapshot.source_ts_ms or 0)
            use_brti = brti_snapshot.value is not None and brti_age <= config.strategy.stale_after_ms
            if book_frozen:
                spot = None
                forecast = None
                price_source = "book frozen"
            elif use_brti:
                spot = brti_snapshot.value
                forecast = brti_vol.forecast()
                price_source = "brti"
            elif config.allow_proxy_pricing and not live and proxy is not None:
                spot = proxy.price
                forecast = composite.forecast()
                price_source = "proxy"
            else:
                spot = None
                forecast = None
                price_source = "unavailable"

            if bbo.valid and bbo.bid is not None and bbo.ask is not None:
                if ticker_watch.divergence_exceeded(
                    book_bid=bbo.bid,
                    book_ask=bbo.ask,
                    tick=price_grid.step(bbo.bid),
                    now_ms=now_ms,
                ):
                    recorder.write(
                        source="system",
                        event_type="book_ticker_divergence",
                        payload={"book_bid": bbo.bid, "book_ask": bbo.ask},
                        receive_ts_ms=now_ms,
                    )
                    if current_orders:
                        cancel = KalshiQuotePlan(
                            tuple(order.order_id for order in current_orders), (), "book divergence"
                        )
                        execute_plan(cancel)
                        if not live:
                            current_orders = executor.resting()
                    stream.reconnect()
                    ticker_watch.reset()
                    continue

            if spot is None or forecast is None:
                if current_orders:
                    cancel = KalshiQuotePlan(tuple(order.order_id for order in current_orders), (), "pricing unavailable")
                    execute_plan(cancel)
                    if not live:
                        current_orders = executor.resting()
                continue

            seconds_to_close = max(0.0, market.close_ts - now_s)
            fair = kalshi_btc15m_fair_value(
                strike_average=market.strike_average,
                spot=spot,
                sigma_per_s=forecast.sigma_per_s,
                seconds_to_close=seconds_to_close,
                observed_final_values=brti_snapshot.final_minute_values if use_brti and seconds_to_close <= 60 else (),
            )
            if not live:
                executor.last_fair_yes = fair.yes
            bid_tox = ask_tox = 0.0
            if toxicity_gate is not None and bbo.valid and bbo.bid is not None and bbo.ask is not None:
                bid_tox, ask_tox = toxicity_gate.toxicity_dollars(
                    now_ms=now_ms,
                    book=book,
                    bbo=bbo,
                    fair_yes=fair.yes,
                    seconds_to_close=seconds_to_close,
                )
            data_age = now_ms - stream.last_message_ms if stream.last_message_ms else 10**9
            decision = strategy.decide(
                fair_yes=fair.yes,
                bbo=bbo,
                inventory=inventory,
                current_orders=current_orders,
                seconds_to_close=seconds_to_close,
                data_age_ms=data_age,
                bid_toxicity_dollars=bid_tox,
                ask_toxicity_dollars=ask_tox,
            )
            if decision.plan.changed:
                # Rate-limit re-quotes; cancels always go through immediately —
                # deferring a cancel bundled with a repost would leave the
                # stale quote resting for the whole throttle window.
                if decision.plan.post and now_s - last_execute_s < config.quote_throttle_s:
                    if decision.plan.cancel_ids:
                        execute_plan(KalshiQuotePlan(decision.plan.cancel_ids, (), "cancel before throttled repost"))
                    continue
                if not execute_plan(decision.plan):
                    continue
                last_execute_s = now_s
                if not live:
                    current_orders = executor.resting()
                # Live state was seeded from the create responses inside
                # execute_plan; the periodic reconcile remains the authority.
                summary = tuple((quote.side, quote.price, quote.count) for quote in decision.plan.post)
                if summary != last_quote_summary or verbose:
                    print(
                        f"[quote] fair={fair.yes:.4f} source={price_source} sigma={forecast.sigma_per_s:.8f} "
                        f"position={inventory.position:.2f} cancel={len(decision.plan.cancel_ids)} post={summary} "
                        f"reason={decision.plan.reason}",
                        flush=True,
                    )
                    last_quote_summary = summary
                recorder.write(
                    source="strategy",
                    event_type="decision",
                    payload={
                        "fair_yes": fair.yes,
                        "expected_average": fair.expected_average,
                        "average_std": fair.average_std,
                        "sigma_per_s": forecast.sigma_per_s,
                        "price_source": price_source,
                        "brti_source_ts_ms": brti_snapshot.source_ts_ms,
                        "book_source_ts_ms": book.last_update_ms,
                        "brti_to_decision_ms": (
                            now_ms - brti_snapshot.source_ts_ms if brti_snapshot.source_ts_ms else None
                        ),
                        "book_to_decision_ms": now_ms - book.last_update_ms if book.last_update_ms else None,
                        "position": inventory.position,
                        "cancel_ids": list(decision.plan.cancel_ids),
                        "post": [quote.__dict__ for quote in decision.plan.post],
                        "reason": decision.plan.reason,
                    },
                    receive_ts_ms=now_ms,
                )
    finally:
        try:
            if live:
                executor.cancel_all(market.ticker)
            else:
                executor.cancel_all()
                recorder.write(
                    source="strategy",
                    event_type="paper_summary",
                    payload={
                        "position": executor.position,
                        "cash_flow": executor.cash_flow,
                        "fees_paid": executor.fees_paid,
                        "fills": executor.fill_count,
                        "maker_fee_multiplier": executor.maker_fee_multiplier,
                    },
                )
                print(
                    f"[paper] fills={executor.fill_count} position={executor.position:.2f} "
                    f"cash_flow={executor.cash_flow:.4f} fees={executor.fees_paid:.4f}",
                    flush=True,
                )
        finally:
            stream.stop()
            for ref in refs:
                ref.stop()
            recorder.close()
    schedule_result_finalizer(
        rest_url=config.rest_url,
        ticker=market.ticker,
        path=path,
        expected_expiration_ts=market.expected_expiration_ts,
    )
    return path


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run the settlement-aware Kalshi BTC 15m maker.")
    parser.add_argument("--ticker")
    parser.add_argument("--output")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--confirm-live", default="", help="must equal KXBTC15M when --live is used")
    parser.add_argument("--windows", type=int, default=1, help="0 runs continuously")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.live and args.confirm_live != "KXBTC15M":
        raise SystemExit("live trading requires --confirm-live KXBTC15M")
    config = KalshiRuntimeConfig.from_env()
    completed = 0
    while args.windows == 0 or completed < args.windows:
        run_one(config=config, ticker=args.ticker, live=args.live, output=args.output, verbose=args.verbose)
        completed += 1
        if args.ticker or args.output:
            break
        time.sleep(1.0)


if __name__ == "__main__":
    main()
