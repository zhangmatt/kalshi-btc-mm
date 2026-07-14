from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .exchange import (
    BrtiState,
    KalshiOrderBook,
    KalshiRestClient,
    KalshiSigner,
    PriceGrid,
    build_order,
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
from .feeds import CompositeReference, reference_streams
from .fair_value import kalshi_btc15m_fair_value
from .volatility import MultiHorizonVolatility


class PaperExecutor:
    def __init__(self):
        self.orders: dict[str, RestingKalshiOrder] = {}
        self._next_id = 1

    def execute(self, plan: KalshiQuotePlan, *, ticker: str, expiration_time: int) -> None:
        for order_id in plan.cancel_ids:
            self.orders.pop(order_id, None)
        for quote in plan.post:
            order_id = f"paper-{self._next_id}"
            self._next_id += 1
            self.orders[order_id] = RestingKalshiOrder(order_id, order_id, quote.side, quote.price, quote.count)

    def resting(self) -> list[RestingKalshiOrder]:
        return list(self.orders.values())

    def cancel_all(self) -> None:
        self.orders.clear()


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

    def execute(self, plan: KalshiQuotePlan, *, ticker: str, expiration_time: int) -> None:
        if plan.cancel_ids:
            started_ms = time.time_ns() // 1_000_000
            responses = self.client.cancel_orders(list(plan.cancel_ids), subaccount=self.subaccount)
            self._record_ack("cancel_ack", started_ms, responses)
            errors = [row.get("error") for row in responses if row.get("error")]
            if errors:
                raise RuntimeError(f"cancel failed: {errors}")
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
                raise RuntimeError(f"post failed: {errors}")

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
        orders = self.resting(ticker)
        if orders:
            self.client.cancel_orders([order.order_id for order in orders], subaccount=self.subaccount)


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
    market = _choose_market(client, ticker)
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
    stream = KalshiWebSocket(
        signer=signer,
        market_ticker=market.ticker,
        order_book=book,
        brti=brti,
        ws_url=config.ws_url,
        recorder=recorder,
        include_private=live,
        insecure_ssl=config.ws_insecure,
        verbose=verbose,
    )
    strategy = KalshiMakerStrategy(price_grid=PriceGrid(market.price_ranges), config=config.strategy)
    executor: Any = LiveExecutor(client, subaccount=config.subaccount, recorder=recorder) if live else PaperExecutor()
    inventory = _inventory(client, market.ticker, config.subaccount) if live else KalshiInventory()
    current_orders: list[RestingKalshiOrder] = executor.resting(market.ticker) if live else executor.resting()
    last_position_poll = 0.0
    last_reconcile = 0.0
    last_brti_ts = 0
    last_quote_summary: Optional[tuple[Any, ...]] = None

    print(
        f"[kalshi] mode={'LIVE' if live else 'PAPER'} ticker={market.ticker} "
        f"strike={market.strike_average:.2f} output={path}",
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
                inventory = _inventory(client, market.ticker, config.subaccount)
                last_position_poll = now_s
            if live and now_s - last_reconcile >= config.order_reconcile_s:
                current_orders = executor.resting(market.ticker)
                last_reconcile = now_s
            elif not live:
                current_orders = executor.resting()

            bbo = book.snapshot()
            proxy = composite.snapshot(now_ms)
            brti_age = now_ms - (brti_snapshot.source_ts_ms or 0)
            use_brti = brti_snapshot.value is not None and brti_age <= config.strategy.stale_after_ms
            if use_brti:
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

            if spot is None or forecast is None:
                if current_orders:
                    cancel = KalshiQuotePlan(tuple(order.order_id for order in current_orders), (), "pricing unavailable")
                    executor.execute(cancel, ticker=market.ticker, expiration_time=market.close_ts - 1)
                    current_orders = executor.resting(market.ticker) if live else executor.resting()
                continue

            seconds_to_close = max(0.0, market.close_ts - now_s)
            fair = kalshi_btc15m_fair_value(
                strike_average=market.strike_average,
                spot=spot,
                sigma_per_s=forecast.sigma_per_s,
                seconds_to_close=seconds_to_close,
                observed_final_values=brti_snapshot.final_minute_values if use_brti and seconds_to_close <= 60 else (),
            )
            data_age = now_ms - stream.last_message_ms if stream.last_message_ms else 10**9
            decision = strategy.decide(
                fair_yes=fair.yes,
                bbo=bbo,
                inventory=inventory,
                current_orders=current_orders,
                seconds_to_close=seconds_to_close,
                data_age_ms=data_age,
            )
            if decision.plan.changed:
                executor.execute(decision.plan, ticker=market.ticker, expiration_time=market.close_ts - 1)
                current_orders = executor.resting(market.ticker) if live else executor.resting()
                if live:
                    last_reconcile = now_s
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
