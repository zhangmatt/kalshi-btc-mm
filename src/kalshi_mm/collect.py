from __future__ import annotations

import argparse
import dataclasses
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .exchange import BrtiState, KalshiOrderBook, KalshiRestClient, KalshiSigner
from .config import KalshiRuntimeConfig
from .websocket import KalshiWebSocket
from .recorder import EventRecorder
from .feeds import CompositeReference, external_depth_streams, reference_streams


def _market_payload(market) -> dict:
    return {
        "ticker": market.ticker,
        "event_ticker": market.event_ticker,
        "title": market.title,
        "status": market.status,
        "open_time": datetime.fromtimestamp(market.open_ts, timezone.utc).isoformat(),
        "close_time": datetime.fromtimestamp(market.close_ts, timezone.utc).isoformat(),
        "expected_expiration_time": datetime.fromtimestamp(market.expected_expiration_ts, timezone.utc).isoformat(),
        "floor_strike": market.strike_average,
        "price_ranges": [dataclasses.asdict(value) for value in market.price_ranges],
        "rules_primary": market.rules_primary,
        "rules_secondary": market.rules_secondary,
        "result": market.result,
    }


def _choose_market(
    client: KalshiRestClient,
    ticker: Optional[str],
    *,
    wait: bool = False,
    retry_s: float = 1.0,
    verbose: bool = False,
):
    if ticker:
        return client.get_market(ticker)
    announced_wait = False
    while True:
        now = time.time()
        try:
            markets = [market for market in client.discover_btc15m(status="open") if market.close_ts > now]
        except Exception as exc:
            if not wait:
                raise
            if verbose or not announced_wait:
                print(f"[collect] market discovery failed ({exc}); retrying", flush=True)
                announced_wait = True
            time.sleep(max(0.1, retry_s))
            continue
        if markets:
            return min(markets, key=lambda market: market.close_ts)
        if not wait:
            raise RuntimeError("no active KXBTC15M market found")
        if verbose or not announced_wait:
            print("[collect] no fully active KXBTC15M market; retrying discovery", flush=True)
            announced_wait = True
        time.sleep(max(0.1, retry_s))


def _record_market_context(recorder: EventRecorder, client: KalshiRestClient, ticker: str) -> None:
    series_ticker = "KXBTC15M"
    queries = (
        ("series_metadata", lambda: client.get_series(series_ticker)),
        ("series_fee_changes", lambda: {"fee_changes": client.get_series_fee_changes(series_ticker)}),
        (
            "market_incentives",
            lambda: {
                "incentive_programs": [
                    value for value in client.get_incentives() if value.get("market_ticker") == ticker
                ]
            },
        ),
    )
    for event_type, query in queries:
        try:
            recorder.write(source="kalshi_rest", event_type=event_type, payload=query())
        except Exception as exc:
            recorder.write(
                source="system",
                event_type=f"{event_type}_error",
                payload={"error": str(exc)},
            )


def schedule_result_finalizer(
    *,
    rest_url: str,
    ticker: str,
    path: Path,
    expected_expiration_ts: int,
    daemon: bool = False,
) -> threading.Thread:
    """Append the resolved outcome without delaying collection of the next market."""

    def finalize() -> None:
        delay = max(0.0, expected_expiration_ts + 2.0 - time.time())
        if delay:
            time.sleep(delay)
        client = KalshiRestClient(base_url=rest_url)
        last_error = "result unavailable"
        for _ in range(45):
            try:
                market = client.get_market(ticker)
                if market.result.lower() in {"yes", "no"}:
                    with EventRecorder(path) as recorder:
                        recorder.write(source="system", event_type="market_result", payload=_market_payload(market))
                    return
                last_error = f"market status={market.status} result={market.result!r}"
            except Exception as exc:  # pragma: no cover - network recovery
                last_error = str(exc)
            time.sleep(2.0)
        with EventRecorder(path) as recorder:
            recorder.write(source="system", event_type="market_result_error", payload={"error": last_error})

    thread = threading.Thread(target=finalize, name=f"result-{ticker}", daemon=daemon)
    thread.start()
    return thread


def collect_one(
    *,
    config: KalshiRuntimeConfig,
    ticker: Optional[str],
    output: Optional[str],
    duration_s: Optional[float],
    include_private: bool,
    verbose: bool,
    result_thread_daemon: bool = False,
    hold_feeds_until_rollover: bool = False,
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
    path = Path(output) if output else Path(config.data_dir) / market.ticker / f"events_{run_id}.jsonl.gz"
    recorder = EventRecorder(path)
    recorder.write(source="system", event_type="market_metadata", payload=_market_payload(market))
    recorder.write(
        source="system",
        event_type="collector_config",
        payload={
            "external_depth_enabled": config.external_depth_enabled,
            "external_depth_levels": config.external_depth_levels,
            "external_depth_sample_hz": config.external_depth_sample_hz,
            "external_depth_venues": ["binance", "coinbase", "kraken"],
        },
    )
    _record_market_context(recorder, client, market.ticker)
    book = KalshiOrderBook(market.ticker)
    brti = BrtiState()
    composite = CompositeReference()
    refs = reference_streams(
        composite,
        recorder,
        insecure_ssl=config.ws_insecure,
        binance_url=config.binance_ws_url,
    )
    depth_refs = (
        external_depth_streams(
            recorder,
            depth=config.external_depth_levels,
            sample_hz=config.external_depth_sample_hz,
            insecure_ssl=config.ws_insecure,
            verbose=verbose,
            binance_url=config.binance_depth_ws_url,
            coinbase_url=config.coinbase_depth_ws_url,
            kraken_url=config.kraken_depth_ws_url,
        )
        if config.external_depth_enabled
        else []
    )
    stream = KalshiWebSocket(
        signer=signer,
        market_ticker=market.ticker,
        order_book=book,
        brti=brti,
        ws_url=config.ws_url,
        recorder=recorder,
        include_private=include_private,
        insecure_ssl=config.ws_insecure,
        verbose=verbose,
    )
    print(f"[collect] ticker={market.ticker} strike={market.strike_average:.2f} output={path}", flush=True)
    if config.ws_insecure:
        print("[collect] warning: websocket TLS certificate verification is disabled", flush=True)
    for ref in refs:
        ref.start()
    for depth_ref in depth_refs:
        depth_ref.start()
    stream.start()
    complete_deadline = market.close_ts + 1.0
    deadline = min(complete_deadline, time.time() + duration_s) if duration_s else complete_deadline
    try:
        while time.time() < deadline:
            time.sleep(1.0)
            recorder.check_health()
            if stream.fatal_error:
                raise RuntimeError(stream.fatal_error)
            if verbose:
                bbo = book.snapshot()
                reference = composite.snapshot()
                brti_snapshot = brti.snapshot()
                now_ms = time.time_ns() // 1_000_000
                brti_age_ms = now_ms - brti_snapshot.source_ts_ms if brti_snapshot.source_ts_ms else None
                ws_age_ms = now_ms - stream.last_message_ms if stream.last_message_ms else None
                depth_ages = ",".join(
                    f"{depth_ref.venue}:{now_ms - depth_ref.last_event_ms if depth_ref.last_event_ms else 'none'}"
                    for depth_ref in depth_refs
                )
                print(
                    f"[collect] bid={bbo.bid if bbo.valid else None} ask={bbo.ask if bbo.valid else None} "
                    f"book_valid={bbo.valid} brti={brti_snapshot.value} brti_age_ms={brti_age_ms} "
                    f"proxy={reference.price if reference else None} ws_age_ms={ws_age_ms} "
                    f"external_l2_age_ms={depth_ages or 'disabled'}",
                    flush=True,
                )
        if hold_feeds_until_rollover and deadline >= market.close_ts:
            # Keep BRTI and external references continuous while Kalshi computes and
            # publishes the next market's opening strike. The recorder and stream
            # must stay healthy through the hold, so this cannot block in
            # _choose_market's bare wait loop.
            announced_hold = False
            while True:
                recorder.check_health()
                if stream.fatal_error:
                    raise RuntimeError(stream.fatal_error)
                try:
                    now = time.time()
                    candidates = [
                        value for value in client.discover_btc15m(status="open") if value.close_ts > now
                    ]
                except Exception as exc:
                    if verbose or not announced_hold:
                        print(f"[collect] rollover discovery failed ({exc}); retrying", flush=True)
                        announced_hold = True
                    candidates = []
                if candidates:
                    break
                if verbose or not announced_hold:
                    print("[collect] holding feeds until the next market opens", flush=True)
                    announced_hold = True
                time.sleep(1.0)
    finally:
        stream.stop()
        for ref in refs:
            ref.stop()
        for depth_ref in depth_refs:
            depth_ref.stop()
        try:
            final_market = client.get_market(market.ticker)
            status_event = "market_status_at_close" if time.time() >= market.close_ts else "market_status_on_exit"
            recorder.write(source="system", event_type=status_event, payload=_market_payload(final_market))
        except Exception as exc:
            recorder.write(source="system", event_type="market_status_error", payload={"error": str(exc)})
        recorder.close()
    if deadline >= market.close_ts:
        schedule_result_finalizer(
            rest_url=config.rest_url,
            ticker=market.ticker,
            path=path,
            expected_expiration_ts=market.expected_expiration_ts,
            daemon=result_thread_daemon,
        )
    return path


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Record Kalshi KXBTC15M, BRTI, and reference exchange events.")
    parser.add_argument("--ticker")
    parser.add_argument("--output")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--windows", type=int, default=1, help="0 records continuously")
    parser.add_argument("--include-private", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    config = KalshiRuntimeConfig.from_env()
    completed = 0
    while args.windows == 0 or completed < args.windows:
        collect_one(
            config=config,
            ticker=args.ticker,
            output=args.output,
            duration_s=args.duration,
            include_private=args.include_private,
            verbose=args.verbose,
            result_thread_daemon=args.windows == 0,
            hold_feeds_until_rollover=args.windows == 0,
        )
        completed += 1
        if args.ticker or args.output or args.duration:
            break
        time.sleep(1.0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[collect] stopped", flush=True)
