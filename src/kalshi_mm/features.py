"""Per-window feature compaction for research (STRATEGY.md §5, §9 P2).

Each closed market window is compacted exactly once into a columnar feature
table sampled at a fixed cadence. Feature files are append-only artifacts:
research jobs scan the growing store; models refit on schedules. Forward-mid
targets are computed within the window only, so no row leaks across windows.
"""
from __future__ import annotations

import argparse
import bisect
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Optional

from .exchange import BrtiState, KalshiMarket, KalshiOrderBook
from .fair_value import kalshi_btc15m_fair_value
from .recorder import iter_events
from .research import PreparedWindow, _window_quality
from .settlement import infer_settlement
from .volatility import MultiHorizonVolatility

EMIT_INTERVAL_MS = 250
TARGET_HORIZONS_MS = (1_000, 5_000, 30_000)
TARGET_MAX_SLACK_MS = 1_500
EXTERNAL_VENUES = ("binance", "coinbase", "kraken")
EXTERNAL_MAX_AGE_MS = 2_000
SCHEMA_VERSION = 1


@dataclass
class _ExternalBookView:
    ts_ms: int = 0
    mid: Optional[float] = None
    microprice: Optional[float] = None
    imbalance: Optional[float] = None


class _FlowTracker:
    """Signed taker flow and trade intensity over rolling windows."""

    def __init__(self, max_window_ms: int = 30_000):
        self.max_window_ms = max_window_ms
        self._trades: Deque[tuple[int, float]] = deque()

    def add(self, ts_ms: int, signed_count: float) -> None:
        self._trades.append((ts_ms, signed_count))
        cutoff = ts_ms - self.max_window_ms
        while self._trades and self._trades[0][0] < cutoff:
            self._trades.popleft()

    def signed_flow(self, now_ms: int, window_ms: int) -> float:
        cutoff = now_ms - window_ms
        return sum(count for ts, count in self._trades if ts >= cutoff)

    def trade_count(self, now_ms: int, window_ms: int) -> int:
        cutoff = now_ms - window_ms
        return sum(1 for ts, _ in self._trades if ts >= cutoff)

    def last_trade_age_ms(self, now_ms: int) -> Optional[int]:
        return now_ms - self._trades[-1][0] if self._trades else None


def window_feature_rows(window: PreparedWindow) -> list[dict[str, Any]]:
    """Stream one stitched window into sampled feature rows with forward targets."""
    market = window.market
    settlement = infer_settlement(window.rows, market)
    outcome = (
        1.0 if settlement and settlement.result == "yes" else 0.0 if settlement else None
    )
    book = KalshiOrderBook(market.ticker)
    brti = BrtiState()
    vol = MultiHorizonVolatility()
    flow = _FlowTracker()
    external: dict[str, _ExternalBookView] = {venue: _ExternalBookView() for venue in EXTERNAL_VENUES}
    external_spot: dict[str, tuple[int, float]] = {}
    rows_out: list[dict[str, Any]] = []
    last_emit_ms = 0
    prev_bid_size: Optional[float] = None
    prev_ask_size: Optional[float] = None

    for row in window.rows:
        ts_ms = int(row["receive_ts_ms"])
        source = row.get("source")
        payload = row.get("payload") or {}
        event_type = row.get("event_type")

        if source == "kalshi":
            book.apply(payload, use_yes_price=True)
            if brti.apply(payload) and brti.value is not None:
                vol.update(brti.source_ts_ms or ts_ms, brti.value)
            if payload.get("type") == "trade":
                msg = payload.get("msg") or {}
                if msg.get("market_ticker") == market.ticker:
                    try:
                        count = float(msg.get("count_fp") or msg.get("count") or 0)
                    except (TypeError, ValueError):
                        count = 0.0
                    taker = msg.get("taker_book_side")
                    if count > 0 and taker in {"bid", "ask"}:
                        flow.add(ts_ms, count if taker == "bid" else -count)
        elif source in EXTERNAL_VENUES:
            if event_type == "spot":
                try:
                    external_spot[source] = (int(row.get("exchange_ts_ms") or ts_ms), float(payload["price"]))
                except (TypeError, ValueError, KeyError):
                    pass
            elif event_type == "external_orderbook":
                view = external[source]
                view.ts_ms = ts_ms
                view.mid = _maybe_float(payload.get("mid"))
                view.microprice = _maybe_float(payload.get("microprice"))
                view.imbalance = _maybe_float(payload.get("depth_imbalance"))

        if ts_ms / 1000.0 > market.close_ts:
            continue
        if ts_ms - last_emit_ms < EMIT_INTERVAL_MS:
            continue
        bbo = book.snapshot()
        forecast = vol.forecast()
        if not bbo.valid or bbo.bid is None or bbo.ask is None or brti.value is None or forecast is None:
            continue
        last_emit_ms = ts_ms

        seconds_to_close = max(0.0, market.close_ts - ts_ms / 1000.0)
        mid = (bbo.bid + bbo.ask) / 2.0
        spread = bbo.ask - bbo.bid
        top_total = bbo.bid_size + bbo.ask_size
        microprice = (
            (bbo.ask * bbo.bid_size + bbo.bid * bbo.ask_size) / top_total if top_total > 0 else mid
        )
        bids3 = book.top_levels("bid", 3)
        asks3 = book.top_levels("ask", 3)
        depth_bid3 = sum(size for _, size in bids3)
        depth_ask3 = sum(size for _, size in asks3)
        depth3 = depth_bid3 + depth_ask3
        imb_top1 = (bbo.bid_size - bbo.ask_size) / top_total if top_total > 0 else 0.0
        imb_top3 = (depth_bid3 - depth_ask3) / depth3 if depth3 > 0 else 0.0
        bid_size_delta = bbo.bid_size - prev_bid_size if prev_bid_size is not None else 0.0
        ask_size_delta = bbo.ask_size - prev_ask_size if prev_ask_size is not None else 0.0
        prev_bid_size, prev_ask_size = bbo.bid_size, bbo.ask_size

        fresh_spots = {
            venue: price
            for venue, (venue_ts, price) in external_spot.items()
            if ts_ms - venue_ts <= EXTERNAL_MAX_AGE_MS
        }
        proxy = _median(list(fresh_spots.values())) if fresh_spots else None
        basis_bps = (proxy - brti.value) / brti.value * 10_000 if proxy is not None else None
        dispersion_bps = (
            (max(fresh_spots.values()) - min(fresh_spots.values())) / brti.value * 10_000
            if len(fresh_spots) > 1
            else None
        )
        micro_leads = [
            (view.microprice - brti.value) / brti.value * 10_000
            for view in external.values()
            if view.microprice is not None and ts_ms - view.ts_ms <= EXTERNAL_MAX_AGE_MS
        ]
        ext_imbalances = [
            view.imbalance
            for view in external.values()
            if view.imbalance is not None and ts_ms - view.ts_ms <= EXTERNAL_MAX_AGE_MS
        ]

        observed = brti.final_minute_values if seconds_to_close <= 60 else ()
        fair = kalshi_btc15m_fair_value(
            strike_average=market.strike_average,
            spot=brti.value,
            sigma_per_s=forecast.sigma_per_s,
            seconds_to_close=seconds_to_close,
            observed_final_values=observed,
        )
        denom = forecast.sigma_per_s * math.sqrt(max(1e-9, seconds_to_close))
        z_moneyness = math.log(brti.value / market.strike_average) / denom if denom > 0 else None

        rows_out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "ticker": market.ticker,
                "ts_ms": ts_ms,
                "seconds_to_close": seconds_to_close,
                # Kalshi book
                "mid": mid,
                "spread": spread,
                "bid": bbo.bid,
                "ask": bbo.ask,
                "bid_size": bbo.bid_size,
                "ask_size": bbo.ask_size,
                "microprice_minus_mid": microprice - mid,
                "imb_top1": imb_top1,
                "imb_top3": imb_top3,
                "depth_top3": depth3,
                "bid_size_delta": bid_size_delta,
                "ask_size_delta": ask_size_delta,
                # Kalshi flow
                "flow_1s": flow.signed_flow(ts_ms, 1_000),
                "flow_5s": flow.signed_flow(ts_ms, 5_000),
                "flow_30s": flow.signed_flow(ts_ms, 30_000),
                "trades_5s": flow.trade_count(ts_ms, 5_000),
                "last_trade_age_ms": flow.last_trade_age_ms(ts_ms),
                # External
                "brti": brti.value,
                "ext_basis_bps": basis_bps,
                "ext_dispersion_bps": dispersion_bps,
                "ext_micro_lead_bps": _median(micro_leads) if micro_leads else None,
                "ext_imbalance": _median(ext_imbalances) if ext_imbalances else None,
                "ext_venues_fresh": len(fresh_spots),
                # State / model
                "sigma_per_s": forecast.sigma_per_s,
                "jump_ratio": forecast.jump_ratio,
                "z_moneyness": z_moneyness,
                "gbm_fair": fair.yes,
                "gbm_fair_minus_mid": fair.yes - mid,
                "final_minute_count": fair.observed_count,
                # Labels (filled below / per window)
                "outcome_yes": outcome,
                "quality_flags": ",".join(window.quality.flags) or "",
            }
        )

    timestamps = [row["ts_ms"] for row in rows_out]
    mids = [row["mid"] for row in rows_out]
    for index, row in enumerate(rows_out):
        for horizon in TARGET_HORIZONS_MS:
            key = f"fwd_mid_delta_{horizon // 1000}s"
            target_index = bisect.bisect_left(timestamps, row["ts_ms"] + horizon)
            # A bounded match only: across an emission gap (invalid book, feed
            # outage) the next row may be much later than the horizon, and the
            # label would silently measure a far longer move.
            if (
                target_index < len(mids)
                and timestamps[target_index] <= row["ts_ms"] + horizon + TARGET_MAX_SLACK_MS
            ):
                row[key] = mids[target_index] - mids[index]
            else:
                row[key] = None
    return rows_out


def _maybe_float(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def read_parquet_rows(path: str | Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - packaging failure
        raise RuntimeError("pip install '.[research]' to enable parquet input") from exc
    table = pq.read_table(path)
    columns = {name: table.column(name).to_pylist() for name in table.column_names}
    count = len(next(iter(columns.values()))) if columns else 0
    return [{name: values[i] for name, values in columns.items()} for i in range(count)]


def write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - packaging failure
        raise RuntimeError("pip install '.[research]' to enable parquet output") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    table = pa.table({column: [row.get(column) for row in rows] for column in columns})
    # Atomic write: a kill mid-write must never leave a truncated file that
    # skip-if-exists compaction would then freeze into the store forever.
    temporary = path.with_name(path.name + ".tmp")
    pq.write_table(table, temporary)
    temporary.replace(path)


def _read_market(path: str | Path) -> Optional[KalshiMarket]:
    """Read just enough of a recording to identify its market."""
    try:
        for row in iter_events(path):
            if row.get("event_type") == "market_metadata":
                return KalshiMarket.from_api(row["payload"])
    except EOFError:
        pass
    return None


def group_recordings_by_market(recording_paths: list[str | Path]) -> list[tuple[KalshiMarket, list[Path]]]:
    """Group fragment files by market ticker without loading full recordings."""
    groups: dict[str, tuple[KalshiMarket, list[Path]]] = {}
    for path in recording_paths:
        market = _read_market(path)
        if market is None:
            print(f"[features] {path}: no market metadata; skipped", flush=True)
            continue
        groups.setdefault(market.ticker, (market, []))[1].append(Path(path))
    ordered = sorted(groups.values(), key=lambda item: item[0].open_ts)
    return [(market, sorted(paths)) for market, paths in ordered]


def _window_is_final(market: KalshiMarket, paths: list[Path]) -> bool:
    """A window may be compacted only once closed and fully readable.

    Compacting an in-progress window would freeze a partial window into the
    feature store permanently (compaction is skip-if-exists). For historical
    windows (closed >1h ago, files untouched) finality is decided from
    metadata — full-file scans of hundreds of recordings just to find close
    markers dominated batch runtime.
    """
    import time as _time

    now = _time.time()
    if now > market.close_ts + 3_600:
        try:
            return all(now - path.stat().st_mtime > 60 for path in paths)
        except OSError:
            return False
    closed = False
    for path in paths:
        try:
            for row in iter_events(path):
                if row.get("event_type") in {"market_status_at_close", "market_result"}:
                    closed = True
        except EOFError:
            return False
    return closed


def _load_window(market: KalshiMarket, paths: list[Path]) -> PreparedWindow:
    """Load one market's fragments, stitched into one globally time-sorted stream.

    Fragments can OVERLAP in time (a crashed process's result-finalizer appends
    to the old file after the replacement started recording); concatenation
    alone leaves rows non-monotonic, which silently breaks every bisect-based
    join and forward-target lookup downstream. Python's sort is stable, so
    equal-timestamp rows keep their within-file order.
    """
    files = sorted(
        ((str(path), list(iter_events(path))) for path in paths),
        key=lambda item: min((int(row["receive_ts_ms"]) for row in item[1]), default=0),
    )
    rows = tuple(
        sorted(
            (row for _, file_rows in files for row in file_rows),
            key=lambda row: int(row["receive_ts_ms"]),
        )
    )
    quality = _window_quality(market, rows, tuple(path for path, _ in files))
    return PreparedWindow(market=market, rows=rows, quality=quality)


def compact_windows(
    recording_paths: list[str | Path],
    out_dir: str | Path,
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Compact each market window (one parquet per market ticker) idempotently.

    Windows are loaded and freed one at a time so memory is bounded by the
    largest single market, not the whole corpus.
    """
    out_dir = Path(out_dir)
    written: list[Path] = []
    for market, paths in group_recordings_by_market(recording_paths):
        target = out_dir / f"{market.ticker}.parquet"
        if target.exists() and not overwrite:
            continue
        if not _window_is_final(market, paths):
            print(f"[features] {market.ticker}: window not finalized; skipped", flush=True)
            continue
        window = _load_window(market, paths)
        rows = window_feature_rows(window)
        if not rows:
            print(f"[features] {market.ticker}: no valid rows; skipped", flush=True)
            continue
        write_parquet(rows, target)
        written.append(target)
        print(
            f"[features] {market.ticker}: rows={len(rows)} "
            f"fragments={window.quality.fragments} flags={','.join(window.quality.flags) or '-'} "
            f"-> {target}",
            flush=True,
        )
        del window, rows
    return written


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compact recordings into per-window feature parquet files."
    )
    parser.add_argument("recordings", nargs="+")
    parser.add_argument("--out", default="data/features")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    written = compact_windows(args.recordings, args.out, overwrite=args.overwrite)
    print(f"[features] wrote {len(written)} window file(s) to {args.out}")


if __name__ == "__main__":
    main()
