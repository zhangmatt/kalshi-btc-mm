"""Model T label generation (STRATEGY.md §5): simulated fills → markout labels.

For each finalized window, replay the baseline quoting policy (fees + latency
on), take every simulated fill, compute its 1s/5s markouts against the window's
own mid sequence, and join the feature row in effect at fill time. Output is a
per-window `*.fills.parquet` that the fit harness consumes.

Labels are deterministic functions of the recordings and the baseline policy,
so regeneration is always safe; like feature compaction, windows are processed
one at a time and unfinalized windows are refused.
"""
from __future__ import annotations

import argparse
import bisect
from pathlib import Path
from typing import Any, Optional

from .features import (
    _load_window,
    _window_is_final,
    group_recordings_by_market,
    read_parquet_rows,
    window_feature_rows,
    write_parquet,
)
from .replay import ReplayResult, replay_rows

# A fill is labeled toxic when its side-adjusted markout is at least this
# adverse. One tick (1c) filters bid-ask bounce noise around zero.
TOXIC_MARKOUT_DOLLARS = -0.01
FEATURE_JOIN_MAX_AGE_MS = 2_000


def _markout_at(
    fill_ts: int, fill_price: float, side: str, mids: list[tuple[int, float]], horizon_ms: int
) -> Optional[float]:
    timestamps = [ts for ts, _ in mids]
    index = bisect.bisect_left(timestamps, fill_ts + horizon_ms)
    if index >= len(mids):
        return None
    future_mid = mids[index][1]
    return future_mid - fill_price if side == "bid" else fill_price - future_mid


def fill_label_rows(
    result: ReplayResult,
    feature_rows: list[dict[str, Any]],
    *,
    ticker: str,
) -> list[dict[str, Any]]:
    """Join each simulated fill with the freshest feature row at fill time."""
    mids = list(result.mids)
    feature_ts = [row["ts_ms"] for row in feature_rows]
    out: list[dict[str, Any]] = []
    for fill in result.fills:
        index = bisect.bisect_right(feature_ts, fill.ts_ms) - 1
        if index < 0 or fill.ts_ms - feature_ts[index] > FEATURE_JOIN_MAX_AGE_MS:
            continue
        features = dict(feature_rows[index])
        markout_1s = _markout_at(fill.ts_ms, fill.price, fill.side, mids, 1_000)
        markout_5s = _markout_at(fill.ts_ms, fill.price, fill.side, mids, 5_000)
        features.update(
            {
                "ticker": ticker,
                "fill_ts_ms": fill.ts_ms,
                "fill_side": fill.side,
                "fill_price": fill.price,
                "fill_count": fill.count,
                "fill_fair_yes": fill.fair_yes,
                "feature_age_ms": fill.ts_ms - feature_ts[index],
                "markout_1s": markout_1s,
                "markout_5s": markout_5s,
                "toxic_5s": (
                    None if markout_5s is None else float(markout_5s <= TOXIC_MARKOUT_DOLLARS)
                ),
            }
        )
        out.append(features)
    return out


def generate_labels(
    recording_paths: list[str | Path],
    out_dir: str | Path,
    *,
    latency_ms: int = 150,
    maker_fee_multiplier: float = 1.0,
    min_edge_dollars: float = 0.01,
    overwrite: bool = False,
    features_dir: Optional[str | Path] = None,
) -> list[Path]:
    out_dir = Path(out_dir)
    written: list[Path] = []
    for market, paths in group_recordings_by_market(recording_paths):
        target = out_dir / f"{market.ticker}.fills.parquet"
        if target.exists() and not overwrite:
            continue
        if not _window_is_final(paths):
            print(f"[labels] {market.ticker}: window not finalized; skipped", flush=True)
            continue
        window = _load_window(market, paths)
        result = replay_rows(
            market,
            list(window.rows),
            min_edge_dollars=min_edge_dollars,
            latency_ms=latency_ms,
            maker_fee_multiplier=maker_fee_multiplier,
        )
        # Reuse the compacted feature store when available; recompute otherwise.
        feature_rows = None
        if features_dir is not None:
            feature_path = Path(features_dir) / f"{market.ticker}.parquet"
            if feature_path.exists():
                feature_rows = read_parquet_rows(feature_path)
        if feature_rows is None:
            feature_rows = window_feature_rows(window)
        rows = fill_label_rows(result, feature_rows, ticker=market.ticker)
        if not rows:
            print(f"[labels] {market.ticker}: no labeled fills; skipped", flush=True)
            del window, result
            continue
        write_parquet(rows, target)
        written.append(target)
        toxic = sum(1 for row in rows if row["toxic_5s"] == 1.0)
        print(
            f"[labels] {market.ticker}: fills={len(rows)} toxic_5s={toxic} "
            f"latency_ms={latency_ms} fee_mult={maker_fee_multiplier} -> {target}",
            flush=True,
        )
        del window, result, rows
    return written


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate Model T fill labels from queue-aware baseline replay."
    )
    parser.add_argument("recordings", nargs="+")
    parser.add_argument("--out", default="data/labels")
    parser.add_argument("--features-dir", default=None, help="reuse compacted feature parquet for joins")
    parser.add_argument("--latency-ms", type=int, default=150)
    parser.add_argument("--maker-fee-multiplier", type=float, default=1.0)
    parser.add_argument("--min-edge", type=float, default=0.01)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    written = generate_labels(
        args.recordings,
        args.out,
        latency_ms=args.latency_ms,
        maker_fee_multiplier=args.maker_fee_multiplier,
        min_edge_dollars=args.min_edge,
        overwrite=args.overwrite,
        features_dir=args.features_dir,
    )
    print(f"[labels] wrote {len(written)} window file(s) to {args.out}")


if __name__ == "__main__":
    main()
