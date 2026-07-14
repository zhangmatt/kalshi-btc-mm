from __future__ import annotations

import argparse
import gzip
import json
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


REQUIRED_CHANNELS = {
    "orderbook_delta",
    "ticker",
    "trade",
    "cfbenchmarks_value",
    "market_lifecycle_v2",
}
REFERENCE_VENUES = {"binance", "coinbase", "kraken"}
CONTEXT_EVENTS = {"market_metadata", "series_metadata", "series_fee_changes", "market_incentives"}


@dataclass(frozen=True)
class AuditResult:
    path: Path
    status: str
    rows: int
    first_ts_ms: Optional[int]
    last_ts_ms: Optional[int]
    start_offset_s: Optional[float]
    end_offset_s: Optional[float]
    subscribed_channels: frozenset[str]
    reference_venues: frozenset[str]
    external_depth_venues: frozenset[str]
    event_counts: Counter[str]
    sequence_gaps: int
    invalid_rows: int
    gzip_complete: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def healthy(self) -> bool:
        return not self.errors


def _parse_iso_ms(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1_000)


def _scan(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    state: dict[str, Any] = {
        "rows": 0,
        "gzip_complete": True,
        "invalid_rows": 0,
        "counts": Counter(),
        "metadata": {},
        "first_ts": None,
        "last_ts": None,
        "channels": set(),
        "venues": set(),
        "context": set(),
        "depth_venues": set(),
        "collector_config": {},
    }
    try:
        with opener(path, "rt", encoding="utf-8") as handle:
            try:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        state["invalid_rows"] += 1
                        continue
                    state["rows"] += 1
                    event_type = str(row.get("event_type", ""))
                    source = str(row.get("source", ""))
                    state["counts"][event_type] += 1
                    if event_type == "market_metadata" and not state["metadata"]:
                        state["metadata"] = row.get("payload") or {}
                    if event_type == "collector_config" and not state["collector_config"]:
                        state["collector_config"] = row.get("payload") or {}
                    receive_ts = row.get("receive_ts_ms")
                    if receive_ts is not None:
                        receive_ts = int(receive_ts)
                        if state["first_ts"] is None or receive_ts < state["first_ts"]:
                            state["first_ts"] = receive_ts
                        if state["last_ts"] is None or receive_ts > state["last_ts"]:
                            state["last_ts"] = receive_ts
                    if source == "kalshi" and event_type == "subscribed":
                        channel = str((row.get("payload") or {}).get("msg", {}).get("channel"))
                        if channel != "None":
                            state["channels"].add(channel)
                    if event_type == "spot" and source in REFERENCE_VENUES:
                        state["venues"].add(source)
                    if event_type == "external_orderbook" and source in REFERENCE_VENUES:
                        state["depth_venues"].add(source)
                    if event_type in CONTEXT_EVENTS:
                        state["context"].add(event_type)
            except EOFError:
                state["gzip_complete"] = False
    except EOFError:
        state["gzip_complete"] = False
    return state


def audit(path: str | Path, *, now_ms: Optional[int] = None, require_live: bool = False) -> AuditResult:
    path = Path(path)
    scan = _scan(path)
    row_count = int(scan["rows"])
    gzip_complete = bool(scan["gzip_complete"])
    invalid_rows = int(scan["invalid_rows"])
    now_ms = now_ms or time.time_ns() // 1_000_000
    counts: Counter[str] = scan["counts"]
    errors: list[str] = []
    warnings: list[str] = []
    if not row_count:
        errors.append("recording contains no readable events")
        return AuditResult(
            path, "empty", 0, None, None, None, None, frozenset(), frozenset(), frozenset(), counts,
            0, invalid_rows, gzip_complete, tuple(errors), tuple(warnings),
        )

    metadata = scan["metadata"]
    try:
        open_ms = _parse_iso_ms(str(metadata["open_time"]))
        close_ms = _parse_iso_ms(str(metadata["close_time"]))
    except (KeyError, ValueError):
        open_ms = close_ms = None
        errors.append("market metadata is missing valid open/close times")

    first_ts = int(scan["first_ts"])
    last_ts = int(scan["last_ts"])
    start_offset = (first_ts - open_ms) / 1_000 if open_ms is not None else None
    end_offset = (last_ts - close_ms) / 1_000 if close_ms is not None else None

    channels = scan["channels"]
    venues = scan["venues"]
    depth_venues = scan["depth_venues"]
    context = scan["context"]
    missing_channels = sorted(REQUIRED_CHANNELS - channels)
    missing_venues = sorted(REFERENCE_VENUES - venues)
    missing_context = sorted(CONTEXT_EVENTS - context)
    if missing_channels:
        errors.append(f"missing websocket subscriptions: {', '.join(missing_channels)}")
    if missing_venues:
        errors.append(f"missing reference venue data: {', '.join(missing_venues)}")
    collector_config = scan["collector_config"]
    if collector_config.get("external_depth_enabled"):
        expected_depth = set(collector_config.get("external_depth_venues") or REFERENCE_VENUES)
        missing_depth = sorted(expected_depth - depth_venues)
        if missing_depth:
            errors.append(f"missing external L2 data: {', '.join(missing_depth)}")
    if missing_context:
        errors.append(f"missing market context: {', '.join(missing_context)}")
    if counts["orderbook_snapshot"] < 1 or counts["orderbook_delta"] < 1:
        errors.append("missing usable L2 order-book snapshot/deltas")
    if counts["cfbenchmarks_value"] < 1:
        errors.append("missing BRTI settlement feed")

    sequence_gaps = counts["orderbook_sequence_gap"]
    if sequence_gaps:
        warnings.append(f"order book recovered from {sequence_gaps} sequence gap(s)")
    if invalid_rows:
        errors.append(f"recording contains {invalid_rows} invalid JSON row(s)")

    has_close = counts["market_status_at_close"] > 0
    if has_close:
        status = "complete"
        if start_offset is not None and start_offset > 5:
            warnings.append(f"recording began {start_offset:.1f}s after market open")
        if end_offset is not None and end_offset < -2:
            errors.append(f"recording ended {-end_offset:.1f}s before market close")
        if require_live and close_ms is not None and now_ms - close_ms > 120_000:
            errors.append(f"no active recording found {(now_ms - close_ms) / 1_000:.1f}s after market close")
    elif close_ms is not None and (now_ms < close_ms + 2_000 or (require_live and now_ms < close_ms + 120_000)):
        status = "in_progress"
        if now_ms - last_ts > 10_000:
            errors.append(f"active recording is stale by {(now_ms - last_ts) / 1_000:.1f}s")
    else:
        status = "partial"
        warnings.append("recording has no close marker")

    if not gzip_complete and status != "in_progress":
        errors.append("gzip stream is truncated")

    return AuditResult(
        path=path,
        status=status,
        rows=row_count,
        first_ts_ms=first_ts,
        last_ts_ms=last_ts,
        start_offset_s=start_offset,
        end_offset_s=end_offset,
        subscribed_channels=frozenset(channels),
        reference_venues=frozenset(venues),
        external_depth_venues=frozenset(depth_venues),
        event_counts=counts,
        sequence_gaps=sequence_gaps,
        invalid_rows=invalid_rows,
        gzip_complete=gzip_complete,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def _latest(root: Path) -> Path:
    files = list(root.glob("**/events_*.jsonl.gz"))
    if not files:
        raise FileNotFoundError(f"no recordings under {root}")
    return max(files, key=lambda value: value.stat().st_mtime_ns)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Check whether a Kalshi BTC 15m recording is research-ready.")
    parser.add_argument("recording", nargs="?")
    parser.add_argument("--data-dir", default="data/kalshi")
    parser.add_argument("--require-live", action="store_true", help="fail if the latest recording is stale")
    args = parser.parse_args(argv)
    path = Path(args.recording) if args.recording else _latest(Path(args.data_dir))
    result = audit(path, require_live=args.require_live)
    verdict = "PASS" if result.healthy else "FAIL"
    print(f"{verdict} status={result.status} rows={result.rows} file={result.path}")
    print(f"coverage_start_s={result.start_offset_s} coverage_end_s={result.end_offset_s}")
    print(f"channels={','.join(sorted(result.subscribed_channels))}")
    print(f"references={','.join(sorted(result.reference_venues))}")
    print(f"external_depth={','.join(sorted(result.external_depth_venues))}")
    print(
        "events="
        f"snapshot:{result.event_counts['orderbook_snapshot']},"
        f"delta:{result.event_counts['orderbook_delta']},"
        f"trade:{result.event_counts['trade']},"
        f"brti:{result.event_counts['cfbenchmarks_value']},"
        f"external_l2:{result.event_counts['external_orderbook']},"
        f"sequence_gap:{result.sequence_gaps}"
    )
    for warning in result.warnings:
        print(f"WARN {warning}")
    for error in result.errors:
        print(f"ERROR {error}")
    if not result.healthy:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
