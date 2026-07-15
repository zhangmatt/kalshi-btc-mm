import json

from kalshi_mm.recorder import EventRecorder
from kalshi_mm.research import prepare_windows


def _metadata(ticker: str) -> dict:
    return {
        "ticker": ticker,
        "event_ticker": "KXBTC15M",
        "title": "test",
        "status": "active",
        "open_time": "2027-01-15T08:00:00+00:00",
        "close_time": "2027-01-15T08:15:00+00:00",
        "expected_expiration_time": "2027-01-15T08:20:00+00:00",
        "floor_strike": 100.0,
        "price_ranges": [{"start": 0, "end": 1, "step": 0.01}],
        "rules_primary": "test",
        "rules_secondary": "test",
        "result": "",
    }


def _write_fragment(path, ticker: str, start_ms: int, *, reconnects: int = 0) -> None:
    with EventRecorder(path) as recorder:
        recorder.write(
            source="system", event_type="market_metadata", payload=_metadata(ticker), receive_ts_ms=start_ms
        )
        recorder.write(
            source="kalshi",
            event_type="feed_status",
            payload={"status": "connected"},
            receive_ts_ms=start_ms + 1,
        )
        for index in range(reconnects):
            recorder.write(
                source="kalshi",
                event_type="feed_status",
                payload={"status": "connected"},
                receive_ts_ms=start_ms + 10 + index,
            )
        recorder.write(
            source="kalshi",
            event_type="cfbenchmarks_value",
            payload={
                "type": "cfbenchmarks_value",
                "msg": {"index_id": "BRTI", "data": json.dumps({"time": start_ms + 20, "value": "100.0"})},
            },
            receive_ts_ms=start_ms + 20,
        )


def test_fragments_of_one_market_are_a_single_window(tmp_path):
    open_ms = 1_800_000_000_000  # matches the 2027 metadata only loosely; grouping is by ticker
    first = tmp_path / "events_a.jsonl"
    second = tmp_path / "events_b.jsonl"
    other = tmp_path / "events_c.jsonl"
    _write_fragment(first, "KXBTC15M-TEST", open_ms)
    _write_fragment(second, "KXBTC15M-TEST", open_ms + 300_000)
    _write_fragment(other, "KXBTC15M-OTHER", open_ms + 900_000)
    windows = prepare_windows([str(second), str(first), str(other)])
    assert len(windows) == 2
    grouped = {window.quality.ticker: window for window in windows}
    assert grouped["KXBTC15M-TEST"].quality.fragments == 2
    assert "fragmented" in grouped["KXBTC15M-TEST"].quality.flags
    assert grouped["KXBTC15M-OTHER"].quality.fragments == 1
    # Fragment rows are stitched chronologically.
    stitched = grouped["KXBTC15M-TEST"].rows
    timestamps = [int(row["receive_ts_ms"]) for row in stitched]
    assert timestamps == sorted(timestamps)


def test_overlapping_fragments_are_globally_time_sorted(tmp_path):
    # A crashed process's finalizer can append to the OLD file after the new
    # one started: fragment time ranges overlap, and naive concatenation
    # leaves rows non-monotonic (breaking every bisect join downstream).
    from kalshi_mm.features import _load_window, group_recordings_by_market

    base = 1_800_000_000_000
    first = tmp_path / "events_a.jsonl"
    second = tmp_path / "events_b.jsonl"
    _write_fragment(first, "KXBTC15M-TEST", base)
    _write_fragment(second, "KXBTC15M-TEST", base + 10)  # overlaps first's range
    ((market, paths),) = group_recordings_by_market([str(first), str(second)])
    window = _load_window(market, paths)
    timestamps = [int(row["receive_ts_ms"]) for row in window.rows]
    assert timestamps == sorted(timestamps)


def test_reconnects_flag_window(tmp_path):
    path = tmp_path / "events_a.jsonl"
    _write_fragment(path, "KXBTC15M-TEST", 1_800_000_000_000, reconnects=1)
    (window,) = prepare_windows([str(path)])
    assert window.quality.kalshi_reconnects == 1
    assert "reconnect" in window.quality.flags
