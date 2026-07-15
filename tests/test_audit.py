import json
from datetime import datetime, timezone

from kalshi_mm.audit import audit


def test_audit_accepts_complete_research_recording(tmp_path):
    path = tmp_path / "events.jsonl"
    open_time = datetime.fromtimestamp(100, timezone.utc).isoformat()
    close_time = datetime.fromtimestamp(200, timezone.utc).isoformat()
    rows = [
        {"receive_ts_ms": 100_000, "source": "system", "event_type": "market_metadata", "payload": {"open_time": open_time, "close_time": close_time}},
        {"receive_ts_ms": 100_001, "source": "kalshi_rest", "event_type": "series_metadata", "payload": {}},
        {"receive_ts_ms": 100_002, "source": "kalshi_rest", "event_type": "series_fee_changes", "payload": {}},
        {"receive_ts_ms": 100_003, "source": "kalshi_rest", "event_type": "market_incentives", "payload": {}},
    ]
    for channel in ("orderbook_delta", "ticker", "trade", "cfbenchmarks_value", "market_lifecycle_v2"):
        rows.append({"receive_ts_ms": 100_004, "source": "kalshi", "event_type": "subscribed", "payload": {"msg": {"channel": channel}}})
    rows.extend(
        [
            {"receive_ts_ms": 100_010, "source": "kalshi", "event_type": "orderbook_snapshot", "payload": {}},
            {"receive_ts_ms": 100_011, "source": "kalshi", "event_type": "orderbook_delta", "payload": {}},
            {"receive_ts_ms": 100_012, "source": "kalshi", "event_type": "cfbenchmarks_value", "payload": {}},
            *({"receive_ts_ms": 100_013, "source": venue, "event_type": "spot", "payload": {}} for venue in ("binance", "coinbase", "kraken")),
            {"receive_ts_ms": 201_000, "source": "system", "event_type": "market_status_at_close", "payload": {}},
        ]
    )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = audit(path, now_ms=201_000)
    assert result.healthy
    assert result.status == "complete"
    # Beyond the 300s gap tolerance a complete-and-old newest file must fail.
    stale = audit(path, now_ms=501_000, require_live=True)
    assert not stale.healthy
    assert any("no active recording" in value for value in stale.errors)
    # Within the gap tolerance it must NOT fail for that reason (restart storm
    # guard); other liveness errors may still apply.
    gap = audit(path, now_ms=321_001, require_live=True)
    assert not any("no active recording" in value for value in gap.errors)


def test_audit_requires_configured_external_depth(tmp_path):
    path = tmp_path / "events.jsonl"
    open_time = datetime.fromtimestamp(100, timezone.utc).isoformat()
    close_time = datetime.fromtimestamp(200, timezone.utc).isoformat()
    rows = [
        {"receive_ts_ms": 100_000, "source": "system", "event_type": "market_metadata", "payload": {"open_time": open_time, "close_time": close_time}},
        {"receive_ts_ms": 100_000, "source": "system", "event_type": "collector_config", "payload": {"external_depth_enabled": True, "external_depth_venues": ["binance", "coinbase", "kraken"]}},
        {"receive_ts_ms": 100_001, "source": "kalshi_rest", "event_type": "series_metadata", "payload": {}},
        {"receive_ts_ms": 100_002, "source": "kalshi_rest", "event_type": "series_fee_changes", "payload": {}},
        {"receive_ts_ms": 100_003, "source": "kalshi_rest", "event_type": "market_incentives", "payload": {}},
    ]
    for channel in ("orderbook_delta", "ticker", "trade", "cfbenchmarks_value", "market_lifecycle_v2"):
        rows.append({"receive_ts_ms": 100_004, "source": "kalshi", "event_type": "subscribed", "payload": {"msg": {"channel": channel}}})
    rows.extend(
        [
            {"receive_ts_ms": 100_010, "source": "kalshi", "event_type": "orderbook_snapshot", "payload": {}},
            {"receive_ts_ms": 100_011, "source": "kalshi", "event_type": "orderbook_delta", "payload": {}},
            {"receive_ts_ms": 100_012, "source": "kalshi", "event_type": "cfbenchmarks_value", "payload": {}},
            *({"receive_ts_ms": 100_013, "source": venue, "event_type": "spot", "payload": {}} for venue in ("binance", "coinbase", "kraken")),
            *({"receive_ts_ms": 100_014, "source": venue, "event_type": "external_orderbook", "payload": {}} for venue in ("binance", "coinbase")),
            {"receive_ts_ms": 201_000, "source": "system", "event_type": "market_status_at_close", "payload": {}},
        ]
    )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = audit(path, now_ms=201_000)
    assert not result.healthy
    assert any("missing external L2 data: kraken" in value for value in result.errors)


def _base_rows(*, with_close: bool) -> list[dict]:
    open_time = datetime.fromtimestamp(100, timezone.utc).isoformat()
    close_time = datetime.fromtimestamp(200, timezone.utc).isoformat()
    rows = [
        {"receive_ts_ms": 100_000, "source": "system", "event_type": "market_metadata", "payload": {"open_time": open_time, "close_time": close_time}},
        {"receive_ts_ms": 100_001, "source": "kalshi_rest", "event_type": "series_metadata", "payload": {}},
        {"receive_ts_ms": 100_002, "source": "kalshi_rest", "event_type": "series_fee_changes", "payload": {}},
        {"receive_ts_ms": 100_003, "source": "kalshi_rest", "event_type": "market_incentives", "payload": {}},
    ]
    for channel in ("orderbook_delta", "ticker", "trade", "cfbenchmarks_value", "market_lifecycle_v2"):
        rows.append({"receive_ts_ms": 100_004, "source": "kalshi", "event_type": "subscribed", "payload": {"msg": {"channel": channel}}})
    rows.extend(
        [
            {"receive_ts_ms": 100_010, "source": "kalshi", "event_type": "orderbook_snapshot", "payload": {}},
            {"receive_ts_ms": 100_011, "source": "kalshi", "event_type": "orderbook_delta", "payload": {}},
            {"receive_ts_ms": 100_012, "source": "kalshi", "event_type": "cfbenchmarks_value", "payload": {}},
            *({"receive_ts_ms": 100_013, "source": venue, "event_type": "spot", "payload": {}} for venue in ("binance", "coinbase", "kraken")),
        ]
    )
    if with_close:
        rows.append({"receive_ts_ms": 201_000, "source": "system", "event_type": "market_status_at_close", "payload": {}})
    return rows


def test_dead_collector_never_passes_require_live(tmp_path):
    # No close marker and the market closed long ago: a live collector would
    # have produced either a close marker or a newer file.
    path = tmp_path / "events.jsonl"
    rows = _base_rows(with_close=False)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    stale = audit(path, now_ms=400_000, require_live=True)
    assert stale.status == "partial"
    assert not stale.healthy
    assert any("collector appears dead" in value for value in stale.errors)
    offline = audit(path, now_ms=400_000)
    assert offline.healthy  # research-mode audits still tolerate fragments


def test_channel_sequence_gap_is_reported(tmp_path):
    path = tmp_path / "events.jsonl"
    rows = _base_rows(with_close=True)
    rows.insert(-1, {"receive_ts_ms": 100_020, "source": "kalshi", "event_type": "trade", "payload": {"type": "trade", "seq": 1, "sid": 4, "msg": {}}})
    rows.insert(-1, {"receive_ts_ms": 100_021, "source": "kalshi", "event_type": "trade", "payload": {"type": "trade", "seq": 3, "sid": 4, "msg": {}}})
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = audit(path, now_ms=201_000)
    assert result.channel_seq_gaps == {"trade": 1}
    assert any("trade channel skipped" in value for value in result.warnings)
