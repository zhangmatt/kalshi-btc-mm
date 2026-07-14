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
    stale = audit(path, now_ms=321_001, require_live=True)
    assert not stale.healthy
    assert any("no active recording" in value for value in stale.errors)


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
