import json

import pytest

from kalshi_mm.recorder import EventRecorder
from kalshi_mm.features import compact_windows, window_feature_rows
from kalshi_mm.research import prepare_windows

pytest.importorskip("pyarrow")


def _write_recording(path) -> None:
    start_s = 1_800_000_000
    metadata = {
        "ticker": "KXBTC15M-TEST",
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
        "result": "yes",
    }
    with EventRecorder(path) as recorder:
        recorder.write(
            source="system", event_type="market_metadata", payload=metadata, receive_ts_ms=start_s * 1000
        )
        recorder.write(
            source="kalshi",
            event_type="orderbook_snapshot",
            payload={
                "type": "orderbook_snapshot",
                "seq": 1,
                "msg": {
                    "market_ticker": "KXBTC15M-TEST",
                    "yes_dollars_fp": [["0.49", "10"], ["0.48", "5"]],
                    "no_dollars_fp": [["0.51", "20"], ["0.52", "5"]],
                },
            },
            receive_ts_ms=start_s * 1000,
        )
        for second in range(12):
            ts_ms = (start_s + second) * 1000
            recorder.write(
                source="kalshi",
                event_type="cfbenchmarks_value",
                payload={
                    "type": "cfbenchmarks_value",
                    "msg": {
                        "index_id": "BRTI",
                        "data": json.dumps({"time": ts_ms, "value": str(100.0 + second * 0.01)}),
                    },
                },
                exchange_ts_ms=ts_ms,
                receive_ts_ms=ts_ms,
            )
            recorder.write(
                source="binance",
                event_type="spot",
                payload={"price": 100.02 + second * 0.01},
                exchange_ts_ms=ts_ms,
                receive_ts_ms=ts_ms + 1,
            )
            recorder.write(
                source="coinbase",
                event_type="external_orderbook",
                payload={
                    "symbol": "BTC-USD",
                    "mid": 100.03 + second * 0.01,
                    "microprice": 100.04 + second * 0.01,
                    "depth_imbalance": 0.25,
                },
                exchange_ts_ms=ts_ms,
                receive_ts_ms=ts_ms + 2,
            )
            recorder.write(
                source="kalshi",
                event_type="trade",
                payload={
                    "type": "trade",
                    "msg": {
                        "market_ticker": "KXBTC15M-TEST",
                        "yes_price_dollars": "0.49",
                        "count_fp": "3",
                        "taker_book_side": "bid" if second % 2 == 0 else "ask",
                        "ts_ms": ts_ms,
                    },
                },
                exchange_ts_ms=ts_ms,
                receive_ts_ms=ts_ms + 3,
            )
        recorder.write(
            source="system",
            event_type="market_status_at_close",
            payload=metadata,
            receive_ts_ms=(start_s + 900) * 1000,
        )


def test_unfinalized_windows_are_not_compacted(tmp_path):
    path = tmp_path / "events.jsonl"
    _write_recording(path)
    # Strip the close marker to simulate an in-progress recording.
    lines = [line for line in path.read_text().splitlines() if "market_status_at_close" not in line]
    path.write_text("\n".join(lines) + "\n")
    out = tmp_path / "features"
    assert compact_windows([str(path)], out) == []


def test_feature_rows_have_book_flow_external_and_targets(tmp_path):
    path = tmp_path / "events.jsonl"
    _write_recording(path)
    (window,) = prepare_windows([str(path)])
    rows = window_feature_rows(window)
    assert rows, "expected feature rows once volatility warms"
    first = rows[0]
    assert first["mid"] == pytest.approx(0.50)
    assert first["imb_top1"] == pytest.approx((10 - 20) / 30)
    assert first["outcome_yes"] == 1.0
    assert first["ext_basis_bps"] is not None and first["ext_basis_bps"] > 0
    assert first["ext_micro_lead_bps"] is not None
    assert first["ext_imbalance"] == pytest.approx(0.25)
    # Signed flow alternates +3/-3; over 30s it nets to 0 or ±3.
    assert abs(first["flow_30s"]) <= 3.0
    # Forward targets exist for early rows (later mids are recorded) and are
    # None once the horizon runs past the window end.
    assert "fwd_mid_delta_1s" in first
    assert rows[-1]["fwd_mid_delta_30s"] is None


def test_compaction_is_idempotent(tmp_path):
    path = tmp_path / "events.jsonl"
    _write_recording(path)
    out = tmp_path / "features"
    first = compact_windows([str(path)], out)
    assert len(first) == 1 and first[0].exists()
    second = compact_windows([str(path)], out)
    assert second == []  # already compacted, not rewritten
    forced = compact_windows([str(path)], out, overwrite=True)
    assert len(forced) == 1
