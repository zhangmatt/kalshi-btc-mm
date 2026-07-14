import json

from kalshi_mm.replay import replay
from kalshi_mm.research import score_recordings
from kalshi_mm.recorder import EventRecorder


def test_recording_replays_and_scores_end_to_end(tmp_path):
    path = tmp_path / "events.jsonl"
    start_s = 1_800_000_000
    metadata = {
        "ticker": "KXBTC15M-TEST",
        "event_ticker": "KXBTC15M-TEST",
        "title": "BTC price up in next 15 mins?",
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
    with EventRecorder(path) as recorder:
        recorder.write(source="system", event_type="market_metadata", payload=metadata, receive_ts_ms=start_s * 1000)
        recorder.write(
            source="kalshi",
            event_type="orderbook_snapshot",
            payload={
                "type": "orderbook_snapshot",
                "seq": 1,
                "msg": {
                    "market_ticker": "KXBTC15M-TEST",
                    "yes_dollars_fp": [["0.49", "10"]],
                    "no_dollars_fp": [["0.51", "10"]],
                },
            },
            receive_ts_ms=start_s * 1000,
        )
        for second in range(12):
            ts_ms = (start_s + second) * 1000
            price = 100.0 + second * 0.01
            recorder.write(
                source="binance",
                event_type="spot",
                payload={"price": price},
                exchange_ts_ms=ts_ms,
                receive_ts_ms=ts_ms,
            )
            recorder.write(
                source="kalshi",
                event_type="cfbenchmarks_value",
                payload={
                    "type": "cfbenchmarks_value",
                    "msg": {
                        "index_id": "BRTI",
                        "data": json.dumps({"time": ts_ms, "value": str(price)}),
                        "avg_60s_data": {"value": str(price)},
                    },
                },
                exchange_ts_ms=ts_ms,
                receive_ts_ms=ts_ms + 1,
            )
        result = dict(metadata)
        result["result"] = "yes"
        recorder.write(source="system", event_type="market_result", payload=result, receive_ts_ms=(start_s + 1200) * 1000)

    replay_result = replay(path)
    assert replay_result.decisions > 0
    assert replay_result.fair_mid_mae is not None
    scores = score_recordings([path])
    assert {score.model for score in scores} >= {"sigma_1.00", "basis_1.00"}
    assert all(score.samples > 0 for score in scores)
    assert all(score.brier is not None for score in scores)
