import pytest

from kalshi_mm.labels import fill_label_rows, generate_labels
from kalshi_mm.replay import ReplayFill, ReplayResult

pytest.importorskip("pyarrow")


def _result(fills, mids):
    return ReplayResult(
        decisions=1,
        orders_posted=1,
        contracts_posted=1.0,
        fills=tuple(fills),
        contracts_filled=sum(f.count for f in fills),
        fill_rate=1.0,
        final_position=0.0,
        cash_flow=0.0,
        fees_paid=0.0,
        latency_ms=150,
        maker_fee_multiplier=1.0,
        gross_fair_edge=0.0,
        markout_1s=None,
        markout_5s=None,
        settlement_pnl=None,
        settlement_result=None,
        settlement_source=None,
        settlement_reference=None,
        fair_mid_mae=None,
        mids=tuple(mids),
    )


def test_fill_labels_side_adjust_markout_and_join_features():
    # Bid fill at 0.49; mid drops to 0.45 by +5s -> markout -0.04 -> toxic.
    fills = [ReplayFill(ts_ms=10_000, side="bid", price=0.49, count=5.0, fair_yes=0.5)]
    mids = [(9_000, 0.50), (11_000, 0.50), (15_000, 0.45)]
    features = [
        {"ts_ms": 9_800, "seconds_to_close": 400.0, "imb_top1": -0.5},
        {"ts_ms": 20_000, "seconds_to_close": 390.0, "imb_top1": 0.9},
    ]
    (row,) = fill_label_rows(_result(fills, mids), features, ticker="T")
    assert row["markout_1s"] == pytest.approx(0.50 - 0.49)
    assert row["markout_5s"] == pytest.approx(0.45 - 0.49)
    assert row["toxic_5s"] == 1.0
    assert row["imb_top1"] == -0.5  # joined the preceding row, not the future one
    assert row["feature_age_ms"] == 200

    # Ask fill: same mids, markout sign flips.
    ask_fills = [ReplayFill(ts_ms=10_000, side="ask", price=0.51, count=5.0, fair_yes=0.5)]
    (ask_row,) = fill_label_rows(_result(ask_fills, mids), features, ticker="T")
    assert ask_row["markout_5s"] == pytest.approx(0.51 - 0.45)
    assert ask_row["toxic_5s"] == 0.0


def test_feature_row_at_fill_timestamp_is_not_joined():
    # A row emitted at the fill's own ts contains the causal trade in its flow
    # features (leakage); the join must use strictly earlier rows.
    fills = [ReplayFill(ts_ms=10_000, side="bid", price=0.49, count=1.0, fair_yes=0.5)]
    mids = [(9_000, 0.50), (11_500, 0.50), (15_200, 0.50)]
    features = [
        {"ts_ms": 9_750, "seconds_to_close": 400.0, "imb_top1": 0.1},
        {"ts_ms": 10_000, "seconds_to_close": 400.0, "imb_top1": -0.9},  # contains the fill's trade
    ]
    (row,) = fill_label_rows(_result(fills, mids), features, ticker="T")
    assert row["imb_top1"] == 0.1
    assert row["feature_age_ms"] == 250


def test_markout_is_none_across_data_gaps():
    fills = [ReplayFill(ts_ms=10_000, side="bid", price=0.49, count=1.0, fair_yes=0.5)]
    # Next mid after the 5s horizon is 40s later (gap) -> no 5s label.
    mids = [(9_000, 0.50), (11_500, 0.50), (50_000, 0.30)]
    features = [{"ts_ms": 9_800, "seconds_to_close": 400.0}]
    (row,) = fill_label_rows(_result(fills, mids), features, ticker="T")
    assert row["markout_1s"] == pytest.approx(0.01)
    assert row["markout_5s"] is None
    assert row["toxic_5s"] is None


def test_stale_features_are_not_joined():
    fills = [ReplayFill(ts_ms=50_000, side="bid", price=0.49, count=1.0, fair_yes=0.5)]
    mids = [(49_000, 0.50), (56_000, 0.50)]
    features = [{"ts_ms": 10_000, "seconds_to_close": 500.0}]
    assert fill_label_rows(_result(fills, mids), features, ticker="T") == []


def test_generate_labels_end_to_end(tmp_path, active_recording):
    out = tmp_path / "labels"
    written = generate_labels([str(active_recording)], out, latency_ms=0, maker_fee_multiplier=0.0)
    assert len(written) == 1
    import pyarrow.parquet as pq

    table = pq.read_table(written[0])
    assert table.num_rows > 0
    names = set(table.column_names)
    assert {"fill_side", "fill_price", "markout_5s", "toxic_5s", "imb_top1", "seconds_to_close"} <= names
    # Idempotent second run.
    assert generate_labels([str(active_recording)], out) == []
