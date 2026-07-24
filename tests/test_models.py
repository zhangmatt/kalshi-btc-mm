import math
import random

import numpy as np
import pytest

from kalshi_mm.models import (
    LinearModel,
    block_bootstrap_ci,
    bucket_of,
    chronological_split,
    fit_logistic,
    fit_ridge,
    load_models,
    save_models,
)


def test_ridge_recovers_known_coefficients():
    rng = np.random.default_rng(3)
    n = 5_000
    x = rng.normal(size=(n, 3))
    y = 2.0 * x[:, 0] - 1.0 * x[:, 1] + 0.0 * x[:, 2] + rng.normal(scale=0.1, size=n)
    model = fit_ridge(x, y, features=("a", "b", "c"), bucket=(90.0, 300.0), l2=1e-6)
    assert model.coefs[0] == pytest.approx(2.0, abs=0.05)
    assert model.coefs[1] == pytest.approx(-1.0, abs=0.05)
    assert abs(model.coefs[2]) < 0.05
    prediction = model.predict([1.0, 0.0, 0.0])
    assert prediction == pytest.approx(model.intercept + 2.0 * (1.0 - model.means[0]) / model.stds[0], abs=0.05)


def test_ridge_prediction_with_missing_feature_is_none():
    model = fit_ridge(
        np.random.default_rng(0).normal(size=(100, 2)),
        np.zeros(100),
        features=("a", "b"),
        bucket=(90.0, 300.0),
    )
    assert model.predict([1.0, None]) is None


def test_logistic_separates_and_calibrates():
    rng = np.random.default_rng(5)
    n = 4_000
    x = rng.normal(size=(n, 2))
    logits = 1.5 * x[:, 0] - 0.5
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logits))).astype(float)
    model = fit_logistic(x, y, features=("a", "b"), bucket=(90.0, 300.0), l2=1e-3)
    high = model.predict([2.0, 0.0])
    low = model.predict([-2.0, 0.0])
    assert high is not None and low is not None and high > 0.7 > 0.3 > low
    assert abs(model.coefs[1]) < 0.15
    assert model.meta["base_rate"] == pytest.approx(float(y.mean()))


def test_model_json_round_trip(tmp_path):
    model = fit_ridge(
        np.random.default_rng(1).normal(size=(200, 2)),
        np.random.default_rng(2).normal(size=200),
        features=("a", "b"),
        bucket=(300.0, 600.0),
    )
    path = tmp_path / "m.json"
    save_models([model], path)
    (loaded,) = load_models(path)
    assert loaded.features == ("a", "b")
    assert loaded.predict([0.5, 0.5]) == pytest.approx(model.predict([0.5, 0.5]))


def test_chronological_split_preserves_given_order():
    # Input order is authoritative (callers order by data timestamp). Ticker
    # strings like 26AUG.../26JUL... do NOT sort chronologically, so the split
    # must never re-sort.
    ids = ["26JUL14-A", "26JUL14-B", "26JUL31-C", "26AUG01-D", "26AUG02-E"]
    train, val, holdout = chronological_split(ids)
    assert train == ["26JUL14-A", "26JUL14-B", "26JUL31-C"]
    assert val == ["26AUG01-D"]
    assert holdout == ["26AUG02-E"]
    # Duplicates collapse without reordering.
    assert chronological_split(["b", "a", "b", "c", "d", "e"])[0] == ["b", "a", "c"]


def test_columnar_feature_store_orders_filters_and_fills_missing(tmp_path):
    import numpy as np

    pytest.importorskip("pyarrow")
    from kalshi_mm.features import write_parquet
    from kalshi_mm.fit import load_feature_store, window_order

    for i, ticker in enumerate(["KXBTC15M-A", "KXBTC15M-B", "KXBTC15M-C"]):
        rows = [{"ticker": ticker, "ts_ms": i * 1_000_000 + j, "imb_top1": float(j)} for j in range(20)]
        write_parquet(rows, tmp_path / f"{ticker}.parquet")
    pattern = str(tmp_path / "*.parquet")
    # Chronological order by first ts (NOT ticker string).
    assert [t for t, _ in window_order(pattern)] == ["KXBTC15M-A", "KXBTC15M-B", "KXBTC15M-C"]
    # Ticker filter skips the holdout window entirely.
    store = load_feature_store(pattern, tickers={"KXBTC15M-A", "KXBTC15M-B"})
    assert set(store) == {"KXBTC15M-A", "KXBTC15M-B"}
    # Columns are unboxed float64 arrays; a present column round-trips.
    assert store["KXBTC15M-A"]["imb_top1"].dtype == np.float64
    assert np.array_equal(store["KXBTC15M-A"]["imb_top1"], np.arange(20.0))
    # A column absent from the parquet is filled with NaN, not dropped.
    assert np.isnan(store["KXBTC15M-A"]["ext_imbalance"]).all()


def test_block_bootstrap_ci_covers_mean():
    values = [1.0, 2.0, 3.0, 4.0, 5.0] * 10
    mean, low, high = block_bootstrap_ci(values)
    assert mean == pytest.approx(3.0)
    assert low < 3.0 < high
    assert high - low < 1.5


def test_mirror_features_aligns_adverse_pressure():
    from kalshi_mm.models import mirror_features

    row = {
        "imb_top1": -0.6,  # book leaning down
        "imb_top3": -0.3,
        "microprice_minus_mid": -0.002,
        "flow_1s": -50.0,
        "flow_5s": -200.0,
        "bid_size_delta": -40.0,  # bid depth evaporating
        "ask_size_delta": 25.0,   # ask depth building
        "ext_micro_lead_bps": -1.5,
        "ext_imbalance": -0.4,
        "gbm_fair_minus_mid": -0.01,
        "spread": 0.01,
        "trades_5s": 80,
    }
    bid = mirror_features(row, "bid")
    ask = mirror_features(row, "ask")
    # Downward pressure is adverse (positive) for resting bids, benign for asks.
    assert bid["adv_imb_top1"] == 0.6 and ask["adv_imb_top1"] == -0.6
    assert bid["adv_flow_5s"] == 200.0 and ask["adv_flow_5s"] == -200.0
    # Own bid depth evaporating and ask depth building are both adverse for bids.
    assert bid["adv_own_depth_delta"] == 40.0
    assert bid["adv_opp_depth_delta"] == 25.0
    # For asks, "own" depth is the ask side (which grew -> benign/negative).
    assert ask["adv_own_depth_delta"] == -25.0
    assert ask["adv_opp_depth_delta"] == -40.0
    # Symmetric features pass through unchanged.
    assert bid["spread"] == ask["spread"] == 0.01
    # None propagates rather than fabricating a value.
    assert mirror_features({"spread": 0.01}, "bid")["adv_imb_top1"] is None
    with pytest.raises(ValueError):
        mirror_features(row, "yes")


def test_bucket_of_boundaries():
    assert bucket_of(90.0) == (90.0, 300.0)
    assert bucket_of(299.9) == (90.0, 300.0)
    assert bucket_of(300.0) == (300.0, 600.0)
    assert bucket_of(899.9) == (600.0, 900.0)
    assert bucket_of(89.0) is None
    assert bucket_of(900.0) is None
