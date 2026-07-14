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


def test_chronological_split_orders_and_partitions():
    ids = [f"W{i:03d}" for i in range(10)]
    random.Random(0).shuffle(ids)
    train, val, holdout = chronological_split(ids)
    assert train == [f"W{i:03d}" for i in range(6)]
    assert val == ["W006", "W007"]
    assert holdout == ["W008", "W009"]


def test_block_bootstrap_ci_covers_mean():
    values = [1.0, 2.0, 3.0, 4.0, 5.0] * 10
    mean, low, high = block_bootstrap_ci(values)
    assert mean == pytest.approx(3.0)
    assert low < 3.0 < high
    assert high - low < 1.5


def test_bucket_of_boundaries():
    assert bucket_of(90.0) == (90.0, 300.0)
    assert bucket_of(299.9) == (90.0, 300.0)
    assert bucket_of(300.0) == (300.0, 600.0)
    assert bucket_of(899.9) == (600.0, 900.0)
    assert bucket_of(89.0) is None
    assert bucket_of(900.0) is None
