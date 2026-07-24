import numpy as np
import pytest

pytest.importorskip("pyarrow")

from kalshi_mm.features import write_parquet
from kalshi_mm.models import T_MIRRORED_FEATURES, load_models


def _feature_window(ticker: str, base_ts: int, outcome: float, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for j in range(400):
        stc = 880.0 - j * 2.0  # spans all three time buckets
        imb = float(rng.normal())
        # forward move correlates with imbalance (V signal) + noise
        move = 0.01 * imb + 0.01 * float(rng.normal())
        rows.append({
            "ticker": ticker,
            "ts_ms": base_ts + j * 250,
            "seconds_to_close": stc,
            "fwd_mid_delta_5s": move,
            "outcome_yes": outcome,
            "gbm_fair": 0.5,
            "mid": 0.5,
            "imb_top1": imb,
            "imb_top3": float(rng.normal()),
            "microprice_minus_mid": float(rng.normal()) * 0.002,
            "flow_1s": float(rng.normal()) * 10,
            "flow_5s": float(rng.normal()) * 30,
            "bid_size_delta": float(rng.normal()) * 50,
            "ask_size_delta": float(rng.normal()) * 50,
            "spread": 0.01,
            "gbm_fair_minus_mid": float(rng.normal()) * 0.01,
            "jump_ratio": 0.8,
            "ext_basis_bps": float(rng.normal()),
            "ext_micro_lead_bps": float(rng.normal()),
            "ext_imbalance": float(rng.normal()),
            "ext_dispersion_bps": abs(float(rng.normal())),
            "trades_5s": float(rng.integers(0, 100)),
        })
    return rows


def _fill_rows(ticker: str, base_ts: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed + 999)
    rows = []
    for k in range(40):
        toxic = float(rng.uniform() < 0.35)
        rows.append({
            "ticker": ticker,
            "fill_ts_ms": base_ts + k * 2000,
            "fill_side": "bid" if k % 2 else "ask",
            "seconds_to_close": 500.0 - k * 5,
            "toxic_5s": toxic,
            "imb_top1": float(rng.normal()),
            "imb_top3": float(rng.normal()),
            "microprice_minus_mid": float(rng.normal()) * 0.002,
            "flow_1s": float(rng.normal()) * 10,
            "flow_5s": float(rng.normal()) * 30,
            "bid_size_delta": float(rng.normal()) * 50,
            "ask_size_delta": float(rng.normal()) * 50,
            "spread": 0.01,
            "gbm_fair_minus_mid": float(rng.normal()) * 0.01,
            "ext_micro_lead_bps": float(rng.normal()),
            "ext_imbalance": float(rng.normal()),
            "trades_5s": float(rng.integers(0, 100)),
        })
    return rows


def test_fit_main_columnar_end_to_end(tmp_path, capsys):
    from kalshi_mm import fit

    feat_dir = tmp_path / "features"
    lbl_dir = tmp_path / "labels"
    for i in range(15):
        ticker = f"KXBTC15M-W{i:02d}"
        base = 1_800_000_000_000 + i * 1_000_000
        write_parquet(_feature_window(ticker, base, float(i % 2), i), feat_dir / f"{ticker}.parquet")
        write_parquet(_fill_rows(ticker, base, i), lbl_dir / f"{ticker}.fills.parquet")

    out = tmp_path / "models"
    fit.main([
        "--features", str(feat_dir / "*.parquet"),
        "--labels", str(lbl_dir / "*.fills.parquet"),
        "--out", str(out),
    ])
    printed = capsys.readouterr().out
    # Split is chronological and holdout is not loaded.
    assert "windows: train=9 val=3 holdout=3" in printed
    assert "P1 calibration" in printed and "Model V" in printed and "Model T" in printed
    # Models serialized and reloadable.
    v = load_models(out / "model_v.json")
    t = load_models(out / "model_t_state.json")
    assert v and t
    assert set(t[0].features) == set(T_MIRRORED_FEATURES)
    # V recovers the planted imbalance signal (positive coef on imb_top1).
    imb_models = [m for m in v if m.meta.get("feature_set") == "book"]
    assert imb_models
    assert imb_models[0].coefs[imb_models[0].features.index("imb_top1")] > 0


def test_vector_predict_matches_scalar():
    from kalshi_mm.fit import _vector_predict
    from kalshi_mm.models import fit_ridge

    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 3))
    y = x[:, 0] - 0.5 * x[:, 1]
    model = fit_ridge(x, y, features=("a", "b", "c"), bucket=(90.0, 300.0), l2=1e-6)
    probe = rng.normal(size=(5, 3))
    vec = _vector_predict(model, probe)
    for i in range(5):
        assert vec[i] == pytest.approx(model.predict(list(probe[i])), abs=1e-9)
