import numpy as np
import pytest

from kalshi_mm.ab import MicropriceSkew, ToxicityGate, build_variants, run_ab, summarize
from kalshi_mm.exchange import KalshiBbo, KalshiOrderBook
from kalshi_mm.models import fit_logistic, save_models
from kalshi_mm.replay import QuoteContext

pytest.importorskip("pyarrow")


def _context(bid_size=10.0, ask_size=30.0):
    book = KalshiOrderBook("T")
    book.apply(
        {
            "type": "orderbook_snapshot",
            "seq": 1,
            "msg": {
                "market_ticker": "T",
                "yes_dollars_fp": [["0.49", str(bid_size)]],
                "no_dollars_fp": [["0.51", str(ask_size)]],
            },
        }
    )
    return QuoteContext(
        ts_ms=1_000,
        seconds_to_close=400.0,
        fair_yes=0.50,
        bbo=KalshiBbo(0.49, 0.51, bid_size=bid_size, ask_size=ask_size, valid=True),
        book=book,
        flow_5s=-5.0,
        flow_1s=-2.0,
    )


def test_microprice_skew_leans_toward_heavy_side():
    ctx = _context(bid_size=10.0, ask_size=30.0)  # ask-heavy -> microprice below mid
    fair, bid_tox, ask_tox = MicropriceSkew(0.5).adjust(ctx)
    assert fair < ctx.fair_yes
    assert bid_tox == ask_tox == 0.0
    fair_up, _, _ = MicropriceSkew(0.5).adjust(_context(bid_size=30.0, ask_size=10.0))
    assert fair_up > 0.50


def test_toxicity_gate_penalizes_leaned_against_side(tmp_path):
    # A model that predicts toxic when adverse-side pressure is positive.
    from kalshi_mm.models import T_MIRRORED_FEATURES, load_models

    rng = np.random.default_rng(0)
    x = rng.normal(size=(800, len(T_MIRRORED_FEATURES)))
    y = (x[:, 0] > 0).astype(float)  # adv_imb_top1 > 0 -> toxic
    model = fit_logistic(x, y, features=T_MIRRORED_FEATURES, bucket=(300.0, 600.0), l2=1e-3)
    path = tmp_path / "model_t.json"
    save_models([model], path)
    gate = ToxicityGate(load_models(path))
    # Ask-heavy book: imbalance negative -> adverse for resting BIDS only.
    ctx = _context(bid_size=10.0, ask_size=30.0)
    fair, bid_tox, ask_tox = gate.adjust(ctx)
    assert fair == ctx.fair_yes
    assert bid_tox > 0.0
    assert bid_tox > ask_tox
    # Outside every model bucket: no tax.
    from dataclasses import replace

    far = replace(ctx, seconds_to_close=800.0)
    assert gate.adjust(far) == (ctx.fair_yes, 0.0, 0.0)


def test_composed_adjuster_applies_skew_and_taxes(tmp_path):
    from kalshi_mm.ab import ComposedAdjuster
    from kalshi_mm.models import T_MIRRORED_FEATURES, save_models, load_models

    rng = np.random.default_rng(2)
    x = rng.normal(size=(800, len(T_MIRRORED_FEATURES)))
    y = (x[:, 0] > 0).astype(float)
    model = fit_logistic(x, y, features=T_MIRRORED_FEATURES, bucket=(300.0, 600.0), l2=1e-3)
    path = tmp_path / "t.json"
    save_models([model], path)
    combo = ComposedAdjuster([MicropriceSkew(0.25), ToxicityGate(load_models(path))], "combo25")
    ctx = _context(bid_size=10.0, ask_size=30.0)  # ask-heavy: micro below mid, bids toxic
    fair, bid_tox, ask_tox = combo.adjust(ctx)
    assert fair < ctx.fair_yes  # skew applied
    assert bid_tox > ask_tox >= 0.0  # gate applied on the adverse side


def test_build_variants_parses_spec(tmp_path):
    variants = build_variants("baseline,micro25,micro50", None)
    assert [v.name for v in variants] == ["baseline", "micro25", "micro50"]
    with pytest.raises(SystemExit):
        build_variants("combo25", None)


def test_external_lead_skew_shifts_fair_toward_the_lead(tmp_path):
    from dataclasses import replace
    from kalshi_mm.ab import ExternalLeadSkew

    ctx = _context()
    ctx = replace(ctx, seconds_to_close=400.0, sigma_per_s=2e-4, fair_yes=0.50, ext_micro_lead_bps=5.0)
    up, _, _ = ExternalLeadSkew(1.0).adjust(ctx)
    assert up > 0.50  # external leads BRTI upward -> raise P(yes)
    down, _, _ = ExternalLeadSkew(1.0).adjust(replace(ctx, ext_micro_lead_bps=-5.0))
    assert down < 0.50
    # No external signal -> no change.
    assert ExternalLeadSkew(1.0).adjust(replace(ctx, ext_micro_lead_bps=None))[0] == 0.50


def test_plus_composes_variants(tmp_path):
    from kalshi_mm.ab import ComposedAdjuster, build_variants
    from kalshi_mm.models import T_MIRRORED_FEATURES, save_models

    rng = np.random.default_rng(4)
    model = fit_logistic(
        rng.normal(size=(300, len(T_MIRRORED_FEATURES))),
        (rng.uniform(size=300) > 0.6).astype(float),
        features=T_MIRRORED_FEATURES, bucket=(300.0, 600.0),
    )
    path = tmp_path / "t.json"
    save_models([model], path)
    (composed,) = build_variants("extlead50+toxgate6", str(path))
    assert isinstance(composed, ComposedAdjuster)
    assert [type(p).__name__ for p in composed.parts] == ["ExternalLeadSkew", "ToxicityGate"]


def test_sweep_variant_specs_parse_scales_and_overrides(tmp_path):
    from kalshi_mm.ab import ComposedAdjuster, build_variants
    from kalshi_mm.models import T_MIRRORED_FEATURES, save_models

    rng = np.random.default_rng(3)
    model = fit_logistic(
        rng.normal(size=(300, len(T_MIRRORED_FEATURES))),
        (rng.uniform(size=300) > 0.6).astype(float),
        features=T_MIRRORED_FEATURES,
        bucket=(300.0, 600.0),
    )
    path = tmp_path / "t.json"
    save_models([model], path)
    combo1, wide2, close30, tox4 = build_variants("combo25x1,wide2,close30,toxgate4", str(path))
    assert isinstance(combo1, ComposedAdjuster)
    assert combo1.parts[1].scale_dollars == 0.01  # x1 = 1-cent gate scale
    assert combo1.parts[0].weight == 0.25
    assert wide2.strategy_overrides == {"min_edge_dollars": 0.02}
    assert close30.strategy_overrides == {"stop_quote_before_close_s": 30.0}
    # Gate-only variant with a cent-scale suffix, no microprice component.
    assert tox4.name == "toxgate4" and tox4.scale_dollars == 0.04
    from kalshi_mm.ab import ToxicityGate
    assert isinstance(tox4, ToxicityGate)
    with pytest.raises(SystemExit):
        build_variants("toxgate", str(tmp_path / "missing.json"))
    with pytest.raises(SystemExit):
        build_variants("modelv", None, str(tmp_path / "missing.json"))
    with pytest.raises(SystemExit):
        build_variants("nonsense", None)


def test_model_v_variant_centers_on_mid_plus_prediction(tmp_path):
    from kalshi_mm.ab import ModelVSkew
    from kalshi_mm.models import FEATURE_SETS, fit_ridge

    features = FEATURE_SETS["book"]
    rng = np.random.default_rng(1)
    x = rng.normal(size=(500, len(features)))
    y = 0.02 * x[:, list(features).index("imb_top1")]  # move follows imbalance
    model = fit_ridge(x, y, features=features, bucket=(300.0, 600.0), l2=1e-6)
    model.meta["feature_set"] = "book"
    variant = ModelVSkew([model])
    ctx = _context(bid_size=30.0, ask_size=10.0)  # bid-heavy: positive imbalance
    fair, bid_tox, ask_tox = variant.adjust(ctx)
    assert bid_tox == ask_tox == 0.0
    assert fair > 0.50  # centered above mid when the book leans up
    # No bucket match -> passthrough.
    from dataclasses import replace

    assert variant.adjust(replace(ctx, seconds_to_close=800.0))[0] == ctx.fair_yes


def test_run_ab_end_to_end(tmp_path, active_recording, capsys):
    variants = build_variants("baseline,micro50", None)
    outcomes = run_ab([str(active_recording)], variants, latency_ms=0, maker_fee_multiplier=0.0)
    assert {outcome.variant for outcome in outcomes} == {"baseline", "micro50"}
    summarize(outcomes)
    out = capsys.readouterr().out
    assert "A/B summary" in out
    assert "micro50" in out
