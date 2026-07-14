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


def test_build_variants_parses_spec(tmp_path):
    variants = build_variants("baseline,micro25,micro50", None)
    assert [v.name for v in variants] == ["baseline", "micro25", "micro50"]
    with pytest.raises(SystemExit):
        build_variants("toxgate", str(tmp_path / "missing.json"))
    with pytest.raises(SystemExit):
        build_variants("nonsense", None)


def test_run_ab_end_to_end(tmp_path, active_recording, capsys):
    variants = build_variants("baseline,micro50", None)
    outcomes = run_ab([str(active_recording)], variants, latency_ms=0, maker_fee_multiplier=0.0)
    assert {outcome.variant for outcome in outcomes} == {"baseline", "micro50"}
    summarize(outcomes)
    out = capsys.readouterr().out
    assert "A/B summary" in out
    assert "micro50" in out
