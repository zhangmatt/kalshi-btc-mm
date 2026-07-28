import numpy as np
import pytest

from kalshi_mm.exchange import KalshiBbo, KalshiOrderBook
from kalshi_mm.models import T_MIRRORED_FEATURES, fit_logistic, save_models


def _book(bid_size, ask_size):
    book = KalshiOrderBook("T")
    book.apply({
        "type": "orderbook_snapshot", "seq": 1,
        "msg": {"market_ticker": "T",
                "yes_dollars_fp": [["0.49", str(bid_size)]],
                "no_dollars_fp": [["0.51", str(ask_size)]]},
    })
    return book


def _toxic_when_adverse_model(tmp_path):
    # Model predicts toxic when adv_imb_top1 (first mirrored feature) > 0.
    rng = np.random.default_rng(0)
    x = rng.normal(size=(800, len(T_MIRRORED_FEATURES)))
    y = (x[:, 0] > 0).astype(float)
    model = fit_logistic(x, y, features=T_MIRRORED_FEATURES, bucket=(300.0, 600.0), l2=1e-3)
    path = tmp_path / "model_t.json"
    save_models([model], path)
    return str(path)


def test_live_gate_taxes_the_side_the_book_leans_against(tmp_path):
    from kalshi_mm.live_gate import LiveToxicityGate

    gate = LiveToxicityGate(_toxic_when_adverse_model(tmp_path), scale_dollars=0.06)
    # Ask-heavy book: negative imbalance -> downward pressure -> adverse for bids.
    book = _book(10, 30)
    bbo = KalshiBbo(0.49, 0.51, bid_size=10.0, ask_size=30.0, valid=True)
    bid_tox, ask_tox = gate.toxicity_dollars(
        now_ms=1_000, book=book, bbo=bbo, fair_yes=0.50, seconds_to_close=400.0
    )
    assert bid_tox > ask_tox
    assert bid_tox > 0.0
    assert bid_tox <= 0.06  # bounded by the scale


def test_live_gate_flow_window_prunes(tmp_path):
    from kalshi_mm.live_gate import LiveToxicityGate

    gate = LiveToxicityGate(_toxic_when_adverse_model(tmp_path))
    gate.on_trade(1_000, "bid", 5.0)
    gate.on_trade(2_000, "ask", 3.0)
    gate.on_trade(10_000, "bid", 1.0)  # >5s after the first two -> they prune
    assert len(gate._flow) == 1


def test_live_gate_off_bucket_returns_zero(tmp_path):
    from kalshi_mm.live_gate import LiveToxicityGate

    gate = LiveToxicityGate(_toxic_when_adverse_model(tmp_path))
    bbo = KalshiBbo(0.49, 0.51, bid_size=10.0, ask_size=30.0, valid=True)
    # seconds_to_close outside the single model bucket -> no tax.
    bid_tox, ask_tox = gate.toxicity_dollars(
        now_ms=1_000, book=_book(10, 30), bbo=bbo, fair_yes=0.50, seconds_to_close=800.0
    )
    assert bid_tox == 0.0 and ask_tox == 0.0
