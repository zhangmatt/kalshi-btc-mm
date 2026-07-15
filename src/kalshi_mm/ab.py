"""Replay A/B harness: compare quoting variants at identical settings (STRATEGY.md §6).

Each variant runs the same queue-aware replay (fees + latency on) over the same
windows; results are decomposed per window and differences vs baseline get
block-bootstrap confidence intervals over windows. This is the promotion
instrument for Models V and T — a variant is only interesting if it improves
net markout-given-fill without collapsing fill count.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .features import _load_window, _window_is_final, group_recordings_by_market
from .models import LinearModel, block_bootstrap_ci, load_models, mirror_features
from .replay import QuoteAdjuster, QuoteContext, ReplayResult, replay_rows


class MicropriceSkew(QuoteAdjuster):
    """Model V proxy: lean fair toward the Kalshi book's microprice."""

    def __init__(self, weight: float):
        self.weight = weight
        self.name = f"micro{int(weight * 100):02d}"

    def adjust(self, ctx: QuoteContext) -> tuple[float, float, float]:
        bbo = ctx.bbo
        total = bbo.bid_size + bbo.ask_size
        if total <= 0 or bbo.bid is None or bbo.ask is None:
            return ctx.fair_yes, 0.0, 0.0
        mid = (bbo.bid + bbo.ask) / 2.0
        microprice = (bbo.ask * bbo.bid_size + bbo.bid * bbo.ask_size) / total
        return ctx.fair_yes + self.weight * (microprice - mid), 0.0, 0.0


def _bucket_model(models: list[LinearModel], seconds_to_close: float) -> Optional[LinearModel]:
    for model in models:
        if model.bucket[0] <= seconds_to_close < model.bucket[1]:
            return model
    return None


def context_raw_features(ctx: QuoteContext) -> dict[str, Optional[float]]:
    """The training feature vocabulary, computed live from the replay context.

    jump_ratio is not tracked in replay and enters as neutral 0 (it is a
    known-dead feature; STRATEGY.md decision log 2026-07-14).
    """
    bbo = ctx.bbo
    total = bbo.bid_size + bbo.ask_size
    mid = (bbo.bid + bbo.ask) / 2.0 if bbo.bid is not None and bbo.ask is not None else None
    microprice = (
        (bbo.ask * bbo.bid_size + bbo.bid * bbo.ask_size) / total if total > 0 and mid is not None else mid
    )
    bids3 = ctx.book.top_levels("bid", 3)
    asks3 = ctx.book.top_levels("ask", 3)
    depth_bid3 = sum(size for _, size in bids3)
    depth_ask3 = sum(size for _, size in asks3)
    depth3 = depth_bid3 + depth_ask3
    return {
        "imb_top1": (bbo.bid_size - bbo.ask_size) / total if total > 0 else 0.0,
        "imb_top3": (depth_bid3 - depth_ask3) / depth3 if depth3 > 0 else 0.0,
        "microprice_minus_mid": (microprice - mid) if microprice is not None and mid is not None else 0.0,
        "flow_1s": ctx.flow_1s,
        "flow_5s": ctx.flow_5s,
        "bid_size_delta": ctx.bid_size_delta,
        "ask_size_delta": ctx.ask_size_delta,
        "ext_micro_lead_bps": ctx.ext_micro_lead_bps,
        "ext_imbalance": ctx.ext_imbalance,
        "spread": (bbo.ask - bbo.bid) if bbo.bid is not None and bbo.ask is not None else None,
        "gbm_fair_minus_mid": (ctx.fair_yes - mid) if mid is not None else 0.0,
        "trades_5s": float(ctx.trades_5s),
        # None + impute_missing=True standardizes to the training mean —
        # actually neutral, unlike a raw 0.0 which is (0-mean)/std biased.
        "jump_ratio": None,
    }


class ModelVSkew(QuoteAdjuster):
    """Fitted Model V consumer: center quotes on mid + predicted 5s move."""

    def __init__(self, models: list[LinearModel], *, weight: float = 1.0):
        self.models = models
        self.weight = weight
        self.name = "modelv"

    def adjust(self, ctx: QuoteContext) -> tuple[float, float, float]:
        model = _bucket_model(self.models, ctx.seconds_to_close)
        bbo = ctx.bbo
        if model is None or bbo.bid is None or bbo.ask is None:
            return ctx.fair_yes, 0.0, 0.0
        raw = context_raw_features(ctx)
        prediction = model.predict([raw.get(name) for name in model.features], impute_missing=True)
        if prediction is None:
            return ctx.fair_yes, 0.0, 0.0
        mid = (bbo.bid + bbo.ask) / 2.0
        return mid + self.weight * prediction, 0.0, 0.0


class ToxicityGate(QuoteAdjuster):
    """Model T consumer: per-side P(toxic) from side-mirrored features.

    The tax activates when predicted probability exceeds the model's own
    training base rate (stored in meta), not a fixed 0.5 — state-model base
    rates run 0.15-0.35 and a fixed threshold would rarely fire.
    """

    def __init__(self, models: list[LinearModel], *, scale_dollars: float = 0.02, base: Optional[float] = None):
        self.models = models
        self.scale_dollars = scale_dollars
        self.base = base
        self.name = "toxgate"

    def adjust(self, ctx: QuoteContext) -> tuple[float, float, float]:
        model = _bucket_model(self.models, ctx.seconds_to_close)
        if model is None:
            return ctx.fair_yes, 0.0, 0.0
        base = self.base if self.base is not None else float(model.meta.get("base_rate", 0.5))
        raw = context_raw_features(ctx)
        taxes = {}
        for side in ("bid", "ask"):
            mirrored = mirror_features(raw, side)
            probability = model.predict(
                [mirrored.get(name) for name in model.features], impute_missing=True
            )
            if probability is None:
                taxes[side] = 0.0
                continue
            excess = max(0.0, probability - base) / max(1e-9, 1.0 - base)
            taxes[side] = excess * self.scale_dollars
        return ctx.fair_yes, taxes["bid"], taxes["ask"]


def build_variants(
    spec: str, model_t_path: Optional[str], model_v_path: Optional[str] = None
) -> list[QuoteAdjuster]:
    variants: list[QuoteAdjuster] = []
    for name in spec.split(","):
        name = name.strip()
        if name == "baseline":
            variants.append(QuoteAdjuster())
        elif name.startswith("micro"):
            variants.append(MicropriceSkew(int(name[5:]) / 100.0))
        elif name == "toxgate":
            if not model_t_path or not Path(model_t_path).exists():
                raise SystemExit("toxgate variant requires --model-t pointing at a model_t json")
            variants.append(ToxicityGate(load_models(model_t_path)))
        elif name == "modelv":
            if not model_v_path or not Path(model_v_path).exists():
                raise SystemExit("modelv variant requires --model-v pointing at model_v.json")
            # Use the book-feature-set bucket models (ext-set models require
            # externals to be fresh; book models degrade more gracefully).
            models = [m for m in load_models(model_v_path) if m.meta.get("feature_set") == "book"]
            variants.append(ModelVSkew(models))
        else:
            raise SystemExit(f"unknown variant: {name}")
    return variants


@dataclass(frozen=True)
class WindowOutcome:
    ticker: str
    variant: str
    fills: int
    contracts: float
    markout_5s: Optional[float]
    fees: float
    settlement_pnl: Optional[float]

    @property
    def markout_dollars(self) -> float:
        """Total (not per-contract) 5s markout; 0 when the variant did not trade.

        Pairing per-contract markouts drops zero-fill windows from the diff —
        which would credit a gate for its best behavior (not trading toxic
        windows) by excluding exactly those windows. Dollars keep them at 0.
        """
        if self.markout_5s is None:
            return 0.0
        return self.markout_5s * self.contracts


def run_ab(
    recording_paths: list[str | Path],
    variants: list[QuoteAdjuster],
    *,
    latency_ms: int,
    maker_fee_multiplier: float,
) -> list[WindowOutcome]:
    outcomes: list[WindowOutcome] = []
    for market, paths in group_recordings_by_market(recording_paths):
        if not _window_is_final(market, paths):
            continue
        window = _load_window(market, paths)
        rows = list(window.rows)
        for variant in variants:
            result: ReplayResult = replay_rows(
                market,
                rows,
                latency_ms=latency_ms,
                maker_fee_multiplier=maker_fee_multiplier,
                adjuster=variant if variant.name != "baseline" else None,
            )
            outcomes.append(
                WindowOutcome(
                    ticker=market.ticker,
                    variant=variant.name,
                    fills=len(result.fills),
                    contracts=result.contracts_filled,
                    markout_5s=result.markout_5s,
                    fees=result.fees_paid,
                    settlement_pnl=result.settlement_pnl,
                )
            )
            print(
                f"[ab] {market.ticker} {variant.name}: fills={len(result.fills)} "
                f"contracts={result.contracts_filled:.1f} markout5s={result.markout_5s} "
                f"settle_pnl={result.settlement_pnl}",
                flush=True,
            )
        del window, rows
    return outcomes


def summarize(outcomes: list[WindowOutcome]) -> None:
    by_variant: dict[str, dict[str, WindowOutcome]] = {}
    for outcome in outcomes:
        by_variant.setdefault(outcome.variant, {})[outcome.ticker] = outcome
    if "baseline" not in by_variant:
        return
    baseline = by_variant["baseline"]
    print("\n== A/B summary (per-window, block-bootstrap 95% CI on diff vs baseline) ==")
    print(
        f"{'variant':>10s} {'windows':>7s} {'fills/w':>8s} {'markout5s':>10s} {'settle_pnl/w':>12s} "
        f"{'d_markout$ (CI)':>28s} {'d_settle (CI)':>28s}"
    )
    for variant, windows in by_variant.items():
        shared = [ticker for ticker in windows if ticker in baseline]
        fills = sum(windows[t].fills for t in shared) / max(1, len(shared))
        markouts = [windows[t].markout_5s for t in shared if windows[t].markout_5s is not None]
        settles = [windows[t].settlement_pnl for t in shared if windows[t].settlement_pnl is not None]
        mean_markout = sum(markouts) / len(markouts) if markouts else float("nan")
        mean_settle = sum(settles) / len(settles) if settles else float("nan")
        if variant == "baseline":
            print(f"{variant:>10s} {len(shared):7d} {fills:8.1f} {mean_markout:10.4f} {mean_settle:12.3f} {'—':>28s} {'—':>28s}")
            continue
        # Markout diffs in DOLLARS so zero-fill windows count as 0 instead of
        # being dropped (a gate must not be graded only on windows it traded).
        d_markout = [windows[t].markout_dollars - baseline[t].markout_dollars for t in shared]
        d_settle = [
            windows[t].settlement_pnl - baseline[t].settlement_pnl
            for t in shared
            if windows[t].settlement_pnl is not None and baseline[t].settlement_pnl is not None
        ]
        m_mean, m_low, m_high = block_bootstrap_ci(d_markout)
        s_mean, s_low, s_high = block_bootstrap_ci(d_settle)
        print(
            f"{variant:>10s} {len(shared):7d} {fills:8.1f} {mean_markout:10.4f} {mean_settle:12.3f} "
            f"{f'{m_mean:+.3f} [{m_low:+.3f},{m_high:+.3f}]':>28s} "
            f"{f'{s_mean:+.3f} [{s_low:+.3f},{s_high:+.3f}]':>28s}"
        )


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="A/B quoting variants over recorded windows.")
    parser.add_argument("recordings", nargs="+")
    parser.add_argument("--variants", default="baseline,micro25,micro50")
    parser.add_argument("--model-t", default="data/models/model_t_state.json")
    parser.add_argument("--model-v", default="data/models/model_v.json")
    parser.add_argument("--latency-ms", type=int, default=150)
    parser.add_argument("--maker-fee-multiplier", type=float, default=0.0)
    args = parser.parse_args(argv)
    variants = build_variants(args.variants, args.model_t, args.model_v)
    outcomes = run_ab(
        args.recordings,
        variants,
        latency_ms=args.latency_ms,
        maker_fee_multiplier=args.maker_fee_multiplier,
    )
    summarize(outcomes)


if __name__ == "__main__":
    main()
