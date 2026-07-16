# 2026-07-16 — First honest A/B: quoting variants vs baseline (124 windows)

**Status:** directional (25 validation windows; promotion needs ≥1,000). First results
under fully corrected simulation: post-only activation, safety cancels, bounded
markouts, strictly-past joins, verified zero maker fees, 150ms latency, 100ms
decision cadence. Train 74 / val 25 / holdout 25 (locked, untouched).

## Fit (val, out-of-sample)

- Model V IC: +0.10 to +0.20 across buckets, all CIs exclude zero; book+ext > book.
- Model T AUC: 0.63–0.69 (state-trained best near close). Yesterday's 0.83 was
  small-sample optimism; this is the honest level.
- P1: mid better-calibrated near close, GBM marginally better mid-window (stable).

## A/B (25 val windows, block-bootstrap 95% CI vs baseline)

| variant | fills/w | markout/contract | settle P&L/w | Δmarkout$ /w (CI) | Δsettle /w (CI) |
|---|---|---|---|---|---|
| baseline | 167.9 | −0.13¢ | −$0.11 | — | — |
| **micro25** | 166.9 | −0.05¢ | −$0.01 | **+$0.49 [+0.23, +0.83]** | +$0.10 [−0.29, +0.50] |
| toxgate | 149.8 | −0.09¢ | +$0.16 | +$0.27 [−0.61, +1.17] | +$0.27 [−0.44, +0.97] |
| modelv (w=1.0) | 92.5 | −0.19¢ | −$1.21 | −$0.53 [−2.08, +0.78] | −$1.10 [−4.79, +2.66] |

## Read

1. **micro25 is the first variant whose improvement CI excludes zero**: a 25%
   microprice lean cuts adverse selection by ~$0.49/window at identical fill
   volume. Baseline itself is near breakeven under true fees — the gap to
   profitability is small and micro25 appears to close most of it on markout.
2. **toxgate points positive on BOTH markout and settlement but is underpowered**
   (wide CI): consistent with T's moderate AUC and an untuned 2¢ scale/threshold.
3. **modelv at full weight (1.0) is harmful-to-noise**: halves fills, negative tilt.
   Classic result — the shrunken simple signal (micro25) beats the full-weight
   fitted model. Retest at w≈0.25 and with T-style shrinkage.
4. Settlement-P&L CIs all include zero — inventory luck dominates at 25 windows,
   as expected; markout$ is the discriminating metric at this sample size.

## Next (Friday screening at ≥300 windows)

- Variants: baseline, micro25, **micro25+toxgate combined**, toxgate with tuned
  scale ∈ {1¢, 2¢, 4¢}, modelv at w ∈ {0.25, 0.5}.
- Same splits; holdout stays locked until a promotion decision.

## Addendum (same day): combo25 = micro25 + toxgate composed, same 25 val windows

| variant | fills/w | markout/contract | settle P&L/w | Δmarkout$ (CI) | Δsettle (CI) |
|---|---|---|---|---|---|
| combo25 | 148.6 | −0.08¢ | **+$0.19** | +$0.36 [−0.64, +1.39] | +$0.30 [−0.37, +0.99] |

Read: best ABSOLUTE settlement P&L of any variant (+$0.19/w vs baseline −$0.11)
and positive point estimates on both diffs — but neither CI excludes zero, and
the effects are clearly SUB-additive (+0.36 vs the singles' +0.49 and +0.27):
the microprice lean and the gate partially defend against the same fills, and
the gate's abstention windows add variance. micro25 alone remains the only
statistically significant variant. Friday: micro25 vs combo at tuned gate
scales (1¢/2¢/4¢) on ≥300 windows decides the graduation candidate.
