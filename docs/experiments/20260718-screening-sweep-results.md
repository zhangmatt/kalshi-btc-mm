# 2026-07-18 — Screening sweep results (60 validation windows)

Pre-registered protocol: docs/experiments/20260717-screening-sweep-preregistration.md.
Primary metric = per-window markout$ diff vs baseline; settle sign = veto.

baseline: 159 fills/w, markout -0.21c, settle -$0.55/w (note: more negative than
the 25-window look's -$0.11 — more regimes = more adverse selection; the family
is further from breakeven than the first look implied).

| variant | fills/w | Δmarkout$ (CI) | Δsettle$ (CI) |
|---|---|---|---|
| micro25 | 158 | -0.05 [-0.29,+0.20] | -0.05 [-0.35,+0.26] |
| micro40 | 157 | +0.16 [-0.19,+0.49] | -0.06 [-0.45,+0.35] |
| combo25x1 | 145 | +0.22 [-0.06,+0.50] | +0.13 [-0.30,+0.53] |
| combo25x2 | 136 | +0.22 [-0.19,+0.63] | +0.27 [-0.36,+0.92] |
| **combo25x4** | 125 | +0.34 [-0.20,+0.86] | **+1.03 [+0.14,+1.96]** |
| modelv25 | 68 | -1.50 [-2.49,-0.63] | -1.83 [-5.16,+1.40] |
| wide2 | 89 | +0.07 [-0.44,+0.58] | +0.43 [-0.46,+1.33] |
| close30 | 171 | -0.32 [-1.06,+0.31] | +0.16 [-1.14,+1.64] |

## Findings

1. **micro25's earlier significance DID NOT REPLICATE.** On the 25-window look it
   was markout +$0.49 CI-excludes-zero; on 60 windows it is -$0.05, CI spans zero.
   Textbook small-sample false positive (or a benign early regime). The pre-
   registered larger look caught it — the process worked as designed.
2. **No variant clears the PRIMARY metric (markout$ CI excluding zero).** By the
   strict rule, nothing is confirmed.
3. **One coherent secondary signal: the toxicity gate, run AGGRESSIVELY.** combo25
   is monotonic in gate scale — x1->x2->x4: fills 145->136->125, settle -0.42->
   -0.28->+0.48, and combo25x4's SETTLE CI excludes zero (+$1.03 [+0.14,+1.96]).
   The gate works; it must be aggressive to matter, and it pays by NOT trading
   toxic windows (fills down 21%). Markout CI still includes zero, so treat as
   promising-not-proven.
4. modelv25 confirmed significantly negative — fitted V as a fair-shift is dead;
   its value is as a signal/feature, not a quote-center.
5. close30 (quote into T-90..30s) and wide2 (2c edge): no benefit, not significant.

## Selection (per pre-registration)

Top-2 to the 1,000-window confirmatory run: **combo25x4** (best settle, sig settle
CI, best markout point) and **combo25x2** (same mechanism, more fills = more power).
Also carry a gate-scale extension (x6, x8) to find where over-abstention starts.
micro-alone family DROPPED — did not replicate. Holdout still locked.
