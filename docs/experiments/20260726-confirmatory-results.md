# 2026-07-26 — 1,000-window confirmatory results (200 validation windows)

Pre-registration: docs/experiments/20260722-confirmatory-preregistration.md (+ amendment).
Fit: train=600 val=200 holdout=200 (LOCKED). Baseline settle -$0.68/window.

| variant | fills/w | Δmarkout$ (CI) | Δsettle$ (CI) | dual-CI? |
|---|---|---|---|---|
| toxgate2 | 134.0 | +0.386 [+0.180,+0.603] | +0.163 [-0.139,+0.459] | markout only |
| toxgate4 | 123.8 | +0.643 [+0.394,+0.928] | +0.439 [+0.011,+0.858] | **YES** |
| **toxgate6** | 115.0 | +0.788 [+0.516,+1.069] | +0.710 [+0.180,+1.233] | **YES (best)** |
| toxgate8 | 108.7 | +0.868 [+0.570,+1.187] | +0.507 [-0.137,+1.147] | markout only |
| combo25x4 | 123.0 | +0.646 [+0.390,+0.929] | +0.490 [+0.006,+0.946] | YES |
| combo25x6 | 114.3 | +0.742 [+0.458,+1.048] | +0.497 [-0.035,+1.020] | markout only |

## Findings (pre-registered rules)

1. FIRST result meeting the promotion bar (BOTH markout$ AND settle$ CIs exclude
   zero): toxgate4, toxgate6, combo25x4 qualify. By the pre-registered tiebreak
   (best settle$ point among qualifiers), the CANDIDATE is **toxgate6**.
2. Clean monotonic optimum: aggression x2->x4->x6 improves both metrics; x8
   over-abstains (fills too low, settle CI reopens). Peak at x6.
3. Attribution test PASSED: gate-only >= combo at every scale -> the microprice
   skew adds nothing -> MICRO DROPPED. Strategy simplifies to gate-only.
4. toxgate6 is the ONLY variant with positive ABSOLUTE simulated settlement P&L
   (+$0.026/window vs baseline -$0.68). Others are merely less negative.

## Discipline / caveats (unchanged)

- SIMULATED: trustworthy RELATIVE ranking; the absolute +$0.026/window is thin
  and could be erased by the sim-to-real gap (market impact, true queue).
- +$0.026/w x ~96 windows/day ~= $2.5/day on baseline sizing — small absolute.
- This EARNS the one permitted holdout touch, but only AFTER the pre-registered
  latency stress-test (75/300ms) on validation, and as a DELIBERATE decision.

## Next (pre-registered)

1. Latency stress-test toxgate6 at 75ms and 300ms on the 200 val windows; a
   settle-sign flip disqualifies it.
2. If it survives: ONE holdout run of toxgate6 (the single permitted touch).
   Holdout settle CI excluding zero -> paper-trading candidate.
3. Then live pilot at throwaway size (measures the sim-to-real gap).
