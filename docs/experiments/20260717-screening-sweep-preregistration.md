# 2026-07-17 — Screening sweep pre-registration (written BEFORE the run)

Runs automatically when the feature store reaches 300 windows (~11:30 UTC).
Fit refreshes on the train split first; sweep runs on the chronological
validation slice (~60 windows). Holdout stays locked.

## Variants (9 comparisons vs baseline)

baseline, micro15, micro25, micro40 (weight response curve),
combo25x1, combo25x2, combo25x4 (gate-scale response on top of micro25),
modelv25 (fitted V, shrunk), wide2 (2c min edge probe),
close30 (quote into T-90..30s — EXPLORATORY: separate hypothesis, distinct
risk profile, excluded from graduation regardless of result).

## Pre-registered decision rule

- PRIMARY metric: per-window markout$ diff vs baseline (powered at n~60).
- SANITY: settlement P&L diff sign; a candidate with negative settle sign is
  not selected even with positive markout.
- SELECTION: top-2 non-exploratory variants advance to the 1,000-window
  confirmatory run. NO promotion decision from this sweep (9 comparisons at
  n~60 cannot clear multiple-look inflation for settle CIs).
- WINNER STRESS: the top variant reruns at 75ms and 300ms latency; a sign flip
  disqualifies it.
- STABILITY: results reported per collection day; a variant winning on only
  one day is flagged regime-dependent.

## Known caveats accepted in advance

25 of the val windows were used in the two prior looks (2026-07-16); the
1,000-window run and holdout remain uncontaminated. Gate uses the state-
trained Model T; fills-trained comparison deferred.
