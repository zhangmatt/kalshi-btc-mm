# 2026-07-22 — 1,000-window confirmatory run pre-registration (BEFORE the run)

Fires automatically at 1,000 feature windows (~2 days from 795). Refits on the
train split, then runs the selected variants on the chronological VALIDATION
slice (~200 windows). HOLDOUT STAYS LOCKED — see promotion gate below.

## Variants (selection carried from the 2026-07-18 screening)

baseline, combo25x2, combo25x4, combo25x6, combo25x8.
(micro-alone family dropped — did not replicate. modelv dropped — sig negative.
x6/x8 extend the gate-scale ladder to locate where over-abstention costs more
than the toxic fills it avoids.)

## Pre-registered decision rule

- PRIMARY: per-window markout$ diff vs baseline, block-bootstrap 95% CI.
- SANITY/veto: settlement P&L diff sign and CI.
- A variant is a PROMOTION CANDIDATE only if, on the ~200 validation windows,
  BOTH its markout$ CI AND its settle$ CI exclude zero. (Screening showed settle
  significant but markout not; the larger sample must resolve markout too.)
- If >=1 candidate qualifies: pick the single best by settle$ point estimate.
  That one variant — and only that one — is then run ONCE on the locked holdout
  (~200 windows). This is the single permitted holdout touch. Promotion to paper
  requires the holdout settle$ CI to also exclude zero.
- If NO candidate qualifies: the maker thesis is not confirmed at this sample.
  Do NOT touch the holdout. Report negative; next step is feature/model work
  (event-time fill features, richer T), not another parameter sweep.
- WINNER STRESS (before any holdout touch): rerun the candidate at 75ms and
  300ms latency; a settle-sign flip disqualifies it.
- STABILITY: report per collection day; flag single-day wins.

## Accepted caveats

Val windows overlap earlier looks (train/holdout do not). Single BTC market,
~10 days of regimes. State-trained Model T; fills-trained deferred. Simulated
queue — live pilot remains mandatory after any promotion regardless of holdout.
