# 2026-07-14 — Feature pipeline smoke test + first descriptive peek

**Status:** descriptive only. 22 windows (~5.5 hours, all future *train*-split data).
Nothing here is promotion evidence (need ≥300 windows to screen; STRATEGY.md §6).

**Hypothesis:** none (pipeline validation + unstructured peek).

**Spec:** `kalshi-mm-features` over all finalized server recordings → 22 parquet windows,
64,365 rows @250ms cadence. Toy per-window Pearson IC of each feature vs `fwd_mid_delta_5s`,
averaged across windows.

## Results

Pipeline: memory-bounded (one window at a time), idempotent, in-progress windows skipped.
22/22 finalized windows compacted; ~0.5MB parquet per window.

Data quality: external-L2-derived columns null in the 8 windows recorded before the
external depth feeds deployed (~08:00 UTC); complete afterward. `fwd_mid_delta_30s`
null 3.3% (window-end truncation, expected).

Toy ICs vs 5s forward mid (mean across windows, ~sd/√n in parens):

| feature | mean IC | windows |
|---|---|---|
| microprice_minus_mid | **+0.173** (0.017) | 22 |
| imb_top1 | **+0.152** (0.014) | 22 |
| imb_top3 | **+0.148** (0.010) | 22 |
| ext_imbalance (external book) | +0.099 (0.017) | 14 |
| flow_5s | −0.026 | 22 |
| gbm_fair_minus_mid | +0.018 | 22 |
| ext_basis_bps | −0.004 | 22 |

Also: median |gbm_fair − mid| ≈ 2¢ per window (max 8¢ in one window), consistent with
the audit's fair-mid MAE.

## Read (preliminary, to be re-tested at ≥300 windows)

- **Kalshi book-shape features (microprice, imbalance) are the strongest early
  candidates** — positive in essentially every window with small cross-window dispersion.
  Caveats: partly mechanical (imbalance predicts the next tick of a discrete mid semi-
  tautologically), and IC vs mid-move ≠ tradeable edge after queue, latency, and fees.
  Model T/V (STRATEGY.md §5) is the right consumer, not naive signal-chasing.
- **GBM fair-minus-mid does not predict 5s mid moves** (~0 IC) — consistent with its
  STRATEGY.md role as calibration anchor/close-layer, not a short-horizon signal.
- **Raw external spot basis showed ~0 IC at 5s** — the naive basis tilt may be too
  slow/stale as constructed (2s freshness, median-of-spots). The external *book*
  imbalance looks more promising than the spot basis. Re-examine with faster
  external features and at 1s horizon.

## Verdict

Pipeline promoted to production use. No feature verdicts — window count insufficient.
Next: re-run this peek at ~300 windows (≈2026-07-17); begin Model T label generation.
