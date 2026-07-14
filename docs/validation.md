# Kalshi validation protocol

The first production milestone is data quality, not order placement.

Collect at least 1,000 complete `KXBTC15M` windows. Keep all events from a window in the same
walk-forward split. Use the first 60% for model construction, the next 20% for parameter selection,
and the final 20% once for promotion.

For each candidate model, report:

- Brier score, log loss, and calibration by probability decile against settlement.
- Difference from the contemporaneous Kalshi midpoint.
- Correlation and hit rate between fair-minus-mid and future midpoint movement at 100ms, 500ms,
  1s, 5s, and 30s.
- Conservative fill rate after queue ahead, not touch rate.
- Spread capture, post-fill markout, fees, inventory PnL, and settlement PnL separately.
- Results by time-to-close, spread, volatility, jump ratio, BRTI basis, and inventory regime.

Compare the market midpoint, settlement-aware independent volatility, cross-venue basis features,
and toxicity-gated quoting at identical size and latency. Include measured feed-to-decision and
request-to-matching-engine latency in replay.

Promote a strategy only when final-holdout net maker markout and settlement PnL are positive after
fees and the block-bootstrap confidence interval excludes zero. Operational promotion also requires
100 consecutive paper windows without an unexplained position discrepancy, post-only violation,
unrecovered sequence gap, or stale order left live.
