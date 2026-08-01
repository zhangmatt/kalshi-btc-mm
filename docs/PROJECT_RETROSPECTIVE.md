# Kalshi BTC 15-Minute Market-Making System — Project Retrospective

*A quantitative research and infrastructure project: building, validating, and
honestly evaluating an automated market-making strategy for Kalshi's KXBTC15M
Bitcoin up/down 15-minute binary options.*

**Duration:** ~3 weeks (July 2026) · **Status:** Research complete; strategy not
deployed to real capital (validated as unprofitable in the tested regime — see
Findings). **Zero capital lost.**

---

## 1. Executive summary

I built an end-to-end automated market-making system for a Kalshi Bitcoin binary
options market: continuous L2 market-data collection on cloud infrastructure, a
settlement-aware fair-value model, a research pipeline with fitted predictive
models, a queue-aware backtest, a pre-registered statistical validation protocol,
and a live paper-trading deployment.

The headline outcome is a **rigorous negative result**: the strategy showed a
statistically significant edge in one market regime but was unprofitable in
another, and the disciplined validation process caught this **before any real
money was risked**. The lasting deliverables are a reusable research/collection
platform, a validated (if modest) relative edge from toxicity-gated quoting, and
a precise diagnosis of the failure mode (inventory risk carried to settlement).

---

## 2. The problem

**Market:** Kalshi KXBTC15M — "Will BTC be higher in the next 15 minutes?" Binary
contracts pay \$1 (Yes) or \$0 (No), trade between \$0–1, and settle on whether the
final-60-second average of CF Benchmarks' BRTI index exceeds the opening
60-second average. A new market opens every 15 minutes (~96/day).

**Strategy class:** passive market making — quote both sides around a fair-value
estimate and earn the spread from uninformed flow, while managing inventory and
avoiding "toxic" fills from informed traders. The P&L identity that governs the
whole effort:

> P&L = uninformed flow captured − informed flow eaten − fees ± inventory/settlement variance

**Motivation:** a documented solo operator reported ~\$50K profit from ~\$25K on a
similar Kalshi crypto market, describing a system that "blends order book and
external features" for fair value. This project set out to independently
build, validate, and — critically — *honestly test* whether that class of
strategy is real.

---

## 3. System architecture

### Data collection (production, unattended on AWS EC2)
- **Kalshi WebSocket feed:** sequenced L2 order-book snapshots + deltas, public
  trades (with aggressor side), ticker/BBO, CF Benchmarks BRTI index (including
  the final-minute settlement-averaging fields), and market-lifecycle events.
- **External reference venues:** spot and full L2 depth (10 levels @ 10 Hz) from
  Binance.US, Coinbase, and Kraken, with per-venue sequence/checksum validation
  and reconnect-safe book reconstruction.
- **Integrity:** exchange + local receive timestamps on every event; orderbook
  sequence-gap detection with snapshot recovery; per-channel sequence auditing.
- **Reliability:** systemd services with health timers that restart on stale
  data; graceful market rollover; ~33–50 MB memory footprint; ~586 MB/day
  compressed. Collected **1,750+ complete market windows** (~3 weeks).
- **Durable archival:** SHA-256-verified uploads to a private, versioned,
  encrypted S3 bucket via an IAM instance role; fail-closed local retention
  (never deletes a file without a verified remote copy).

### Fair-value & volatility models
- **Settlement-aware GBM model:** prices P(final-60s-average ≥ strike) by
  moment-matching the discrete arithmetic average of BRTI fixings under geometric
  Brownian motion, conditioning exactly on Kalshi's published partial
  final-minute average. **Validated against a 200,000-path Monte Carlo simulation
  — agreement within ±1.6 standard errors across all tested scenarios.** Captures
  a material effect (up to ~8¢ vs a naive terminal-binary model near expiry).
- **Multi-horizon volatility forecast** from 1-second BRTI observations (EWMA +
  realized-vol horizons), with jump-ratio estimation.

### Research pipeline
- **Columnar feature store:** each closed window compacted once into Parquet
  (~30 features: book imbalance/microprice/flow, external-venue basis, state
  variables, forward-mid targets), append-only, maintained by systemd timers.
  ~580,000 feature rows.
- **Model V (value):** ridge regression predicting short-horizon (5s) mid-price
  movement from book + external features, per time-to-close bucket.
- **Model T (toxicity):** L2-regularized logistic regression predicting fill
  toxicity (adverse post-fill markout), using **side-mirrored features** (so
  directional signals don't cancel across bid/ask fills) and a **counterfactual
  state-training** scheme that turns every book snapshot into two training
  samples (~20× the fill-only data).
- All models: standardized, per-bucket, serialized to JSON; NumPy implementations
  (closed-form ridge, Newton-Raphson logistic) — no sklearn dependency.

### Backtest & execution
- **Queue-aware replay:** conservative fill model (fills only after observed
  aggressor volume clears displayed queue-ahead), a **decision-to-matching-engine
  latency model** (post-only rejection of crossed orders, safety-cancels during
  invalid-book/stale-data periods), and honest fee accounting.
- **Live runner:** real-time quoting loop with inventory skew, staleness/divergence
  gates, private-fill-channel reconciliation, and a paper mode with the same
  queue-aware fill simulation. Deployed to **paper trading** (zero real orders).
- **Verified fidelity:** paper-runner P&L matched replay P&L on identical windows
  (−\$2.54 vs −\$2.82), confirming the two simulators agree.

---

## 4. Research methodology (the part I'm most proud of)

The project's rigor, not its P&L, is its strongest feature. Every result was
produced under a **pre-registered protocol** designed to defeat the ways
backtests lie:

- **Unit of independence = the market window**, not the tick (millions of
  correlated 250 ms rows are not millions of observations).
- **Chronological 60/20/20 train/validation/holdout split**, ordered by data
  timestamp (a subtle bug where ticker strings sorted `AUG < JUL` was caught and
  fixed before it could scramble the split).
- **The 20% holdout was never touched** during research — enforced in code.
- **Block-bootstrap confidence intervals over windows** for every comparison.
- **Pre-registration:** each screening/confirmatory run's variants and decision
  rules were written and committed *before* the run, with a permanent experiment
  log (successes and failures both recorded) to prevent post-hoc rationalization.
- **Multiple-looks discipline:** deliberately refused to fine-tune parameters on
  the validation set (a coarse robust optimum generalizes better than a fitted
  one).

### Adversarial code auditing
The system was subjected to **three independent multi-agent code audits** (fanned
out across research, simulation, and infrastructure subsystems), each finding
verified by hand. These caught **14+ result-invalidating defects** before they
could corrupt conclusions, including:
- a fee model defaulting to the wrong value (would have flipped P&L signs),
- simulation optimism (granting fills a live system couldn't get),
- train/serve feature leakage in the label pipeline,
- the timestamp-sort split bug above,
- health-monitoring holes that could mask a dead collector.

This "assume the backtest is lying until proven otherwise" posture is, in my
view, the single most important quant skill the project demonstrates.

---

## 5. Key results

**The confirmatory run (1,000 windows, 200-window validation):** a
toxicity-gated variant ("toxgate6" — quote around the settlement-aware fair, and
skip the ~23% of fills the order book flags as toxic) passed a **pre-registered
dual-CI significance bar**: Δmarkout +\$0.79/window [95% CI +0.52, +1.07] *and*
Δsettlement +\$0.71/window [+0.18, +1.23] vs baseline, with positive absolute
simulated P&L. The toxicity gate was the only lever, across many tested
(microprice skew, fitted value model, external-led fair value) that produced a
robust *relative* improvement.

**The honest negative finding:** when deployed to paper trading and re-tested on
a later, harsher market regime, the strategy **lost money** (−\$2.5/window), and a
diagnostic confirmed this was **regime, not a bug** (paper and replay agreed on
identical data). The confirmatory edge was regime-dependent; the toxicity gate
reliably adds ~\$1/window but could not overcome a base regime where making this
market is broadly unprofitable.

**Root-cause diagnosis:** an external-spot-led fair value (the "blend external
features" idea) was tested and *improved per-fill markout* but *worsened
settlement P&L* — because it accumulated more directional inventory carried to
settlement. This isolated the true failure mode: **inventory risk carried into
the binary settlement**, not stale pricing per se.

---

## 6. Honest conclusions

- A statistically significant *relative* edge (toxicity-gated quoting) exists and
  is robust across regimes.
- *Absolute* profitability is regime-dependent and was negative in the tested
  live period; the strategy is **not deployable to real capital as-is**.
- The disciplined validation ladder (backtest → holdout → paper → live pilot)
  did its job: it surfaced the losing regime at a cost of **\$0**, exactly where a
  less rigorous process would have gone live and lost real money.
- Remaining untested lever with the strongest evidence: **inventory flattening /
  hard-skew before settlement.**

---

## 7. Skills & technologies demonstrated

**Quantitative research:** market microstructure (order-flow toxicity, adverse
selection, microprice, queue dynamics); derivatives pricing (settlement-aware
GBM, arithmetic-average Asian-style payoff, Monte Carlo validation); statistical
methodology (pre-registered walk-forward validation, block bootstrap,
overfitting avoidance, honest hypothesis testing).

**Software & data engineering:** Python (threading/concurrency, NumPy, PyArrow);
real-time WebSocket ingestion and order-book reconstruction; columnar data
pipelines; memory-bounded processing (rewrote a fit that OOM'd at 3.3 GB into a
2.1 GB vectorized columnar path with no loss of resolution); 97-test suite.

**Cloud & production infrastructure:** AWS EC2, systemd service/timer
orchestration with health checks and auto-recovery, S3 durable archival with
integrity verification and IAM instance roles, RSA-PSS API request signing,
fail-closed data-retention design, unattended multi-day operation.

**Engineering judgment:** adversarial multi-agent code review; a decision log and
experiment log maintained throughout; and — hardest of all — the discipline to
report a negative result honestly rather than talk a backtest into looking
profitable.

**Scale:** ~4,000 lines of production Python, 97 automated tests, 1,750+ market
windows / ~580K feature rows collected and analyzed, three independent code
audits, dozens of pre-registered experiments.
