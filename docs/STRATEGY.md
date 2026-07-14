# KXBTC15M Market-Making Strategy Plan

**Status:** Active reference document. Maintained alongside research, development, and trading.
**Last revised:** 2026-07-14 (v1.0)
**Companion documents:** `docs/validation.md` (statistical protocol), `README.md` (operations).

How to use this document: every research experiment, code change to strategy logic, and
trading decision should be traceable to a section here. When reality contradicts this plan,
update the plan and record why in the Decision Log (§12). Claims are labeled **[FACT]**
(measured/verified), **[EVIDENCE]** (external, credible, unverified by us), or
**[HYPOTHESIS]** (our belief, to be tested).

---

## 1. Strategy Statement

**Queue-disciplined passive market making on Kalshi KXBTC15M, driven by two fitted
models — short-horizon value and fill toxicity — built from Kalshi L2 order-book features
blended with external crypto-venue features, with an exact settlement-mathematics layer
in the final two minutes.**

Maker P&L identity that governs everything:

```
P&L = uninformed flow captured − informed flow eaten − fees ± inventory/settlement variance
```

The durable edge is not spread capture (commoditized) but **separating uninformed from
informed flow faster and more accurately than competing makers**, and refusing the
informed side.

## 2. Market and Evidence Base

- **[FACT]** KXBTC15M: BTC up/down binary, 96 windows/day, settles on the final-60-second
  BRTI average vs. the opening strike average. Observed: ~$400K dollar volume and ~15K
  trades per active window; spreads 1–2¢ mid-market, 0.1¢ in the tapered tails.
- **[FACT]** BRTI publishes at 1Hz and is computed from constituent exchange order books;
  external venues (Coinbase, Binance, Kraken) lead it mechanically.
- **[FACT]** 15-minute settlement bounds inventory risk; positions cannot be carried.
- **[EVIDENCE]** Existence proof: a solo operator (Python/EC2/TimescaleDB stack) reports
  ~$50K profit from ~$25K capital over a few months making markets on a Kalshi market of
  this type, with $40K in the final month "as the pricing and quoting logic came together."
  His fair value blended order-book and external features, fitted on a large L2 corpus.
  User has verified the claim's authenticity. The *number* remains unaudited.
- **[EVIDENCE]** DRW is hiring for a dedicated prediction-markets desk explicitly targeting
  market making with dynamic skew, order-flow/book-imbalance "sniping," and sub-second
  event momentum on Kalshi/Polymarket. Institutional competition is arriving; the current
  regime is soft but decaying. Speed-to-validated-strategy matters.
- **[FACT]** Our single replayed window (defective data, fees+latency on) lost $6.59.
  Nothing measured so far demonstrates that *our* current quoting logic is profitable.
  The existence proof says the market can pay; it does not say our model does.

## 3. Edge Layers, Ranked by Expected ROI

1. **Toxicity avoidance (Model T).** Most maker losses concentrate in a small minority of
   fills taken just before adverse moves. Predicting fill toxicity and pulling/widening/
   shading beats out-predicting the market the rest of the time. **[HYPOTHESIS]** This is
   the highest-ROI research target and likely what made the incumbent's P&L inflect.
2. **Short-horizon value (Model V).** A fitted estimate of the mid 1–30s ahead, from book +
   external features, to center quotes. Better centering improves capture AND reduces
   adverse selection simultaneously.
3. **Settlement mathematics near close.** **[FACT]** The final-60s-averaging payoff makes
   naive terminal-binary pricing wrong by up to ~8¢ at T−61s (Monte-Carlo-verified, our
   `fair_value.py` model matches MC within noise). A correct-averaging maker gets paid by
   anyone pricing the terminal payoff. This is our most differentiated, least crowded layer.
4. **Queue discipline.** In a 1–2¢-spread market, queue position is alpha. Keep-vs-cancel
   policy and requote thresholds are tuned research parameters, not plumbing.
   **[FACT]** Naive count-matching caused a cancel/repost loop (126K orders/window in
   replay) before the v0.2.1 fix; queue-priority errors are P&L-relevant at this scale.
5. **Inventory/skew policy.** Reservation-price skew exists (2¢ at max inventory); its
   magnitude and shape are untuned. Lowest priority — settlement flattens inventory every
   15 minutes.

## 4. Explicit Non-Goals (and reconsideration triggers)

- **No taking/sniping.** 7¢-rate taker fees kill marginal taker edges; we would race
  DRW-class infrastructure. Reconsider only if a signal shows taker-side edge > 2× the
  round-trip fee at our measured latency.
- **No cross-platform Polymarket relative value.** Different oracle and payoff definitions
  ⇒ settlement-basis risk, plus a second collection stack. Reconsider after live maker
  profitability, and only with synchronized Polymarket collection running ≥1 month.
- **No breadth/census expansion yet.** The incumbent's post indicates a single-market
  strategy ("quotes both sides of *a* market"; "looking to *expand* to more markets").
  Breadth is optionality, not the replication path. Reconsider at Phase L2 (scale) or if
  KXBTC15M edge decays below promotion thresholds.
- **No ML beyond ridge/logistic per bucket until window counts justify it.** With
  window-level validation, weeks of data support only low-variance models. Reconsider at
  ≥2,000 windows.

## 5. Model Specifications

### Shared feature set (computed per decision point from recordings)

*Kalshi book:* multi-level (top-3/top-5) imbalance; microprice − mid; signed trade flow
over 1s/5s/30s; trade intensity; depth depletion/replenishment rate at BBO; spread state;
queue-ahead at our candidate price.
*External:* composite external spot vs BRTI basis; external microprice-led spot estimate;
external momentum 1s/5s; cross-venue dispersion; external book imbalance (from our 10-level
L2 feeds).
*State:* seconds-to-close; moneyness z = ln(spot/strike)/(σ√T); vol forecast σ; jump ratio;
partial final-minute average state (inside T−60s).

### Model T — fill toxicity (build FIRST)

- **Target:** sign/magnitude of 1s and 5s markout on fills (simulated via queue-aware
  replay initially; real fills once live).
- **Form:** logistic (toxic/benign) + ridge (markout magnitude), fit per time-to-close
  bucket (≥T−120s, T−120s..T−60s excluded → close cutoff governs).
- **Use:** per-side toxicity dollars fed into the existing `bid_toxicity_dollars` /
  `ask_toxicity_dollars` strategy hooks (already plumbed, currently unused).
- **Promotion test:** replay A/B (gated vs ungated), fees on, 150ms latency: net
  markout-given-fill improves AND fill count does not drop below 60% of ungated, across
  the validation split, block-bootstrap CI excluding zero.

### Model V — short-horizon value

- **Target:** Δmid at 1s/5s/30s (primary); settlement outcome (calibration check).
- **Form:** ridge predicting a *correction* to an anchor: `fair = anchor + f(features)`,
  anchor = market mid (default) or GBM fair (near close). Per time-to-close bucket.
- **Promotion test:** must beat BOTH the mid baseline and the analytic GBM model on
  out-of-sample markout-given-fill in replay — not merely on IC. A value model that
  predicts mids but doesn't improve fills is not promoted.

### Close-window layer (T−120s to close cutoff)

- `fair_value.py` settlement model (moment-matched average, conditioned on Kalshi's
  published partial final-minute average). Already MC-verified. **[FACT]**
- Research question: how early does the averaging edge exceed fees+adverse selection?
  Current close cutoff (90s) may be conservative or aggressive — tune in replay.

## 6. Research Protocol (binding)

Full statistical rules live in `docs/validation.md`; the non-negotiables:

- **Unit of independence = market window** (stitched by ticker across restart fragments —
  `research.py` does this). Tick-level counts are descriptive only.
- **Splits:** chronological 60/20/20 with 1-window embargo. The 20% test split is touched
  once per promoted strategy, ever.
- **Data quality:** `--strict` exclusions (BRTI gaps >5s, reconnects, late starts) and the
  July-13 laptop recording quarantined. Every excluded window is counted and reported.
- **All replay evidence with fees ON (multiplier=1) and latency ON (150ms), plus stress
  runs at 75/300ms and fee 0/1.** A strategy whose sign flips under stress is an artifact.
- **Power discipline:** per-window IC standard error ≈ 1/√N windows. Screen features at
  ≥300 windows (detect |IC|≳0.12); promote strategies at ≥1,000 windows with block-bootstrap
  95% CI excluding zero on net markout AND settlement P&L.
- **Experiment log:** every experiment gets an entry in `docs/experiments/` (date,
  hypothesis, spec, result, verdict — including failures). No unlogged experiments; the
  graveyard is the defense against self-p-hacking.
- **Metrics vocabulary:** Brier/log-loss by decile × time bucket (calibration); per-window
  IC and hit rate at 100ms/500ms/1s/5s/30s (signals); fill rate, markout-given-fill,
  spread capture, adverse selection, fees, inventory P&L, settlement P&L (economics) —
  always decomposed, never a single P&L number.

## 7. Execution Policy

- **Post-only always** (`post_only=True` on every order — enforced in `build_order`).
- **Cancel-safety asymmetry:** safety cancels (stale data, divergence, close cutoff) are
  never throttled; re-quotes are throttled (`KALSHI_QUOTE_THROTTLE_S`, default 0.25s).
- **Queue preservation:** resting orders at the right price are kept while remaining size
  ≥ max(min size, 50% of target), clamped by target (v0.2.1 semantics).
- **Gates (all live):** book validity, data staleness (2s), ticker-vs-book divergence
  (sustained 5s → cancel all + reconnect), pricing availability, close cutoff (90s),
  position cap (50), notional cap, cash reserve.
- **Reconciliation:** private fill channel drives inventory in-loop; REST poll (1s) is the
  authority; divergence >1 contract twice consecutively = hard stop and cancel-all.
- **Failure posture:** ≥10 consecutive execution errors = hard stop. Process exit always
  attempts cancel-all; GTD expiry at close−1s is the backstop.

## 8. Risk Framework by Phase

| Phase | Capital | Position cap | Kill criteria |
|---|---|---|---|
| Paper | $0 | 50 | discrepancy/orphan/post-only violation → restart phase clock |
| Live pilot | ≤$500 collateral | 5–10 | day loss > $50; any unexplained position; any uncancelled orphan |
| Scale-up | staged ×2 max per week | tuned | weekly net < 0 after fees → freeze size; 2 losing weeks → back to paper |

Live pilot's **primary deliverables are measurements, not P&L**: empirical maker fee
(settles §10.1), real ack-latency distribution (recalibrates replay), realized-vs-simulated
fill rate (calibrates queue model), real toxicity labels (retrains Model T).

## 9. Phase Gates

- **P0 — Collection (ACTIVE).** Collector + archival running unattended. Exit: 300 clean
  windows (~3 days from 2026-07-14).
- **P1 — Calibration baseline.** `kalshi-mm-research --strict`: model-vs-mid Brier/log-loss
  by time bucket. Decides the quoting anchor (mid vs model vs regime blend). Exit: written
  result in experiment log.
- **P2 — Feature layer + Model T.** Build feature computation (one-time code), fit and
  promote/reject Model T per §5. Exit: A/B replay verdict at ≥300 windows.
- **P3 — Model V + close layer.** Same discipline. Exit: promoted or rejected at ≥1,000
  windows for any strategy candidate.
- **P4 — Paper.** 100 consecutive clean windows (validation.md operational gate), paper
  decisions cross-checked against replay on identical data.
- **P5 — Live pilot.** ≥2 weeks at pilot size. Exit to scale only when live markout matches
  paper within CI and §10.1 is resolved.
- **P6 — Scale + revisit non-goals (§4).**

## 10. Open Verification Queue

1. **Maker fee for CRYPTO15M — CRITICAL, blocks trust in every absolute number.** Current
   assumption: standard 0.0175 quadratic maker rate (conservative). Resolve via fee-schedule
   PDF (bot-blocked for us; human download works) or the fee field on the first real fill.
   If actual = 0, every replay improves by ~0.4¢/contract at mid prices.
2. **Incumbent footprint study.** He is likely in our recordings. Measure: persistent
   two-sided depth at consistent offsets, requote cadence, quote lifetimes, pull-latency
   after external moves. Output: what we must beat. (Research task, data already on disk.)
3. **Real ack latency.** `LiveExecutor` records request→engine timestamps; replace the
   150ms replay constant with the measured distribution once live.
4. **`market_lifecycle_v2` sequence skips** (39 in one audit): benign (multi-market channel)
   or lossy? Determine whether our own market's lifecycle events can be missed.
5. **Queue-sim calibration:** conservative queue-ahead assumption vs. reality — measurable
   only live (P5).

## 11. Known Limitations of Our Measurements

- Replay cannot model our own market impact or others' reactions to our quotes.
- Queue simulation is conservative on queue position, and cancel-in-flight fills are now
  modeled, but self-fill price improvement is not.
- Paper fills come from the same queue simulator — paper validates operations and state
  machines, not fill economics.
- The volatility forecaster's 1800s horizon never fills within one window; cross-window
  warm-start (implemented) mitigates but regime shifts at window boundaries remain.
- One clean recorded exchange outage/halt has never been observed; behavior under a Kalshi
  trading pause is untested (orders carry `cancel_order_on_pause=True`).

## 12. Decision Log (append-only)

- **2026-07-14** — v1.0 of this plan. Strategy reframed from "settlement-GBM maker with
  microstructure candidates" to "two fitted models (toxicity, value) + settlement close
  layer + queue discipline," on the strength of (a) the verified operator existence proof
  and its single-market, fitted-fair-value reading, (b) audit finding that adverse
  selection, not pricing elegance, is the dominant P&L channel. Breadth/census demoted to
  optionality. GBM model reclassified from "the fair value" to close-window layer + Model V
  anchor/feature.
- **2026-07-14** — Fees assumed at standard maker rate (multiplier 1) until verified;
  all prior zero-fee results invalidated.
- **2026-07-14** — 150ms latency became the replay default; zero-latency results
  invalidated for promotion purposes.
- **2026-07-14 (later)** — P2 tooling complete ahead of data: feature store
  (`kalshi-mm-features` + 5-min compaction timer on the collector host), Model T
  label pipeline (`kalshi-mm-labels`), fit/eval harness with holdout lock
  (`kalshi-mm-fit`), and the A/B promotion instrument (`kalshi-mm-ab` with
  baseline/microprice-skew/toxicity-gate variants). Toxic-fill label defined as
  side-adjusted 5s markout ≤ −1¢ (one tick, filters bounce noise). From here the
  binding constraint is window count: screening at ≥300 (~2026-07-17), promotion
  at ≥1,000 (~2026-07-25).
