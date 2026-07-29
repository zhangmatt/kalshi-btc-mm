# 2026-07-29 — Paper mode loses; cause is REGIME, not a bug

Paper (toxgate6, 71 settled windows, Jul 28-29): settle -$2.54/window, win 39%,
mean end position 21 contracts.

Diagnostic — replay toxgate6 on the EXACT 20 windows paper traded:
| | settle/win | markout/contract |
|---|---|---|
| replay baseline | -$4.04 | -1.11c |
| replay toxgate6 | -$2.82 | -1.16c |
| paper toxgate6 (same period) | -$2.54 | — |

## Findings
1. paper (-$2.54) ~= replay (-$2.82) -> the live runner and the replay sim
   AGREE on the same data. No runner-vs-replay bug; tooling is internally
   consistent.
2. REGIME shift: baseline -$0.68 (Jul13-28 validation) -> -$4.04 (Jul28-29);
   markout -0.4c -> -1.1c. Adverse selection ~tripled. Making this market is
   broadly unprofitable in the current regime.
3. The gate is a robust RELATIVE edge (+$1.2/window vs baseline here, +$0.7 in
   validation) but cannot rescue a losing regime: +$1.2 on a -$4 baseline = -$2.8.

## Conclusion
toxgate6's ABSOLUTE profitability is regime-dependent; the confirmatory +$0.026
was a gentle-regime figure. Going straight to live would have lost real money
this week. The disciplined ladder caught it at zero cost — the core validation
of the whole process.

## Next direction
A REGIME FILTER: measure realized adverse-selection / vol in a rolling window
and stand down (stop quoting) when elevated, so bad regimes go to ~flat instead
of -$2.8. Turns a regime-dependent coin flip into a trade-only-when-favorable
strategy. Also revisit inventory flattening before close (21-contract carry).
