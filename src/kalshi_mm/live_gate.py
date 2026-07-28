"""Wire the validated toxicity gate (toxgate6) into the live quoting loop.

Research validated the gate as a replay QuoteAdjuster; this adapter feeds it
live order-book + trade-flow state so the runner can apply the same per-side
toxicity edge in paper/live. External-venue features are imputed to the training
mean for now (the runner does not yet collect external L2 depth); the gate's
dominant features are Kalshi book microstructure, so this is a documented,
minor degradation to be closed before scaling live.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Optional

from .ab import ToxicityGate
from .models import load_models
from .replay import QuoteContext

FLOW_WINDOW_MS = 5_000
DELTA_SAMPLE_MS = 250  # match the feature store's bid/ask-size-delta cadence


class LiveToxicityGate:
    """Compute per-side toxicity dollars from live state, using the fitted model."""

    def __init__(self, model_path: str, *, scale_dollars: float = 0.06):
        self.gate = ToxicityGate(load_models(model_path), scale_dollars=scale_dollars)
        self._flow: deque[tuple[int, float]] = deque()
        self._prev_bid_size: Optional[float] = None
        self._prev_ask_size: Optional[float] = None
        self._last_delta_ms = 0

    def on_trade(self, ts_ms: int, taker_book_side: str, count: float) -> None:
        if taker_book_side not in {"bid", "ask"} or count <= 0:
            return
        self._flow.append((ts_ms, count if taker_book_side == "bid" else -count))
        cutoff = ts_ms - FLOW_WINDOW_MS
        while self._flow and self._flow[0][0] < cutoff:
            self._flow.popleft()

    def toxicity_dollars(
        self,
        *,
        now_ms: int,
        book: Any,
        bbo: Any,
        fair_yes: float,
        seconds_to_close: float,
    ) -> tuple[float, float]:
        """Return (bid_toxicity_dollars, ask_toxicity_dollars) for the strategy."""
        # Size deltas sampled on the feature store's cadence so live values match
        # the distribution the model was trained on.
        bid_delta = ask_delta = 0.0
        if self._prev_bid_size is not None:
            bid_delta = bbo.bid_size - self._prev_bid_size
            ask_delta = bbo.ask_size - self._prev_ask_size
        if self._last_delta_ms == 0 or now_ms - self._last_delta_ms >= DELTA_SAMPLE_MS:
            self._prev_bid_size, self._prev_ask_size = bbo.bid_size, bbo.ask_size
            self._last_delta_ms = now_ms
        ctx = QuoteContext(
            ts_ms=now_ms,
            seconds_to_close=seconds_to_close,
            fair_yes=fair_yes,
            bbo=bbo,
            book=book,
            flow_5s=sum(count for _, count in self._flow),
            flow_1s=sum(count for ts, count in self._flow if ts >= now_ms - 1_000),
            trades_5s=len(self._flow),
            bid_size_delta=bid_delta,
            ask_size_delta=ask_delta,
            ext_micro_lead_bps=None,  # imputed to training mean by the gate
            ext_imbalance=None,
        )
        _, bid_tox, ask_tox = self.gate.adjust(ctx)
        return bid_tox, ask_tox
