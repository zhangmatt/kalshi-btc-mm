from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Iterable, Mapping, Optional

from .exchange import KalshiMarket


@dataclass(frozen=True)
class SettlementOutcome:
    result: str
    source: str
    reference_value: Optional[float] = None


def _rounded_cents(value: float) -> Decimal:
    return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def infer_settlement(rows: Iterable[Mapping[str, Any]], market: KalshiMarket) -> Optional[SettlementOutcome]:
    """Resolve from the official result, final BRTI average, or last pre-close BRTI tick."""

    rows = list(rows)
    for event_type in ("market_result", "market_status_at_close", "market_metadata"):
        for row in reversed(rows):
            if row.get("event_type") != event_type:
                continue
            result = str((row.get("payload") or {}).get("result", "")).lower()
            if result in {"yes", "no"}:
                return SettlementOutcome(result=result, source="official")

    close_ms = market.close_ts * 1_000
    complete_average: Optional[tuple[int, float]] = None
    last_tick: Optional[tuple[int, float]] = None
    for row in rows:
        payload = row.get("payload") or {}
        if payload.get("type") != "cfbenchmarks_value":
            continue
        msg = payload.get("msg") or {}
        if msg.get("index_id") != "BRTI":
            continue
        try:
            raw = json.loads(msg.get("data") or "{}")
            source_ts_ms = int(raw["time"])
            value = float(raw["value"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue

        if source_ts_ms <= close_ms and (last_tick is None or source_ts_ms > last_tick[0]):
            last_tick = (source_ts_ms, value)

        final = msg.get("last_60s_windowed_average_15min") or {}
        try:
            count = int(final.get("window_size", 0))
            window_end_ms = int(final["window_end_ts_exclusive"])
            average = float(final["value"])
        except (KeyError, TypeError, ValueError):
            continue
        # Observed feeds publish window_end exactly at close; a loose tolerance
        # could accept an average offset by one fixing on a knife-edge market.
        if count >= 60 and abs(window_end_ms - close_ms) <= 250:
            if complete_average is None or window_end_ms > complete_average[0]:
                complete_average = (window_end_ms, average)

    if complete_average is not None:
        average = complete_average[1]
        result = "yes" if _rounded_cents(average) >= _rounded_cents(market.strike_average) else "no"
        return SettlementOutcome(result=result, source="brti_final_60s_average", reference_value=average)

    if last_tick is not None:
        value = last_tick[1]
        result = "yes" if _rounded_cents(value) >= _rounded_cents(market.strike_average) else "no"
        return SettlementOutcome(result=result, source="last_preclose_brti_tick", reference_value=value)

    return None
