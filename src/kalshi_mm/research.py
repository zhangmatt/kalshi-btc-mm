from __future__ import annotations

import argparse
import bisect
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from .exchange import BrtiState, KalshiMarket, KalshiOrderBook
from .recorder import iter_events
from .replay import extract_brti_ticks
from .settlement import infer_settlement
from .fair_value import kalshi_btc15m_fair_value
from .volatility import MultiHorizonVolatility

VOL_PREWARM_WINDOW_MS = 3_600_000


@dataclass(frozen=True)
class Prediction:
    ts_ms: int
    mid: float
    fairs: dict[str, float]


@dataclass(frozen=True)
class WindowQuality:
    ticker: str
    paths: tuple[str, ...]
    fragments: int
    start_offset_s: Optional[float]
    brti_max_gap_s: float
    kalshi_reconnects: int
    flags: tuple[str, ...]


@dataclass(frozen=True)
class PreparedWindow:
    market: KalshiMarket
    rows: tuple[dict[str, Any], ...]
    quality: WindowQuality


@dataclass(frozen=True)
class ModelScore:
    model: str
    samples: int
    windows: int
    brier: Optional[float]
    log_loss: Optional[float]
    mean_abs_fair_mid: float
    signal_ic: dict[int, Optional[float]]
    directional_hit_rate: dict[int, Optional[float]]


def _correlation(xs: list[float], ys: list[float]) -> Optional[float]:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    covariance = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    return covariance / math.sqrt(vx * vy) if vx > 0 and vy > 0 else None


def _future_signal_ic(predictions: list[Prediction], model: str, horizon_ms: int) -> Optional[float]:
    timestamps = [row.ts_ms for row in predictions]
    signals: list[float] = []
    moves: list[float] = []
    for row in predictions:
        index = bisect.bisect_left(timestamps, row.ts_ms + horizon_ms)
        if index >= len(predictions):
            continue
        signals.append(row.fairs[model] - row.mid)
        moves.append(predictions[index].mid - row.mid)
    return _correlation(signals, moves)


def _future_hit_rate(predictions: list[Prediction], model: str, horizon_ms: int) -> Optional[float]:
    timestamps = [row.ts_ms for row in predictions]
    hits = 0
    samples = 0
    for row in predictions:
        index = bisect.bisect_left(timestamps, row.ts_ms + horizon_ms)
        if index >= len(predictions):
            continue
        signal = row.fairs[model] - row.mid
        move = predictions[index].mid - row.mid
        if abs(signal) <= 1e-12 or abs(move) <= 1e-12:
            continue
        hits += int(signal * move > 0)
        samples += 1
    return hits / samples if samples else None


def prepare_windows(paths: Iterable[str | Path]) -> list[PreparedWindow]:
    """Group recording fragments by market so one market is exactly one window."""
    groups: dict[str, dict[str, Any]] = {}
    for path in paths:
        rows = list(iter_events(path))
        metadata = next((row["payload"] for row in rows if row["event_type"] == "market_metadata"), None)
        if metadata is None:
            raise ValueError(f"{path} has no market metadata")
        market = KalshiMarket.from_api(metadata)
        group = groups.setdefault(market.ticker, {"market": market, "files": []})
        group["files"].append((str(path), rows))

    windows: list[PreparedWindow] = []
    for group in groups.values():
        market: KalshiMarket = group["market"]
        files = sorted(
            group["files"],
            key=lambda item: min((int(row["receive_ts_ms"]) for row in item[1]), default=0),
        )
        rows = tuple(row for _, file_rows in files for row in file_rows)
        windows.append(
            PreparedWindow(
                market=market,
                rows=rows,
                quality=_window_quality(market, rows, tuple(path for path, _ in files)),
            )
        )
    windows.sort(key=lambda window: window.market.open_ts)
    return windows


def _window_quality(
    market: KalshiMarket,
    rows: tuple[dict[str, Any], ...],
    paths: tuple[str, ...],
) -> WindowQuality:
    open_ms = market.open_ts * 1_000
    close_ms = market.close_ts * 1_000
    first_ts = min((int(row["receive_ts_ms"]) for row in rows), default=None)
    start_offset = (first_ts - open_ms) / 1_000 if first_ts is not None else None
    brti_ts = [ts for ts, _ in extract_brti_ticks(rows) if open_ms <= ts <= close_ms]
    brti_max_gap = max(
        ((later - earlier) / 1_000 for earlier, later in zip(brti_ts, brti_ts[1:])),
        default=0.0,
    )
    reconnects = sum(
        1
        for row in rows
        if row.get("source") == "kalshi"
        and row.get("event_type") == "feed_status"
        and (row.get("payload") or {}).get("status") == "connected"
    )
    reconnects = max(0, reconnects - len(paths))  # one connect per fragment is expected
    flags: list[str] = []
    if len(paths) > 1:
        flags.append("fragmented")
    if brti_max_gap > 5.0:
        flags.append("brti_gap")
    if reconnects > 0:
        flags.append("reconnect")
    if start_offset is not None and start_offset > 120.0:
        flags.append("late_start")
    return WindowQuality(
        ticker=market.ticker,
        paths=paths,
        fragments=len(paths),
        start_offset_s=start_offset,
        brti_max_gap_s=brti_max_gap,
        kalshi_reconnects=reconnects,
        flags=tuple(flags),
    )


def _window_predictions(
    window: PreparedWindow,
    prewarm_ticks: list[tuple[int, float]],
) -> tuple[list[Prediction], Optional[float]]:
    market = window.market
    rows = window.rows
    settlement = infer_settlement(rows, market)
    outcome = 1.0 if settlement and settlement.result == "yes" else 0.0 if settlement else None
    book = KalshiOrderBook(market.ticker)
    brti = BrtiState()
    volatility = MultiHorizonVolatility()
    for ts_ms, value in prewarm_ticks:
        volatility.update(ts_ms, value)
    latest_venues: dict[str, tuple[int, float]] = {}
    predictions: list[Prediction] = []
    last_emit_ms = 0

    for row in rows:
        ts_ms = int(row["receive_ts_ms"])
        source = row["source"]
        payload = row["payload"]
        if source in {"binance", "coinbase", "kraken"} and row["event_type"] == "spot":
            price = float(payload["price"])
            latest_venues[source] = (int(row.get("exchange_ts_ms") or ts_ms), price)
        if source == "kalshi":
            book.apply(payload, use_yes_price=True)
            if brti.apply(payload) and brti.value is not None:
                volatility.update(brti.source_ts_ms or ts_ms, brti.value)
        if ts_ms / 1000.0 > market.close_ts:
            continue
        if ts_ms - last_emit_ms < 250:
            continue
        bbo = book.snapshot()
        forecast = volatility.forecast()
        if not bbo.valid or bbo.bid is None or bbo.ask is None or brti.value is None or forecast is None:
            continue
        seconds_to_close = max(0.0, market.close_ts - ts_ms / 1000.0)
        observed = brti.final_minute_values if seconds_to_close <= 60 else ()
        fresh_prices = [price for venue_ts, price in latest_venues.values() if ts_ms - venue_ts <= 2_000]
        proxy = sorted(fresh_prices)[len(fresh_prices) // 2] if fresh_prices else brti.value
        fairs: dict[str, float] = {"market_mid": (bbo.bid + bbo.ask) / 2.0}
        for multiplier in (0.75, 1.0, 1.25):
            name = f"sigma_{multiplier:.2f}"
            fairs[name] = kalshi_btc15m_fair_value(
                strike_average=market.strike_average,
                spot=brti.value,
                sigma_per_s=forecast.sigma_per_s * multiplier,
                seconds_to_close=seconds_to_close,
                observed_final_values=observed,
            ).yes
        for beta in (0.25, 0.5, 1.0):
            name = f"basis_{beta:.2f}"
            tilted_spot = brti.value + beta * (proxy - brti.value)
            fairs[name] = kalshi_btc15m_fair_value(
                strike_average=market.strike_average,
                spot=tilted_spot,
                sigma_per_s=forecast.sigma_per_s,
                seconds_to_close=seconds_to_close,
                observed_final_values=observed,
            ).yes
        microprice = fairs["market_mid"]
        if bbo.bid_size > 0 and bbo.ask_size > 0:
            microprice = (bbo.ask * bbo.bid_size + bbo.bid * bbo.ask_size) / (bbo.bid_size + bbo.ask_size)
        for weight in (0.25, 0.5):
            fairs[f"microprice_{weight:.2f}"] = fairs["sigma_1.00"] + weight * (
                microprice - fairs["market_mid"]
            )
            fairs[f"microprice_{weight:.2f}"] = min(0.999, max(0.001, fairs[f"microprice_{weight:.2f}"]))
        predictions.append(Prediction(ts_ms, (bbo.bid + bbo.ask) / 2.0, fairs))
        last_emit_ms = ts_ms
    return predictions, outcome


def score_windows(prepared: list[PreparedWindow], *, strict: bool = False) -> list[ModelScore]:
    """Score models per market window, warming volatility from prior windows' BRTI."""
    windows: list[tuple[list[Prediction], Optional[float]]] = []
    prior_ticks: list[tuple[int, float]] = []
    for window in prepared:
        first_ts = min((int(row["receive_ts_ms"]) for row in window.rows), default=0)
        prewarm = [tick for tick in prior_ticks if first_ts - VOL_PREWARM_WINDOW_MS <= tick[0] < first_ts]
        if not (strict and window.quality.flags):
            windows.append(_window_predictions(window, prewarm))
        prior_ticks.extend(extract_brti_ticks(window.rows))
        cutoff = first_ts - VOL_PREWARM_WINDOW_MS
        prior_ticks = [tick for tick in prior_ticks if tick[0] >= cutoff]

    model_names = sorted({name for predictions, _ in windows for row in predictions for name in row.fairs})
    scores = []
    for model in model_names:
        all_rows = [row for predictions, _ in windows for row in predictions]
        resolved = [(row, outcome) for predictions, outcome in windows if outcome is not None for row in predictions]
        brier = sum((row.fairs[model] - outcome) ** 2 for row, outcome in resolved) / len(resolved) if resolved else None
        log_loss = None
        if resolved:
            losses = []
            for row, outcome in resolved:
                probability = min(1 - 1e-9, max(1e-9, row.fairs[model]))
                losses.append(-(outcome * math.log(probability) + (1 - outcome) * math.log(1 - probability)))
            log_loss = sum(losses) / len(losses)
        mae = sum(abs(row.fairs[model] - row.mid) for row in all_rows) / len(all_rows) if all_rows else float("nan")
        horizons = (100, 500, 1_000, 5_000, 30_000)
        signal_ic: dict[int, Optional[float]] = {}
        hit_rate: dict[int, Optional[float]] = {}
        for horizon in horizons:
            ic_values = [_future_signal_ic(predictions, model, horizon) for predictions, _ in windows]
            valid_ic = [value for value in ic_values if value is not None]
            signal_ic[horizon] = sum(valid_ic) / len(valid_ic) if valid_ic else None
            hit_values = [_future_hit_rate(predictions, model, horizon) for predictions, _ in windows]
            valid_hits = [value for value in hit_values if value is not None]
            hit_rate[horizon] = sum(valid_hits) / len(valid_hits) if valid_hits else None
        scores.append(
            ModelScore(
                model=model,
                samples=len(all_rows),
                windows=len(windows),
                brier=brier,
                log_loss=log_loss,
                mean_abs_fair_mid=mae,
                signal_ic=signal_ic,
                directional_hit_rate=hit_rate,
            )
        )
    return scores


def score_recordings(paths: Iterable[str | Path], *, strict: bool = False) -> list[ModelScore]:
    return score_windows(prepare_windows(paths), strict=strict)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Compare Kalshi settlement-vol and basis models.")
    parser.add_argument("recordings", nargs="+")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exclude windows flagged for BRTI gaps, reconnects, or very late starts",
    )
    args = parser.parse_args(argv)
    prepared = prepare_windows(args.recordings)
    print("# window quality (one market = one window)")
    for window in prepared:
        quality = window.quality
        print(
            f"# {quality.ticker} fragments={quality.fragments} "
            f"start_offset_s={quality.start_offset_s if quality.start_offset_s is not None else 'n/a'} "
            f"brti_max_gap_s={quality.brti_max_gap_s:.1f} reconnects={quality.kalshi_reconnects} "
            f"flags={','.join(quality.flags) or '-'}"
        )
    excluded = [window.quality.ticker for window in prepared if args.strict and window.quality.flags]
    if excluded:
        print(f"# excluded {len(excluded)} flagged window(s): {', '.join(excluded)}")
    horizons = (100, 500, 1_000, 5_000, 30_000)
    suffixes = [f"ic_{horizon}ms" for horizon in horizons] + [f"hit_{horizon}ms" for horizon in horizons]
    print(",".join(["model", "samples", "windows", "brier", "log_loss", "mae_fair_mid", *suffixes]))
    for score in score_windows(prepared, strict=args.strict):
        values = [
            score.model,
            str(score.samples),
            str(score.windows),
            str(score.brier) if score.brier is not None else "",
            str(score.log_loss) if score.log_loss is not None else "",
            str(score.mean_abs_fair_mid),
            *[str(score.signal_ic[horizon]) if score.signal_ic[horizon] is not None else "" for horizon in horizons],
            *[
                str(score.directional_hit_rate[horizon])
                if score.directional_hit_rate[horizon] is not None
                else ""
                for horizon in horizons
            ],
        ]
        print(",".join(values))


if __name__ == "__main__":
    main()
