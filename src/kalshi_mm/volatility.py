from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, Optional


@dataclass(frozen=True)
class VolatilityForecast:
    sigma_per_s: float
    ewma_sigma_per_s: float
    horizon_sigmas: dict[int, float]
    jump_ratio: float


class MultiHorizonVolatility:
    """One-second sampled, multi-horizon realized-volatility forecast."""

    def __init__(
        self,
        horizons_s: tuple[int, ...] = (10, 60, 300, 1800),
        *,
        ewma_half_life_s: float = 60.0,
        max_history_s: int = 3600,
    ):
        if not horizons_s or any(value <= 1 for value in horizons_s):
            raise ValueError("horizons must be greater than one second")
        if ewma_half_life_s <= 0:
            raise ValueError("ewma_half_life_s must be positive")
        self.horizons_s = tuple(sorted(set(horizons_s)))
        self.ewma_decay = math.exp(math.log(0.5) / ewma_half_life_s)
        self.max_history_ms = max_history_s * 1000
        self._seconds: Deque[tuple[int, float]] = deque()

    def update(self, ts_ms: int, price: float) -> None:
        if price <= 0:
            return
        second = int(ts_ms) // 1000
        if self._seconds and self._seconds[-1][0] == second:
            self._seconds[-1] = (second, float(price))
        elif not self._seconds or second > self._seconds[-1][0]:
            self._seconds.append((second, float(price)))
        cutoff = int(ts_ms) - self.max_history_ms
        while self._seconds and self._seconds[0][0] * 1000 < cutoff:
            self._seconds.popleft()

    def forecast(self, *, min_sigma: float = 1e-6) -> Optional[VolatilityForecast]:
        points = list(self._seconds)
        if len(points) < 4:
            return None
        returns: list[tuple[int, float]] = []
        for (t0, p0), (t1, p1) in zip(points, points[1:]):
            dt = t1 - t0
            if dt <= 0:
                continue
            returns.append((t1, math.log(p1 / p0) / math.sqrt(dt)))
        if len(returns) < 3:
            return None

        variance = returns[0][1] ** 2
        for _, ret in returns[1:]:
            variance = self.ewma_decay * variance + (1.0 - self.ewma_decay) * ret * ret
        ewma_sigma = math.sqrt(max(0.0, variance))

        latest_second = returns[-1][0]
        horizon_sigmas: dict[int, float] = {}
        for horizon in self.horizons_s:
            sample = [ret for second, ret in returns if second > latest_second - horizon]
            if len(sample) >= 3:
                horizon_sigmas[horizon] = math.sqrt(sum(ret * ret for ret in sample) / len(sample))

        if not horizon_sigmas:
            return None
        weights = {10: 0.15, 60: 0.35, 300: 0.30, 1800: 0.20}
        weighted = [(sigma, weights.get(horizon, 1.0)) for horizon, sigma in horizon_sigmas.items()]
        realized_sigma = sum(sigma * weight for sigma, weight in weighted) / sum(weight for _, weight in weighted)
        sigma = max(min_sigma, 0.5 * ewma_sigma + 0.5 * realized_sigma)

        abs_returns = sorted(abs(ret) for _, ret in returns)
        median = abs_returns[len(abs_returns) // 2]
        jump_cutoff = max(5.0 * median, 1e-12)
        jump_variance = sum(ret * ret for _, ret in returns if abs(ret) > jump_cutoff)
        total_variance = sum(ret * ret for _, ret in returns)
        jump_ratio = jump_variance / total_variance if total_variance > 0 else 0.0
        return VolatilityForecast(sigma, ewma_sigma, horizon_sigmas, jump_ratio)


def forecast_from_ticks(ticks: Iterable[tuple[int, float]]) -> Optional[VolatilityForecast]:
    model = MultiHorizonVolatility()
    for ts_ms, price in ticks:
        model.update(ts_ms, price)
    return model.forecast()
