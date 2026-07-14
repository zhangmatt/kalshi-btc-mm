from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from .gbm import clamp, gaussian_cdf


@dataclass(frozen=True)
class SettlementFairValue:
    yes: float
    no: float
    expected_average: float
    average_std: float
    observed_count: int
    remaining_count: int


def _future_fixing_times(
    seconds_to_close: float,
    remaining_count: int,
    *,
    total_fixings: int,
    observed_count: int,
    settlement_window_s: float = 60.0,
) -> list[float]:
    """Times, in seconds, of the remaining settlement fixings.

    Before the settlement window begins, all fixings sit in the final minute.
    Once Kalshi reports partial final-minute observations, the remaining fixings
    are distributed over the time left until close.
    """
    if remaining_count <= 0 or seconds_to_close <= 0:
        return []
    if observed_count == 0 and seconds_to_close >= settlement_window_s:
        fixing_interval = settlement_window_s / total_fixings
        first = seconds_to_close - settlement_window_s + fixing_interval
        return [first + fixing_interval * index for index in range(remaining_count)]
    spacing = seconds_to_close / remaining_count
    return [spacing * (index + 1) for index in range(remaining_count)]


def arithmetic_average_moments(
    *,
    spot: float,
    sigma_per_s: float,
    fixing_times_s: Sequence[float],
    drift_per_s: float = 0.0,
) -> tuple[float, float]:
    """Return mean and variance of a discrete arithmetic average under GBM."""
    if spot <= 0:
        raise ValueError("spot must be positive")
    if sigma_per_s < 0:
        raise ValueError("sigma_per_s cannot be negative")
    if not fixing_times_s:
        return spot, 0.0

    means = [spot * math.exp(drift_per_s * max(0.0, t)) for t in fixing_times_s]
    variance_sum = 0.0
    sigma_sq = sigma_per_s * sigma_per_s
    for i, ti in enumerate(fixing_times_s):
        for j, tj in enumerate(fixing_times_s):
            covariance = means[i] * means[j] * (math.exp(sigma_sq * min(ti, tj)) - 1.0)
            variance_sum += covariance
    count = len(fixing_times_s)
    return sum(means) / count, variance_sum / (count * count)


def kalshi_btc15m_fair_value(
    *,
    strike_average: float,
    spot: float,
    sigma_per_s: float,
    seconds_to_close: float,
    observed_final_values: Iterable[float] = (),
    drift_per_s: float = 0.0,
    fixings: int = 60,
    min_probability: float = 0.001,
    max_probability: float = 0.999,
) -> SettlementFairValue:
    """Price YES for Kalshi's final-60s-average >= initial-60s-average payoff.

    Observed final-minute BRTI values are conditioned on exactly. The remaining
    arithmetic average is approximated by matching the first two GBM moments to
    a lognormal distribution, which is stable and fast enough for a quote loop.
    """
    observed = [float(value) for value in observed_final_values if float(value) > 0]
    if len(observed) > fixings:
        observed = observed[-fixings:]
    remaining = fixings - len(observed)
    observed_sum = sum(observed)

    if remaining == 0 or seconds_to_close <= 0:
        if remaining == 0:
            average = observed_sum / fixings if observed else spot
        else:
            # Time exhausted with missing fixings: the unobserved fixings settle at
            # (approximately) the current spot, not zero.
            average = (observed_sum + remaining * spot) / fixings
        yes = 1.0 if average >= strike_average else 0.0
        return SettlementFairValue(yes, 1.0 - yes, average, 0.0, len(observed), remaining if seconds_to_close <= 0 else 0)

    fixing_times = _future_fixing_times(
        seconds_to_close,
        remaining,
        total_fixings=fixings,
        observed_count=len(observed),
    )
    future_mean, future_variance = arithmetic_average_moments(
        spot=spot,
        sigma_per_s=sigma_per_s,
        fixing_times_s=fixing_times,
        drift_per_s=drift_per_s,
    )
    expected_total = observed_sum + remaining * future_mean
    required_future_sum = fixings * strike_average - observed_sum
    expected_average = expected_total / fixings
    average_variance = (remaining * remaining * future_variance) / (fixings * fixings)
    average_std = math.sqrt(max(0.0, average_variance))

    if required_future_sum <= 0:
        probability = 1.0
    elif future_variance <= 0 or future_mean <= 0:
        probability = 1.0 if remaining * future_mean >= required_future_sum else 0.0
    else:
        target_future_average = required_future_sum / remaining
        cv_sq = future_variance / (future_mean * future_mean)
        log_variance = math.log1p(max(0.0, cv_sq))
        if log_variance <= 0:
            probability = 1.0 if future_mean >= target_future_average else 0.0
        else:
            log_mean = math.log(future_mean) - 0.5 * log_variance
            z = (math.log(target_future_average) - log_mean) / math.sqrt(log_variance)
            probability = 1.0 - gaussian_cdf(z)

    yes = clamp(probability, min_probability, max_probability)
    return SettlementFairValue(
        yes=yes,
        no=1.0 - yes,
        expected_average=expected_average,
        average_std=average_std,
        observed_count=len(observed),
        remaining_count=remaining,
    )
