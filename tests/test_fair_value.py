import math

import pytest

from kalshi_mm.fair_value import arithmetic_average_moments, kalshi_btc15m_fair_value


def test_arithmetic_average_uses_correlated_fixings():
    mean, variance = arithmetic_average_moments(
        spot=100.0,
        sigma_per_s=0.001,
        fixing_times_s=[1.0, 2.0, 3.0],
    )
    assert mean == pytest.approx(100.0)
    assert variance > 0


def test_settlement_fair_is_near_half_at_the_strike():
    fair = kalshi_btc15m_fair_value(
        strike_average=100.0,
        spot=100.0,
        sigma_per_s=0.001,
        seconds_to_close=900.0,
    )
    assert 0.45 < fair.yes < 0.55
    assert fair.remaining_count == 60


def test_final_minute_conditions_on_observed_average():
    high = kalshi_btc15m_fair_value(
        strike_average=100.0,
        spot=100.0,
        sigma_per_s=0.001,
        seconds_to_close=1.0,
        observed_final_values=[101.0] * 59,
    )
    low = kalshi_btc15m_fair_value(
        strike_average=100.0,
        spot=100.0,
        sigma_per_s=0.001,
        seconds_to_close=1.0,
        observed_final_values=[99.0] * 59,
    )
    assert high.yes > 0.99
    assert low.yes < 0.01


def test_pre_window_fixings_are_in_final_minute():
    result = kalshi_btc15m_fair_value(
        strike_average=100.0,
        spot=100.0,
        sigma_per_s=0.0,
        drift_per_s=0.001,
        seconds_to_close=900.0,
    )
    expected = sum(100.0 * math.exp(0.001 * second) for second in range(841, 901)) / 60.0
    assert result.expected_average == pytest.approx(expected)
