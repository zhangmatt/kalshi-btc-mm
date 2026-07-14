from kalshi_mm.volatility import MultiHorizonVolatility


def test_volatility_resamples_multiple_ticks_in_same_second():
    model = MultiHorizonVolatility(horizons_s=(10,))
    for second in range(12):
        model.update(second * 1000, 100.0 + second * 0.01)
        model.update(second * 1000 + 900, 100.0 + second * 0.01 + 0.001)
    forecast = model.forecast()
    assert forecast is not None
    assert forecast.sigma_per_s > 0
    assert 10 in forecast.horizon_sigmas


def test_constant_prices_use_minimum_sigma():
    model = MultiHorizonVolatility(horizons_s=(10,))
    for second in range(12):
        model.update(second * 1000, 100.0)
    forecast = model.forecast(min_sigma=1e-5)
    assert forecast is not None
    assert forecast.sigma_per_s == 1e-5

