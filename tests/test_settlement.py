from kalshi_mm.exchange import KalshiMarket, PriceRange
from kalshi_mm.settlement import infer_settlement


def _market() -> KalshiMarket:
    return KalshiMarket(
        ticker="KXBTC15M-TEST",
        event_ticker="KXBTC15M",
        title="test",
        status="closed",
        open_ts=840,
        close_ts=1_000,
        expected_expiration_ts=1_300,
        strike_average=100.0,
        price_ranges=(PriceRange(0.0, 1.0, 0.01),),
        rules_primary="test",
        rules_secondary="test",
    )


def _brti_event(*, value: float, source_ts_ms: int, average: float | None = None, count: int = 0):
    final = None
    if average is not None:
        final = {
            "value": str(average),
            "window_size": count,
            "window_end_ts_exclusive": 1_000_000,
        }
    return {
        "event_type": "cfbenchmarks_value",
        "payload": {
            "type": "cfbenchmarks_value",
            "msg": {
                "index_id": "BRTI",
                "data": f'{{"time":{source_ts_ms},"value":"{value}"}}',
                "last_60s_windowed_average_15min": final,
            },
        },
    }


def test_official_result_takes_precedence():
    rows = [
        _brti_event(value=90.0, source_ts_ms=1_000_000, average=90.0, count=60),
        {"event_type": "market_result", "payload": {"result": "yes"}},
    ]
    outcome = infer_settlement(rows, _market())
    assert outcome is not None
    assert outcome.result == "yes"
    assert outcome.source == "official"


def test_complete_final_average_is_preferred_over_last_tick():
    rows = [_brti_event(value=101.0, source_ts_ms=1_000_000, average=99.5, count=60)]
    outcome = infer_settlement(rows, _market())
    assert outcome is not None
    assert outcome.result == "no"
    assert outcome.source == "brti_final_60s_average"
    assert outcome.reference_value == 99.5


def test_last_preclose_tick_is_explicit_fallback():
    rows = [
        _brti_event(value=101.0, source_ts_ms=999_000),
        _brti_event(value=90.0, source_ts_ms=1_001_000),
    ]
    outcome = infer_settlement(rows, _market())
    assert outcome is not None
    assert outcome.result == "yes"
    assert outcome.source == "last_preclose_brti_tick"
