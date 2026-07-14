from kalshi_mm.collect import _choose_market
from kalshi_mm.exchange import KalshiMarket


def _market(ticker: str, status: str, open_time: str, close_time: str) -> KalshiMarket:
    return KalshiMarket.from_api(
        {
            "ticker": ticker,
            "event_ticker": "KXBTC15M",
            "title": ticker,
            "status": status,
            "open_time": open_time,
            "close_time": close_time,
            "expected_expiration_time": close_time,
            "floor_strike": 60_000,
            "price_ranges": [{"start": "0", "end": "1", "step": "0.01"}],
        }
    )


class DiscoveryClient:
    def __init__(self):
        self.calls = 0

    def discover_btc15m(self, *, status="open"):
        self.calls += 1
        if self.calls == 1:
            return []
        return [_market("NEXT", "active", "2099-01-01T00:00:00Z", "2099-01-01T00:15:00Z")]


def test_choose_market_waits_until_market_is_fully_active(monkeypatch):
    client = DiscoveryClient()
    monkeypatch.setattr("kalshi_mm.collect.time.sleep", lambda _seconds: None)
    market = _choose_market(client, None, wait=True)
    assert market.ticker == "NEXT"
    assert client.calls == 2
