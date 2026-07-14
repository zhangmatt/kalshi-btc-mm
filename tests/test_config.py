from kalshi_mm.config import KalshiRuntimeConfig


def test_binance_us_reference_feed_selects_binance_us_depth():
    config = KalshiRuntimeConfig.from_env(
        {
            "KALSHI_API_KEY_ID": "test",
            "KALSHI_PRIVATE_KEY_PATH": "/tmp/test.pem",
            "KALSHI_BINANCE_WS_URL": "wss://stream.binance.us:9443/ws/btcusdt@bookTicker",
        }
    )
    assert config.binance_depth_ws_url == "wss://stream.binance.us:9443/ws/btcusdt@depth20@100ms"


def test_explicit_binance_depth_url_wins():
    config = KalshiRuntimeConfig.from_env(
        {
            "KALSHI_API_KEY_ID": "test",
            "KALSHI_PRIVATE_KEY_PATH": "/tmp/test.pem",
            "KALSHI_BINANCE_WS_URL": "wss://stream.binance.us:9443/ws/btcusdt@bookTicker",
            "KALSHI_BINANCE_DEPTH_WS_URL": "wss://example.test/depth",
        }
    )
    assert config.binance_depth_ws_url == "wss://example.test/depth"
