from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping

from .exchange import (
    DEMO_REST_URL,
    DEMO_WS_URL,
    PROD_REST_URL,
    PROD_WS_URL,
    KalshiCredentials,
)
from .strategy import KalshiStrategyConfig


def _bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    value = env.get(name)
    return default if value is None else value.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class KalshiRuntimeConfig:
    environment: str = "production"
    rest_url: str = PROD_REST_URL
    ws_url: str = PROD_WS_URL
    data_dir: str = "data/kalshi"
    subaccount: int = 0
    position_poll_s: float = 1.0
    order_reconcile_s: float = 1.0
    quote_throttle_s: float = 0.25
    allow_proxy_pricing: bool = False
    ws_insecure: bool = False
    binance_ws_url: str = "wss://stream.binance.com:9443/ws/btcusdt@trade"
    external_depth_enabled: bool = True
    external_depth_levels: int = 10
    external_depth_sample_hz: float = 10.0
    binance_depth_ws_url: str = "wss://stream.binance.com:9443/ws/btcusdt@depth20@100ms"
    coinbase_depth_ws_url: str = "wss://advanced-trade-ws.coinbase.com"
    kraken_depth_ws_url: str = "wss://ws.kraken.com/v2"
    strategy: KalshiStrategyConfig = field(default_factory=KalshiStrategyConfig)
    credentials: KalshiCredentials = field(default_factory=lambda: KalshiCredentials.from_env(os.environ))

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> "KalshiRuntimeConfig":
        environment = env.get("KALSHI_ENV", "production").lower()
        demo = environment == "demo"
        binance_ws_url = env.get(
            "KALSHI_BINANCE_WS_URL",
            "wss://stream.binance.com:9443/ws/btcusdt@trade",
        )
        # BRTI and the other reference venues are USD-quoted. Binance.US lists a
        # real BTCUSD pair; Binance.com only lists BTCUSDT, whose ~10bp stablecoin
        # basis contaminates cross-venue features — prefer the USD pair on .us.
        default_binance_depth_url = (
            "wss://stream.binance.us:9443/ws/btcusd@depth20@100ms"
            if "stream.binance.us" in binance_ws_url
            else "wss://stream.binance.com:9443/ws/btcusdt@depth20@100ms"
        )
        strategy = KalshiStrategyConfig(
            base_count=float(env.get("KALSHI_ORDER_COUNT", "5")),
            max_abs_position=float(env.get("KALSHI_MAX_ABS_POSITION", "50")),
            max_open_count=float(env.get("KALSHI_MAX_OPEN_COUNT", "20")),
            min_edge_dollars=float(env.get("KALSHI_MIN_EDGE_DOLLARS", "0.01")),
            inventory_skew_dollars=float(env.get("KALSHI_INVENTORY_SKEW_DOLLARS", "0.02")),
            stale_after_ms=int(env.get("KALSHI_STALE_AFTER_MS", "2000")),
            stop_quote_before_close_s=float(env.get("KALSHI_CLOSE_CUTOFF_S", "90")),
            maker_fee_multiplier=float(env.get("KALSHI_MAKER_FEE_MULTIPLIER", "1")),
            max_quote_notional_dollars=float(env.get("KALSHI_MAX_QUOTE_NOTIONAL_DOLLARS", "10")),
            cash_reserve_dollars=float(env.get("KALSHI_CASH_RESERVE_DOLLARS", "10")),
            min_order_count=float(env.get("KALSHI_MIN_ORDER_COUNT", "1")),
            requote_remaining_fraction=float(env.get("KALSHI_REQUOTE_REMAINING_FRACTION", "0.5")),
        )
        return cls(
            environment=environment,
            rest_url=env.get("KALSHI_REST_URL", DEMO_REST_URL if demo else PROD_REST_URL),
            ws_url=env.get("KALSHI_WS_URL", DEMO_WS_URL if demo else PROD_WS_URL),
            data_dir=env.get("KALSHI_DATA_DIR", "data/kalshi"),
            subaccount=int(env.get("KALSHI_SUBACCOUNT", "0")),
            position_poll_s=float(env.get("KALSHI_POSITION_POLL_S", "1.0")),
            order_reconcile_s=float(env.get("KALSHI_ORDER_RECONCILE_S", "1.0")),
            quote_throttle_s=float(env.get("KALSHI_QUOTE_THROTTLE_S", "0.25")),
            allow_proxy_pricing=_bool(env, "KALSHI_ALLOW_PROXY_PRICING", False),
            ws_insecure=_bool(env, "KALSHI_WS_INSECURE", _bool(env, "WS_INSECURE", False)),
            binance_ws_url=binance_ws_url,
            external_depth_enabled=_bool(env, "KALSHI_EXTERNAL_DEPTH_ENABLED", True),
            external_depth_levels=int(env.get("KALSHI_EXTERNAL_DEPTH_LEVELS", "10")),
            external_depth_sample_hz=float(env.get("KALSHI_EXTERNAL_DEPTH_SAMPLE_HZ", "10")),
            binance_depth_ws_url=env.get(
                "KALSHI_BINANCE_DEPTH_WS_URL",
                default_binance_depth_url,
            ),
            coinbase_depth_ws_url=env.get(
                "KALSHI_COINBASE_DEPTH_WS_URL",
                "wss://advanced-trade-ws.coinbase.com",
            ),
            kraken_depth_ws_url=env.get(
                "KALSHI_KRAKEN_DEPTH_WS_URL",
                "wss://ws.kraken.com/v2",
            ),
            strategy=strategy,
            credentials=KalshiCredentials.from_env(env),
        )
