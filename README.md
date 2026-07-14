# Kalshi BTC 15-minute market maker

Standalone research and maker-execution system for Kalshi `KXBTC15M` contracts.

The system records Kalshi's sequenced YES order book, BRTI settlement feed, BTC prices, and bounded
external L2 books from Binance, Coinbase, and Kraken. It prices the actual settlement rule, replays
maker quotes with queue-ahead constraints, and keeps live order placement behind an explicit
confirmation flag.

## Model

`KXBTC15M` resolves YES when the final 60-second BRTI average is at least the market's official
initial 60-second average (`floor_strike`). The fair-value model computes correlated arithmetic-
average moments under GBM and moment-matches them to a lognormal distribution. During the final
minute it conditions on Kalshi's published partial average and count.

Volatility is estimated independently from one-second BRTI observations with EWMA and multiple
realized-volatility horizons. Cross-exchange basis and order-book microprice are research candidates,
not assumed alpha.

## Layout

- `exchange.py`: RSA signing, REST V2, tapered tick grid, positions, orders, and L2 book.
- `websocket.py`: authenticated subscriptions, sequence-gap recovery, and reconnects.
- `fair_value.py`, `volatility.py`: settlement pricing and volatility forecasting.
- `strategy.py`: post-only quotes, signed inventory skew, collateral and stale-data gates.
- `collect.py`, `recorder.py`, `feeds.py`: synchronized event collection.
- `replay.py`, `research.py`: queue-aware fills, attribution, and alpha comparisons.
- `runner.py`: paper-by-default cancel/replace loop with guarded live mode.

## Setup

```bash
cd kalshi_btc_mm
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

Set credentials in your shell. Keep the PEM outside this directory.

```bash
export KALSHI_API_KEY_ID="your-key-id"
export KALSHI_PRIVATE_KEY_PATH="$HOME/.config/kalshi/private-key.pem"
```

US-hosted collectors can select the reachable Binance.US book midpoint feed:

```bash
export KALSHI_BINANCE_WS_URL="wss://stream.binance.us:9443/ws/btcusdt@bookTicker"
export KALSHI_BINANCE_DEPTH_WS_URL="wss://stream.binance.us:9443/ws/btcusdt@depth20@100ms"
```

External L2 collection defaults to ten levels sampled at no more than 10 Hz per venue. Configure it
with `KALSHI_EXTERNAL_DEPTH_ENABLED`, `KALSHI_EXTERNAL_DEPTH_LEVELS`, and
`KALSHI_EXTERNAL_DEPTH_SAMPLE_HZ`. Depth is research data and is not used by live quotes by default.

## Collect And Analyze

```bash
kalshi-mm-collect --windows 1 --verbose
kalshi-mm-replay data/kalshi/<ticker>/events_<run>.jsonl.gz
kalshi-mm-research data/kalshi/*/events_*.jsonl.gz
```

Continuous collection rolls to the next market at close and finalizes outcomes asynchronously:

```bash
kalshi-mm-collect --windows 0
```

Finalized gzip recordings can be checksum-verified into private S3 storage while retaining seven
days locally. The uploader never deletes an active or unverified file. See `docs/s3-archive.md` and
the systemd units under `deploy/`.

Paper quoting reads production data but does not place orders:

```bash
kalshi-mm-run --windows 1 --verbose
```

Live order placement requires both `--live` and the explicit acknowledgement:

```bash
kalshi-mm-run --live --confirm-live KXBTC15M --windows 1
```

Do not enable live placement until replay and paper results show positive net markouts after fees,
stable inventory reconciliation, and no unexplained sequence gaps. See `docs/validation.md`.
