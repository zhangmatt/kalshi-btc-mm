# Claude handoff — 2026-07-14

This records work completed after the previous collector audit. No credentials or private-key
material are included.

## Collector and market-data pipeline

- The EC2 collector remains a persistent `systemd` service running
  `python -m kalshi_mm.collect --windows 0`.
- Added full L2 feeds for Binance.US, Coinbase, and Kraken alongside Kalshi order-book snapshots,
  deltas, trades, lifecycle data, and CF Benchmarks BRTI.
- External books are reconstructed from snapshots and incremental updates. Feed-specific sequence
  handling records `sequence_gap` events and forces recovery rather than silently using a corrupt
  book.
- The collector audit checks gzip readability, market coverage, Kalshi channels, reference feeds,
  external L2 coverage, and sequence gaps.
- Latest verified live audit at 2026-07-14 08:28 UTC: `PASS`, 178,900 rows, Kalshi snapshot/delta/
  trade/BRTI present, all three external L2 feeds present, and zero detected sequence gaps.
- `kalshi-collector-health.timer` runs every minute. The collector had zero service restarts and no
  errors/exceptions in the prior 30-minute journal check.

Relevant files: `src/kalshi_mm/feeds.py`, `src/kalshi_mm/collect.py`,
`src/kalshi_mm/audit.py`, `src/kalshi_mm/config.py`, and corresponding tests.

## Durable S3 archival

- Added `src/kalshi_mm/archive.py` and the `kalshi-mm-archive` entry point.
- Only stable gzip files containing `market_status_at_close` are eligible. Normally a
  `market_result`/`market_result_error` marker is also required; a 15-minute settlement grace
  fallback prevents permanent blockage if Kalshi delays or omits the result event.
- Active, unreadable, and unclosed recordings are never uploaded or deleted.
- Each S3 upload uses SSE-S3 (`AES256`), an explicit SHA-256 object checksum, SHA-256 metadata, and
  a byte count. A `HeadObject` check validates all three before an atomic local `.uploaded.json`
  marker is written.
- Repeated runs are idempotent. A second live run uploaded zero objects and recognized all 13
  existing markers.
- Local retention is seven days. Disk pressure starts only below 10 GiB free and targets 15 GiB.
  Both deletion paths can delete only marked files and now perform a fresh remote size, metadata
  digest, and S3 checksum verification immediately before deleting the local copy.
- Archive health checks S3 access, delayed pending files, disk reserve, and the newest archived
  object's remote integrity.

Live verification at 2026-07-14 08:30 UTC:

- IAM instance role: `kalshi-btc-mm-archive`; no static `~/.aws/credentials` file.
- Private bucket: `kalshi-btc-mm-i-01522d7f7f34e1885-us-east-1`.
- 13 objects independently re-read with `HeadObject`: 43,705,638 bytes, AES256 encryption,
  zero size/digest failures.
- 17 local recordings, 13 verified upload markers, 55 MiB local data, 34.57 GiB free disk.
- Four unmarked files are active or incomplete and are intentionally retained.
- `kalshi-archive.timer` runs every five minutes; `kalshi-archive-health.timer` runs every fifteen
  minutes. Both are enabled and survive reboot.

The AWS bootstrap is `deploy/bootstrap-aws-archive.sh`. It creates a private, versioned bucket,
blocks public access, requires TLS, creates a prefix-scoped EC2 role, and attaches its instance
profile. Runtime values are in `/home/ubuntu/.config/kalshi/archive.env` on EC2; no AWS keys are
stored there.

## Validation

- Local suite: 40 tests passed.
- Python compile check passed.
- Shell syntax and `systemd-analyze verify` passed. The only verify messages came from unrelated
  OS XFS units using removed accounting options.
- Archive upload, remote checksum verification, idempotent rerun, health service, collector health,
  service persistence, and disk checks all passed against the live EC2 instance.

## Operational state

The collector can now be left unattended. Research recordings stay hot on EC2 for seven days and
are archived continuously to S3. Local deletion is fail-closed: inability to prove the remote copy
is intact preserves the local file. No Glacier/lifecycle transition is configured yet because the
dataset is still being actively analyzed.
