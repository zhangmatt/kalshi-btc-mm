# S3 archival

`kalshi-mm-archive` uploads only closed gzip recordings. A file must contain the close marker and
either a settlement result marker or be older than the settlement grace period. Active files are
never uploaded or deleted.

Each upload uses SSE-S3 encryption and an explicit SHA-256 object checksum. The uploader performs
`HeadObject` verification of the byte count, metadata digest, and S3 checksum before writing a local
upload marker. Retention and disk-pressure cleanup can delete only files with a matching marker.

The EC2 instance should use an instance profile rather than static AWS keys. From an authenticated
AWS CloudShell in the same account, run `deploy/bootstrap-aws-archive.sh` with the instance ID. The
script creates a private, versioned bucket and a role restricted to the archive prefix, then attaches
the instance profile.

Runtime configuration belongs in `/home/ubuntu/.config/kalshi/collector.env`:

```text
AWS_REGION=us-east-1
KALSHI_S3_BUCKET=kalshi-btc-mm-<instance-id>-us-east-1
KALSHI_S3_PREFIX=kalshi-btc15m
KALSHI_LOCAL_RETENTION_DAYS=7
KALSHI_RESERVE_FREE_GB=10
KALSHI_TARGET_FREE_GB=15
```

The archive timer runs every five minutes. The health timer checks S3 access, pending finalized
files, and free disk every fifteen minutes. Inspect them with:

```bash
systemctl status kalshi-archive.timer kalshi-archive-health.timer
journalctl -u kalshi-archive.service -n 50 --no-pager
kalshi-mm-archive --health
```

