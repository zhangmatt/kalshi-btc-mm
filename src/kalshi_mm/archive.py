from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class ArchiveConfig:
    data_dir: Path
    bucket: str
    prefix: str = "kalshi-btc15m"
    region: str = "us-east-1"
    retention_days: float = 7.0
    stable_age_s: float = 60.0
    settlement_grace_s: float = 900.0
    reserve_free_gb: float = 10.0
    target_free_gb: float = 15.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> "ArchiveConfig":
        return cls(
            data_dir=Path(env.get("KALSHI_DATA_DIR", "data/kalshi")).expanduser(),
            bucket=env.get("KALSHI_S3_BUCKET", "").strip(),
            prefix=env.get("KALSHI_S3_PREFIX", "kalshi-btc15m").strip("/"),
            region=env.get("AWS_REGION", env.get("AWS_DEFAULT_REGION", "us-east-1")),
            retention_days=float(env.get("KALSHI_LOCAL_RETENTION_DAYS", "7")),
            stable_age_s=float(env.get("KALSHI_ARCHIVE_STABLE_AGE_S", "60")),
            settlement_grace_s=float(env.get("KALSHI_SETTLEMENT_GRACE_S", "900")),
            reserve_free_gb=float(env.get("KALSHI_RESERVE_FREE_GB", "10")),
            target_free_gb=float(env.get("KALSHI_TARGET_FREE_GB", "15")),
        )

    def validate(self) -> None:
        if not self.bucket:
            raise RuntimeError("KALSHI_S3_BUCKET is not configured")
        if self.retention_days < 0:
            raise ValueError("KALSHI_LOCAL_RETENTION_DAYS cannot be negative")
        if self.stable_age_s < 0 or self.settlement_grace_s < 0:
            raise ValueError("archive timing values cannot be negative")
        if self.reserve_free_gb < 0 or self.target_free_gb < self.reserve_free_gb:
            raise ValueError("target free space must be at least the reserve")


@dataclass(frozen=True)
class RecordingState:
    closed: bool
    finalized: bool
    readable: bool


@dataclass(frozen=True)
class UploadMarker:
    bucket: str
    key: str
    size: int
    sha256: str
    mtime_ns: int
    uploaded_at: float

    @classmethod
    def from_path(cls, path: Path) -> "UploadMarker":
        row = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            bucket=str(row["bucket"]),
            key=str(row["key"]),
            size=int(row["size"]),
            sha256=str(row["sha256"]),
            mtime_ns=int(row["mtime_ns"]),
            uploaded_at=float(row["uploaded_at"]),
        )


def marker_path(recording: Path) -> Path:
    return recording.with_name(recording.name + ".uploaded.json")


def inspect_recording(path: Path) -> RecordingState:
    closed = False
    finalized = False
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                event_type = str(json.loads(line).get("event_type", ""))
                closed = closed or event_type == "market_status_at_close"
                finalized = finalized or event_type in {"market_result", "market_result_error"}
    except (EOFError, OSError, json.JSONDecodeError, UnicodeDecodeError):
        return RecordingState(closed=closed, finalized=finalized, readable=False)
    return RecordingState(closed=closed, finalized=finalized, readable=True)


def recording_is_ready(path: Path, config: ArchiveConfig, now: Optional[float] = None) -> bool:
    now = time.time() if now is None else now
    age = now - path.stat().st_mtime
    if age < config.stable_age_s:
        return False
    state = inspect_recording(path)
    if not state.readable:
        return False
    # Runner recordings never contain market_status_at_close; the collector's
    # events_* files must, or they are still being written.
    if path.name.startswith("events_") and not state.closed:
        return False
    return state.finalized or age >= config.settlement_grace_s


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_key(path: Path, config: ArchiveConfig) -> str:
    relative = path.resolve().relative_to(config.data_dir.resolve())
    return "/".join(value for value in (config.prefix, relative.as_posix()) if value)


def _write_marker(path: Path, marker: UploadMarker) -> None:
    payload = {
        "bucket": marker.bucket,
        "key": marker.key,
        "size": marker.size,
        "sha256": marker.sha256,
        "mtime_ns": marker.mtime_ns,
        "uploaded_at": marker.uploaded_at,
        "uploaded_at_iso": datetime.fromtimestamp(marker.uploaded_at, timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, separators=(",", ":"), sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def _verified_marker(path: Path, config: ArchiveConfig) -> Optional[UploadMarker]:
    candidate = marker_path(path)
    if not candidate.exists():
        return None
    try:
        marker = UploadMarker.from_path(candidate)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None
    if marker.bucket != config.bucket or marker.key != object_key(path, config):
        return None
    stat = path.stat()
    if marker.size != stat.st_size or marker.mtime_ns != stat.st_mtime_ns:
        return None
    return marker


def _remote_marker_is_valid(marker: UploadMarker, config: ArchiveConfig, s3: Any) -> bool:
    expected_checksum = base64.b64encode(bytes.fromhex(marker.sha256)).decode("ascii")
    try:
        head = s3.head_object(Bucket=config.bucket, Key=marker.key, ChecksumMode="ENABLED")
    except Exception as exc:
        print(f"[archive] remote verification error key={marker.key}: {exc}", flush=True)
        return False
    metadata = {str(k).lower(): str(v) for k, v in (head.get("Metadata") or {}).items()}
    return (
        int(head.get("ContentLength", -1)) == marker.size
        and metadata.get("sha256") == marker.sha256
        and str(head.get("ChecksumSHA256") or "") == expected_checksum
    )


def upload_recording(path: Path, config: ArchiveConfig, s3: Any) -> UploadMarker:
    before = path.stat()
    size = before.st_size
    digest = sha256_file(path)
    after_hash = path.stat()
    if after_hash.st_size != size or after_hash.st_mtime_ns != before.st_mtime_ns:
        raise RuntimeError(f"recording changed while hashing: {path}")
    checksum = base64.b64encode(bytes.fromhex(digest)).decode("ascii")
    key = object_key(path, config)
    with path.open("rb") as handle:
        s3.put_object(
            Bucket=config.bucket,
            Key=key,
            Body=handle,
            ContentLength=size,
            ContentType="application/gzip",
            ServerSideEncryption="AES256",
            ChecksumSHA256=checksum,
            Metadata={"sha256": digest, "source-size": str(size)},
        )
    head = s3.head_object(Bucket=config.bucket, Key=key, ChecksumMode="ENABLED")
    after_upload = path.stat()
    if after_upload.st_size != size or after_upload.st_mtime_ns != before.st_mtime_ns:
        raise RuntimeError(f"recording changed while uploading: {path}")
    remote_size = int(head.get("ContentLength", -1))
    metadata = {str(k).lower(): str(v) for k, v in (head.get("Metadata") or {}).items()}
    remote_checksum = str(head.get("ChecksumSHA256") or "")
    if remote_size != size or metadata.get("sha256") != digest or remote_checksum != checksum:
        raise RuntimeError(
            f"S3 verification failed for s3://{config.bucket}/{key}: "
            f"size={remote_size}/{size} sha256={metadata.get('sha256')}/{digest} "
            f"checksum={remote_checksum}/{checksum}"
        )
    marker = UploadMarker(config.bucket, key, size, digest, before.st_mtime_ns, time.time())
    _write_marker(marker_path(path), marker)
    return marker


def recordings(data_dir: Path) -> list[Path]:
    paths = list(data_dir.glob("**/events_*.jsonl.gz")) + list(data_dir.glob("**/runner_*.jsonl.gz"))
    return sorted(paths, key=lambda value: value.stat().st_mtime)


def archive_ready(config: ArchiveConfig, s3: Any, *, now: Optional[float] = None) -> tuple[int, int, int]:
    now = time.time() if now is None else now
    uploaded = 0
    skipped = 0
    failed = 0
    for path in recordings(config.data_dir):
        if _verified_marker(path, config):
            skipped += 1
            continue
        if not recording_is_ready(path, config, now):
            continue
        try:
            marker = upload_recording(path, config, s3)
        except Exception as exc:
            # One bad file must not block every later file; the health check
            # alarms on any persistent backlog.
            failed += 1
            print(f"[archive] upload failed for {path}: {exc}", flush=True)
            continue
        uploaded += 1
        print(
            f"[archive] uploaded bytes={marker.size} sha256={marker.sha256[:12]} "
            f"s3://{marker.bucket}/{marker.key}",
            flush=True,
        )
    return uploaded, skipped, failed


def _delete_local(path: Path, marker: UploadMarker, *, reason: str) -> int:
    size = path.stat().st_size
    path.unlink()
    marker_path(path).unlink(missing_ok=True)
    print(f"[archive] deleted local bytes={size} reason={reason} key={marker.key}", flush=True)
    return size


def apply_retention(config: ArchiveConfig, s3: Any, *, now: Optional[float] = None) -> tuple[int, int]:
    now = time.time() if now is None else now
    deleted = 0
    freed = 0
    uploaded_paths: list[tuple[Path, UploadMarker]] = []
    for path in recordings(config.data_dir):
        marker = _verified_marker(path, config)
        if marker:
            uploaded_paths.append((path, marker))
    cutoff = now - config.retention_days * 86_400
    for path, marker in uploaded_paths:
        if marker.uploaded_at <= cutoff and _remote_marker_is_valid(marker, config, s3):
            freed += _delete_local(path, marker, reason="retention")
            deleted += 1

    usage = shutil.disk_usage(config.data_dir)
    reserve = int(config.reserve_free_gb * 1024**3)
    target = int(config.target_free_gb * 1024**3)
    if usage.free < reserve:
        remaining = [
            (path, marker)
            for path, marker in uploaded_paths
            if path.exists()
        ]
        for path, marker in remaining:
            if not _remote_marker_is_valid(marker, config, s3):
                continue
            freed += _delete_local(path, marker, reason="disk_pressure")
            deleted += 1
            if shutil.disk_usage(config.data_dir).free >= target:
                break
    return deleted, freed


def health_check(config: ArchiveConfig, s3: Any, *, now: Optional[float] = None) -> None:
    now = time.time() if now is None else now
    config.validate()
    s3.head_bucket(Bucket=config.bucket)
    ready_count = 0
    pending: list[Path] = []
    markers: list[UploadMarker] = []
    for path in recordings(config.data_dir):
        marker = _verified_marker(path, config)
        if marker:
            markers.append(marker)
            ready_count += 1
        elif recording_is_ready(path, config, now):
            ready_count += 1
            pending.append(path)
    if pending:
        oldest_age = now - min(path.stat().st_mtime for path in pending)
        if oldest_age > max(config.settlement_grace_s, 1_800):
            raise RuntimeError(f"{len(pending)} archival file(s) pending; oldest_age_s={oldest_age:.0f}")
    if markers:
        newest = max(markers, key=lambda marker: marker.uploaded_at)
        if not _remote_marker_is_valid(newest, config, s3):
            raise RuntimeError(f"newest archived object failed remote verification: {newest.key}")
    usage = shutil.disk_usage(config.data_dir)
    if usage.free < int(config.reserve_free_gb * 1024**3):
        raise RuntimeError(f"free disk below reserve: {usage.free / 1024**3:.2f} GiB")
    print(
        f"[archive] healthy ready={ready_count} pending={len(pending)} "
        f"free_gib={usage.free / 1024**3:.2f} bucket={config.bucket}",
        flush=True,
    )


def _client(config: ArchiveConfig) -> Any:
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - packaging failure
        raise RuntimeError("boto3 is required for S3 archival") from exc
    return boto3.client("s3", region_name=config.region)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Archive finalized Kalshi recordings to verified S3 storage.")
    parser.add_argument("--health", action="store_true")
    args = parser.parse_args(argv)
    config = ArchiveConfig.from_env()
    config.validate()
    s3 = _client(config)
    if args.health:
        health_check(config, s3)
        return
    uploaded, skipped, failed = archive_ready(config, s3)
    deleted, freed = apply_retention(config, s3)
    print(
        f"[archive] complete uploaded={uploaded} already_uploaded={skipped} failed={failed} "
        f"deleted={deleted} freed_bytes={freed}",
        flush=True,
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
