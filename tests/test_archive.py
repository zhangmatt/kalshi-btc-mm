import gzip
import json
from pathlib import Path

import pytest

from kalshi_mm.archive import (
    ArchiveConfig,
    apply_retention,
    marker_path,
    object_key,
    recording_is_ready,
    upload_recording,
)


class FakeS3:
    def __init__(self, *, corrupt_head: bool = False):
        self.objects = {}
        self.corrupt_head = corrupt_head

    def put_object(self, *, Bucket, Key, Body, ContentLength, ContentType, ServerSideEncryption, ChecksumSHA256, Metadata):
        body = Body.read()
        assert len(body) == ContentLength
        self.objects[(Bucket, Key)] = {
            "body": body,
            "metadata": Metadata,
            "encryption": ServerSideEncryption,
            "checksum": ChecksumSHA256,
        }

    def head_object(self, *, Bucket, Key, ChecksumMode):
        assert ChecksumMode == "ENABLED"
        row = self.objects[(Bucket, Key)]
        return {
            "ContentLength": len(row["body"]) + int(self.corrupt_head),
            "Metadata": row["metadata"],
            "ChecksumSHA256": row["checksum"],
        }


def _recording(path: Path, *event_types: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for event_type in event_types:
            handle.write(json.dumps({"event_type": event_type}) + "\n")


def _config(tmp_path: Path, **changes) -> ArchiveConfig:
    values = {
        "data_dir": tmp_path,
        "bucket": "research-bucket",
        "stable_age_s": 0,
        "settlement_grace_s": 900,
        "reserve_free_gb": 0,
        "target_free_gb": 0,
    }
    values.update(changes)
    return ArchiveConfig(**values)


def test_only_closed_finalized_recordings_are_immediately_ready(tmp_path):
    active = tmp_path / "ACTIVE" / "events_active.jsonl.gz"
    closed = tmp_path / "CLOSED" / "events_closed.jsonl.gz"
    final = tmp_path / "FINAL" / "events_final.jsonl.gz"
    _recording(active, "orderbook_delta")
    _recording(closed, "market_status_at_close")
    _recording(final, "market_status_at_close", "market_result")
    config = _config(tmp_path)
    now = max(path.stat().st_mtime for path in (active, closed, final)) + 1
    assert not recording_is_ready(active, config, now)
    assert not recording_is_ready(closed, config, now)
    assert recording_is_ready(final, config, now)


def test_closed_recording_uses_settlement_grace_fallback(tmp_path):
    path = tmp_path / "TICKER" / "events_run.jsonl.gz"
    _recording(path, "market_status_at_close")
    config = _config(tmp_path, settlement_grace_s=60)
    assert recording_is_ready(path, config, path.stat().st_mtime + 61)


def test_upload_is_encrypted_verified_and_marked(tmp_path):
    path = tmp_path / "TICKER" / "events_run.jsonl.gz"
    _recording(path, "market_status_at_close", "market_result")
    config = _config(tmp_path)
    s3 = FakeS3()
    marker = upload_recording(path, config, s3)
    remote = s3.objects[(config.bucket, object_key(path, config))]
    assert remote["body"] == path.read_bytes()
    assert remote["encryption"] == "AES256"
    assert remote["metadata"]["sha256"] == marker.sha256
    assert marker_path(path).exists()


def test_failed_remote_verification_never_writes_marker(tmp_path):
    path = tmp_path / "TICKER" / "events_run.jsonl.gz"
    _recording(path, "market_status_at_close", "market_result")
    with pytest.raises(RuntimeError, match="verification failed"):
        upload_recording(path, _config(tmp_path), FakeS3(corrupt_head=True))
    assert not marker_path(path).exists()


def test_retention_deletes_only_verified_local_copy(tmp_path):
    uploaded = tmp_path / "UPLOADED" / "events_old.jsonl.gz"
    unuploaded = tmp_path / "LOCAL" / "events_old.jsonl.gz"
    _recording(uploaded, "market_status_at_close", "market_result")
    _recording(unuploaded, "market_status_at_close", "market_result")
    config = _config(tmp_path, retention_days=1)
    s3 = FakeS3()
    marker = upload_recording(uploaded, config, s3)
    deleted, freed = apply_retention(config, s3, now=marker.uploaded_at + 86_401)
    assert deleted == 1
    assert freed > 0
    assert not uploaded.exists()
    assert unuploaded.exists()


def test_retention_preserves_local_copy_when_remote_verification_fails(tmp_path):
    path = tmp_path / "UPLOADED" / "events_old.jsonl.gz"
    _recording(path, "market_status_at_close", "market_result")
    config = _config(tmp_path, retention_days=1)
    good_s3 = FakeS3()
    marker = upload_recording(path, config, good_s3)
    bad_s3 = FakeS3()
    deleted, freed = apply_retention(config, bad_s3, now=marker.uploaded_at + 86_401)
    assert deleted == 0
    assert freed == 0
    assert path.exists()
    assert marker_path(path).exists()


def test_runner_recordings_are_ready_once_finalized(tmp_path):
    # Runner files never contain market_status_at_close.
    path = tmp_path / "TICKER" / "runner_x.jsonl.gz"
    _recording(path, "market_result")
    config = _config(tmp_path)
    assert recording_is_ready(path, config, path.stat().st_mtime + 1)
    unfinalized = tmp_path / "TICKER" / "runner_y.jsonl.gz"
    _recording(unfinalized, "market_metadata")
    assert not recording_is_ready(unfinalized, config, unfinalized.stat().st_mtime + 1)


class FailingS3(FakeS3):
    def __init__(self, fail_key_fragment: str):
        super().__init__()
        self.fail_key_fragment = fail_key_fragment

    def put_object(self, *, Bucket, Key, **kwargs):
        if self.fail_key_fragment in Key:
            raise RuntimeError("simulated S3 outage for this key")
        super().put_object(Bucket=Bucket, Key=Key, **kwargs)


def test_upload_failure_does_not_block_later_files(tmp_path):
    import os

    from kalshi_mm.archive import archive_ready

    first = tmp_path / "A" / "events_a.jsonl.gz"
    second = tmp_path / "B" / "events_b.jsonl.gz"
    _recording(first, "market_status_at_close", "market_result")
    _recording(second, "market_status_at_close", "market_result")
    os.utime(first, ns=(1, 1))  # force first in mtime order
    config = _config(tmp_path)
    s3 = FailingS3("events_a")
    uploaded, skipped, failed = archive_ready(config, s3, now=second.stat().st_mtime + 1)
    assert failed == 1
    assert uploaded == 1
    assert marker_path(second).exists()
    assert not marker_path(first).exists()
