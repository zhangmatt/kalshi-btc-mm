from __future__ import annotations

import gzip
import json
import queue
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Optional, TextIO


class EventRecorder:
    """Asynchronous append-only recorder for deterministic market replay."""

    _STOP = object()

    def __init__(self, path: str | Path, *, flush_every: int = 100):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_every = max(1, flush_every)
        self._lines = 0
        self._queue: queue.Queue[object] = queue.Queue()
        self._closed = threading.Event()
        self._error: Optional[BaseException] = None
        if self.path.suffix == ".gz":
            self._file: TextIO = gzip.open(self.path, "at", encoding="utf-8")
        else:
            self._file = self.path.open("a", encoding="utf-8")
        self._thread = threading.Thread(target=self._run, name="event-recorder", daemon=True)
        self._thread.start()

    def write(
        self,
        *,
        source: str,
        event_type: str,
        payload: Mapping[str, Any],
        exchange_ts_ms: Optional[int] = None,
        receive_ts_ms: Optional[int] = None,
    ) -> None:
        if self._closed.is_set():
            raise RuntimeError("event recorder is closed")
        if self._error is not None:
            raise RuntimeError("event recorder writer failed") from self._error
        self._queue.put({
            "schema_version": 1,
            "receive_ts_ms": receive_ts_ms or time.time_ns() // 1_000_000,
            "exchange_ts_ms": exchange_ts_ms,
            "source": source,
            "event_type": event_type,
            "payload": dict(payload),
        })

    def _run(self) -> None:
        try:
            while True:
                item = self._queue.get()
                if item is self._STOP:
                    break
                encoded = json.dumps(item, separators=(",", ":"), sort_keys=True)
                self._file.write(encoded + "\n")
                self._lines += 1
                if self._lines % self.flush_every == 0:
                    self._file.flush()
        except BaseException as exc:  # pragma: no cover - filesystem failure
            self._error = exc
        finally:
            self._file.flush()
            self._file.close()

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._queue.put(self._STOP)
        self._thread.join()
        if self._error is not None:
            raise RuntimeError("event recorder writer failed") from self._error

    def check_health(self, *, max_queue_depth: int = 100_000) -> None:
        """Fail the owning process instead of silently running without durable data."""
        if self._error is not None:
            raise RuntimeError("event recorder writer failed") from self._error
        if not self._thread.is_alive() and not self._closed.is_set():
            raise RuntimeError("event recorder writer stopped unexpectedly")
        depth = self._queue.qsize()
        if depth > max_queue_depth:
            raise RuntimeError(f"event recorder is falling behind: queue_depth={depth}")

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    def __enter__(self) -> "EventRecorder":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def iter_events(path: str | Path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
