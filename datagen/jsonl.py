"""Streaming JSONL helpers with crash-safe appends."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Iterator


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(records: Iterable[dict], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


class DurableAppender:
    """Appends one JSON record per line and fsyncs after each write, so a crash loses nothing
    that was reported as written."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("a", encoding="utf-8")

    def append(self, record: dict) -> None:
        self._f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._f.flush()
        os.fsync(self._f.fileno())

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def repair_jsonl_tail(path: str | Path) -> int:
    """Drop a trailing partial/corrupt line left by a crash mid-write. Returns bytes removed.

    Only the final line is ever touched: earlier lines were fsynced before the next write began.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return 0
    data = path.read_bytes()
    cut = len(data)
    if not data.endswith(b"\n"):
        cut = data.rfind(b"\n") + 1  # 0 when the file is a single partial line
    else:
        last_start = data.rfind(b"\n", 0, len(data) - 1) + 1
        try:
            json.loads(data[last_start:])
        except json.JSONDecodeError:
            cut = last_start
    if cut == len(data):
        return 0
    with path.open("r+b") as f:
        f.truncate(cut)
    return len(data) - cut
