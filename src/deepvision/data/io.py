"""Small helpers shared by the pipeline steps: file scanning, resumable JSONL caches, logging."""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Iterable, Iterator

from .config import DataConfig, Source


def setup_logging(log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger("deepvision")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def _ignored(rel: str, patterns: Iterable[str]) -> bool:
    rel = "/" + rel
    return any(p in rel for p in patterns)


def iter_source_files(cfg: DataConfig, source: Source) -> Iterator[tuple[str, int, float]]:
    """Yield (path relative to raw_root in POSIX form, size in bytes, mtime) for every video of a source."""
    base = cfg.raw_root / source.dir
    if not base.is_dir():
        return
    stack = [base]
    while stack:
        current = stack.pop()
        with os.scandir(current) as it:
            for entry in it:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                    continue
                if os.path.splitext(entry.name)[1].lower() not in cfg.video_extensions:
                    continue
                rel = Path(entry.path).relative_to(cfg.raw_root).as_posix()
                if _ignored(rel, cfg.ignore_patterns):
                    continue
                st = entry.stat()
                yield rel, st.st_size, st.st_mtime


class JsonlCache:
    """Append-only JSONL file keyed by a string field; lets long runs resume after interruption."""

    def __init__(self, path: Path, key: str = "path"):
        self.path = path
        self.key = key
        self.records: dict[str, dict] = {}
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # a partially written last line after a crash
                    self.records[rec[self.key]] = rec
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")

    def __contains__(self, k: str) -> bool:
        return k in self.records

    def get(self, k: str) -> dict | None:
        return self.records.get(k)

    def add(self, rec: dict) -> None:
        self.records[rec[self.key]] = rec
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()
