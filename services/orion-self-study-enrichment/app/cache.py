"""Content-hash-keyed disk cache for enrichment results.

Mirrors `graphify-out/cache/semantic/`'s existing pattern: a gitignored
directory tree, keyed by a sha256 hash of the input content, storing one
JSON file per entry. Never committed -- see CLAUDE.md sec 2 ("never commit
local cache files") and the PR #1076 incident referenced there (a prior
graphify cache dir was committed by mistake and had to be reverted).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def content_hash(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_path(cache_dir: str | Path, key: str) -> Path:
    # Two-level fanout (first 2 hex chars) mirrors graphify's cache.py shape
    # so the directory tree doesn't dump thousands of files into one dir.
    return Path(cache_dir) / key[:2] / f"{key}.json"


def read_cached(cache_dir: str | Path, key: str) -> dict[str, Any] | None:
    path = cache_path(cache_dir, key)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_cached(cache_dir: str | Path, key: str, value: dict[str, Any]) -> Path:
    path = cache_path(cache_dir, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2), encoding="utf-8")
    return path


@dataclass(frozen=True)
class CacheLookup:
    key: str
    path: Path
    hit: bool
    value: dict[str, Any] | None
