"""Generic append-only JSONL corpus sink, any pydantic BaseModel payload.

Originally written for InnerStateFeaturesV1 only (Plan 2 training data);
reused as-is (2026-07-13) for MoodArcCorpusRowV1 -- append() only ever
calls payload.model_dump(mode="json"), so it was already schema-agnostic
in practice. Type hint widened to match.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class InnerStateCorpusSink:
    def __init__(self, path: str) -> None:
        self._path: Optional[Path] = Path(path) if path else None

    @property
    def enabled(self) -> bool:
        return self._path is not None

    def append(self, payload: BaseModel) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload.model_dump(mode="json"), separators=(",", ":"))
        # flush (not fsync) per tick: this is re-derivable training data, and the
        # self-state tick runs ~every 2s — a per-line fsync is needless I/O.
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
