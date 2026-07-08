"""Append-only JSONL corpus sink for InnerStateFeaturesV1 (Plan 2 training data)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from orion.schemas.telemetry.inner_state import InnerStateFeaturesV1


class InnerStateCorpusSink:
    def __init__(self, path: str) -> None:
        self._path: Optional[Path] = Path(path) if path else None

    @property
    def enabled(self) -> bool:
        return self._path is not None

    def append(self, payload: InnerStateFeaturesV1) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload.model_dump(mode="json"), separators=(",", ":"))
        # flush (not fsync) per tick: this is re-derivable training data, and the
        # self-state tick runs ~every 2s — a per-line fsync is needless I/O.
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
