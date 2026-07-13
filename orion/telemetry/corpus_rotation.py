"""Shared contract for JSONL corpus-sink rotation naming and file resolution.

Single source of truth for the rotated-filename pattern
`InnerStateCorpusSink._try_rotate()` (orion/telemetry/corpus_sink.py,
promoted 2026-07-13 from services/orion-spark-introspector/app/
inner_state_sink.py) produces, and for resolving "all files backing one
corpus path" (the active file plus any rotated siblings) on the read side.
2026-07-13, found by code review: this pattern/resolver was independently
duplicated in three places (the sink itself, scripts/fit_phi_encoder.py,
services/orion-spark-introspector/train/evals/eval_phi_encoder_health.py) --
three copies that had to be kept byte-for-byte in sync with no gate
catching drift. Consolidated here.

Must stay free of any dependency on services/* -- services depend on
orion/, never the reverse (same rule as orion/telemetry/corpus_gate.py).
"""
from __future__ import annotations

import re
from pathlib import Path

ROTATED_SUFFIX_RE = re.compile(r"^\d{8}T\d{6}\.\d{6}Z(\.\d+)?$")


def resolve_rotated_corpus_files(path: Path) -> list[Path]:
    """All files backing one corpus path, oldest first: any rotated
    siblings (matching ROTATED_SUFFIX_RE exactly, not a bare "{name}.*"
    glob -- a manually-placed backup, a .gz archive, or a stray editor
    temp file sharing the corpus basename prefix must never be treated as
    a real rotated file) plus the current active file, if present.

    Rotated filenames sort correctly by name (the timestamp format is
    lexically ordered), so no filesystem-metadata read is needed here,
    only glob + sort.
    """
    files: list[Path] = []
    if path.parent.is_dir():
        prefix_len = len(path.name) + 1  # +1 for the separating "."
        files.extend(
            sorted(
                p for p in path.parent.glob(f"{path.name}.*")
                if ROTATED_SUFFIX_RE.match(p.name[prefix_len:])
            )
        )
    if path.is_file():
        files.append(path)
    return files
