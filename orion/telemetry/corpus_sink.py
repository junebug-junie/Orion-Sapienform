"""Generic append-only JSONL corpus sink, any pydantic BaseModel payload.

Originally written for InnerStateFeaturesV1 only (Plan 2 training data);
reused as-is (2026-07-13) for MoodArcCorpusRowV1 -- append() only ever
calls payload.model_dump(mode="json"), so it was already schema-agnostic
in practice. Type hint widened to match.

Promoted here (2026-07-13, field-channel-raw-corpus-collector) from
services/orion-spark-introspector/app/inner_state_sink.py: a second
service (orion-field-digester, for field_channel_corpus.v1) now needs this
same sink, and per CLAUDE.md's cross-service seam rule a service must not
import from another service's app/ internals. This class had zero real
spark-introspector-specific coupling (its only import was already
orion.telemetry.corpus_rotation), so the move is a pure relocation, no
behavior change. The old location was deleted outright in the same patch
(exactly one real caller, services/orion-spark-introspector/app/worker.py,
updated to import from here) -- no re-export shim left behind.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from orion.telemetry.corpus_rotation import ROTATED_SUFFIX_RE

logger = logging.getLogger(__name__)


class InnerStateCorpusSink:
    def __init__(self, path: str, *, max_bytes: int = 200_000_000, max_rotated_files: int = 5) -> None:
        self._path: Optional[Path] = Path(path) if path else None
        self._max_bytes = max_bytes
        # Clamp negative values rather than let them reach the deletion
        # slice below: rotated[-1:] on a negative max_rotated_files keeps
        # only the OLDEST rotated file and deletes every newer one -- the
        # exact inverse of this setting's intent (found by code review,
        # 2026-07-13). 0 is a legitimate "no retention, just rotate"
        # choice and is left as-is (rotated[0:] correctly deletes
        # everything, which is unsurprising for that value).
        self._max_rotated_files = max(0, max_rotated_files)

    @property
    def enabled(self) -> bool:
        return self._path is not None

    def append(self, payload: BaseModel) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._rotate_if_needed()
        line = json.dumps(payload.model_dump(mode="json"), separators=(",", ":"))
        # flush (not fsync) per tick: this is re-derivable training data, and the
        # self-state tick runs ~every 2s — a per-line fsync is needless I/O.
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()

    def _rotate_if_needed(self) -> None:
        if self._path is None:
            return
        # Rotation is housekeeping, not the write itself: a transient
        # filesystem hiccup here (ESTALE/EACCES on a degraded network
        # mount -- /mnt/telemetry genuinely is one in production -- not
        # just the common ENOENT pathlib already treats as "doesn't
        # exist") must skip rotation for this tick rather than lose the
        # row entirely by raising past append()'s caller (found by code
        # review, 2026-07-13).
        try:
            self._try_rotate()
        except OSError as exc:
            logger.warning("Corpus sink rotation check failed for %s: %s -- skipping rotation this tick", self._path, exc)

    def _try_rotate(self) -> None:
        assert self._path is not None
        if not self._path.exists():
            return
        if self._path.stat().st_size < self._max_bytes:
            return
        # Microsecond precision (not just seconds) plus an explicit
        # existence-check/counter fallback: Path.rename() onto an existing
        # path silently overwrites it, which for THIS operation means
        # silently destroying a previously-rotated backup -- exactly the
        # data-loss failure mode this whole rotation mechanism exists to
        # prevent for the active file. Two rotations within the same
        # microsecond are not realistically reachable at this cadence, but
        # the counter fallback costs nothing and removes the assumption
        # entirely rather than just making the collision window smaller.
        base_name = f"{self._path.name}.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}"
        rotated_path = self._path.with_name(base_name)
        suffix = 0
        while rotated_path.exists():
            suffix += 1
            rotated_path = self._path.with_name(f"{base_name}.{suffix}")
        self._path.rename(rotated_path)
        logger.warning(
            "Rotated corpus sink %s -> %s (exceeded max_bytes=%d)",
            self._path, rotated_path, self._max_bytes,
        )
        self._prune_old_rotations()

    def _prune_old_rotations(self) -> None:
        if self._path is None:
            return
        prefix_len = len(self._path.name) + 1  # +1 for the separating "."
        # Filter to files whose suffix matches ROTATED_SUFFIX_RE exactly --
        # NOT a bare "{name}.*" glob, which would also match a manually-
        # placed backup, a .gz archive, a stray editor temp file, or
        # anything else sharing the corpus basename prefix (found by code
        # review, 2026-07-13, independently by two review angles). Only
        # delete files this class actually created.
        candidates = [
            p for p in self._path.parent.glob(f"{self._path.name}.*")
            if ROTATED_SUFFIX_RE.match(p.name[prefix_len:])
        ]
        # Sort by filename, not st_mtime: the rotated name already embeds a
        # sortable UTC timestamp (%Y%m%dT%H%M%S.%fZ, optionally a ".N"
        # collision suffix, which still sorts after its un-suffixed sibling
        # since ".1" > "" lexically). mtime is not a reliable ordering
        # signal -- clock skew, NFS-backed volumes, or a backup/restore
        # touching the corpus directory can leave mtimes that don't match
        # real rotation order, which would prune a newer file and keep an
        # older one (found by code review, 2026-07-13).
        rotated = sorted(candidates, key=lambda p: p.name, reverse=True)
        for old in rotated[self._max_rotated_files:]:
            try:
                old.unlink()
                logger.info("Pruned old rotated corpus file: %s", old)
            except OSError as exc:
                logger.warning("Failed to prune rotated corpus file %s: %s", old, exc)
