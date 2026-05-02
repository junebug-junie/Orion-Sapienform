"""Deterministic ``signal_id`` values for ``OrionSignalV1`` (phase-2: full SHA-256 hex)."""
from __future__ import annotations

import hashlib
from uuid import uuid4


def make_signal_id(organ_id: str, source_event_id: str | None) -> str:
    """Return a stable id from ``organ_id:source_event_id``, else a random 32-hex id."""
    if source_event_id:
        return hashlib.sha256(f"{organ_id}:{source_event_id}".encode()).hexdigest()
    return uuid4().hex
