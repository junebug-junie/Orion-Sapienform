"""Shared stub-signal detection for Hub inspect cache and adapter tests (spec §5.9)."""
from __future__ import annotations

from orion.signals.models import OrionSignalV1

_STUB_NOTE_MARKERS = ("stub adapter",)


def is_stub_signal(sig: OrionSignalV1) -> bool:
    """True when the signal is a placeholder stub emission, not real organ truth."""
    for note in sig.notes or []:
        lowered = str(note).lower()
        if any(marker in lowered for marker in _STUB_NOTE_MARKERS):
            return True
    if sig.summary and any(marker in str(sig.summary).lower() for marker in _STUB_NOTE_MARKERS):
        return True
    dims = sig.dimensions or {}
    if dims.keys() == {"level", "confidence"} and dims.get("level") == 0.5 and dims.get("confidence") == 0.5:
        if not sig.source_event_id:
            return True
    return False
