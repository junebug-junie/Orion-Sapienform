"""Causal provenance helpers for OrionSignalV1 (shared by gateway and tests)."""
from __future__ import annotations

from typing import Dict

from orion.signals.models import OrionOrganRegistryEntry, OrionSignalV1


def with_missed_parent_notes(
    signal: OrionSignalV1,
    prior_signals: Dict[str, OrionSignalV1],
    registry: Dict[str, OrionOrganRegistryEntry],
) -> OrionSignalV1:
    """
    If any structural parent organ from the registry is missing from ``prior_signals``,
    append a single audit note (spec §7.B: missed link within the time window).
    """
    entry = registry.get(signal.organ_id)
    if entry is None:
        return signal
    missed = [p for p in (entry.causal_parent_organs or []) if p not in prior_signals]
    if not missed:
        return signal
    msg = (
        "missed causal link: no recent signal within window for parent organ(s): "
        + ", ".join(missed)
    )
    combined = list(signal.notes) + [msg]
    new_notes = combined[-5:]
    return signal.model_copy(update={"notes": new_notes})
