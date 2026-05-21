"""Detect placeholder stub adapter emissions (spec §5.9)."""
from __future__ import annotations

from typing import Dict, FrozenSet

from orion.signals.models import OrionSignalV1
from orion.signals.registry import ORGAN_REGISTRY

_GENERIC_PLACEHOLDER_DIMS: FrozenSet[str] = frozenset({"level", "confidence"})

# Organs whose gateway adapters still emit placeholder signals (excludes biometrics).
_STUB_ORGAN_IDS: FrozenSet[str] = frozenset(
    {
        "collapse_mirror",
        "equilibrium",
        "recall",
        "spark",
        "autonomy",
        "world_pulse",
        "social_memory",
        "social_room_bridge",
        "vision",
        "agent_chain",
        "planner",
        "dream",
        "state_journaler",
        "topic_foundry",
        "concept_induction",
        "graph_cognition",
        "chat_stance",
        "journaler",
        "power_guard",
        "security_watcher",
    }
)


def _is_placeholder_dimensions(dimensions: Dict[str, float]) -> bool:
    if not dimensions:
        return False
    keys = set(dimensions.keys())
    if keys != _GENERIC_PLACEHOLDER_DIMS:
        return False
    return dimensions.get("level") == 0.5 and dimensions.get("confidence") == 0.5


def is_stub_signal(sig: OrionSignalV1) -> bool:
    """Return True when a signal matches Organ Signals stub exclusion rules (§5.9)."""
    for note in sig.notes or []:
        if "stub adapter" in str(note).lower():
            return True
    if sig.summary and "stub adapter" in str(sig.summary).lower():
        return True
    if _is_placeholder_dimensions(sig.dimensions):
        return True
    if sig.organ_id in _STUB_ORGAN_IDS and not sig.source_event_id:
        entry = ORGAN_REGISTRY.get(sig.organ_id)
        if entry and sig.signal_kind in (entry.signal_kinds or []):
            return True
    return False
