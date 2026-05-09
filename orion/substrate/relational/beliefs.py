"""Output contracts for the Cognitive Unification Layer read model."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from orion.core.schemas.cognitive_substrate import BaseSubstrateNodeV1


@dataclass(frozen=True)
class AnchorBeliefSliceV1:
    """Categorised substrate nodes for a single anchor (e.g. 'orion', 'relationship')."""

    anchor: str
    concepts: list[BaseSubstrateNodeV1] = field(default_factory=list)
    tensions: list[BaseSubstrateNodeV1] = field(default_factory=list)
    goals: list[BaseSubstrateNodeV1] = field(default_factory=list)
    drives: list[BaseSubstrateNodeV1] = field(default_factory=list)
    snapshots: list[BaseSubstrateNodeV1] = field(default_factory=list)
    events: list[BaseSubstrateNodeV1] = field(default_factory=list)
    degraded: bool = False
    tier_outcomes: list[str] = field(default_factory=list)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class UnifiedRelationalBeliefSetV1:
    """Top-level canonical read model produced by CognitiveUnificationLayer."""

    anchors: dict[str, AnchorBeliefSliceV1]
    generated_at: str = field(default_factory=_utc_now_iso)
    cold_anchors: list[str] = field(default_factory=list)
    degraded_producers: list[str] = field(default_factory=list)
    lineage: list[str] = field(default_factory=list)
