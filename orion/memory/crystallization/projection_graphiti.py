from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from orion.memory.crystallization.schemas import MemoryCrystallizationV1


@dataclass
class GraphitiProjectionResult:
    episode_ids: list[str] = field(default_factory=list)
    entity_ids: list[str] = field(default_factory=list)
    edge_ids: list[str] = field(default_factory=list)
    synced_at: datetime | None = None
    canonical_mutated: bool = False


class GraphitiAdapter:
    """Additive temporal graph projection — cannot mutate canonical crystallizations."""

    def __init__(self, *, enabled: bool = False, url: str | None = None, falkordb_uri: str | None = None):
        self.enabled = enabled
        self.url = (url or "").strip()
        self.falkordb_uri = (falkordb_uri or "").strip()

    def can_sync(self, crystallization: MemoryCrystallizationV1) -> bool:
        if not self.enabled:
            return False
        return crystallization.status in ("active", "proposed")

    def sync_crystallization(self, crystallization: MemoryCrystallizationV1) -> GraphitiProjectionResult:
        """Project crystallization to Graphiti/FalkorDB without mutating canonical state."""
        if not self.can_sync(crystallization):
            return GraphitiProjectionResult()

        # Stub projection: records deterministic placeholder IDs when enabled.
        # Real Graphiti client wiring is an open loop; adapter never writes canonical state.
        cid = crystallization.crystallization_id
        now = datetime.now(timezone.utc)
        return GraphitiProjectionResult(
            episode_ids=[f"gep_{cid}"],
            entity_ids=[f"gent_{cid}"],
            edge_ids=[f"ged_{cid}"],
            synced_at=now,
            canonical_mutated=False,
        )

    def apply_projection_refs(
        self,
        crystallization: MemoryCrystallizationV1,
        result: GraphitiProjectionResult,
    ) -> MemoryCrystallizationV1:
        """Update projection refs only — never changes summary, claims, or governance."""
        updated = crystallization.model_copy(deep=True)
        updated.projection_refs.graphiti_episode_ids = list(result.episode_ids)
        updated.projection_refs.graphiti_entity_ids = list(result.entity_ids)
        updated.projection_refs.graphiti_edge_ids = list(result.edge_ids)
        updated.projection_refs.synced_at = result.synced_at
        updated.updated_at = datetime.now(timezone.utc)
        return updated

    def neighborhood(self, crystallization_id: str) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "nodes": [], "edges": []}
        return {
            "enabled": True,
            "crystallization_id": crystallization_id,
            "nodes": [],
            "edges": [],
            "note": "graphiti_neighborhood_stub",
        }
