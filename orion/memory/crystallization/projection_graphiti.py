"""Projection: MemoryCrystallizationV1 -> Graphiti/FalkorDB temporal graph.

Graphiti/FalkorDB is an additive temporal graph projection. It is NOT the
existing RDF memory_graph path and must never overload /api/memory/graph/*.

Constraints (spec section 10.4):
- Graphiti output cannot canonize, mutate, or delete crystallizations.
- Sync metadata is recorded in projection_refs only.

The adapter here is transport-agnostic: it builds episode payloads and
returns updated projection refs. The actual FalkorDB/Graphiti client wiring
is behind GRAPHITI_ENABLED in the crystallizer service and may be a no-op
until the backend is deployed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from orion.schemas.memory_crystallization import (
    CrystallizationProjectionRefsV1,
    MemoryCrystallizationV1,
)

PROJECTABLE_STATUSES = frozenset({"active", "proposed"})


class ProjectionNotAllowed(Exception):
    """Raised when a crystallization's status forbids Graphiti projection."""


def build_graphiti_episode(crystallization: MemoryCrystallizationV1) -> dict[str, Any]:
    """Build a Graphiti episode payload. Pre-canonical inputs are labeled."""
    if crystallization.status not in PROJECTABLE_STATUSES:
        raise ProjectionNotAllowed(
            f"status {crystallization.status!r} must not project to Graphiti"
        )
    return {
        "schema_version": "graphiti_episode_projection.v1",
        "episode_name": f"crystallization:{crystallization.crystallization_id}",
        "body": f"[{crystallization.kind}] {crystallization.subject} — {crystallization.summary}",
        "source_kind": "memory_crystallization",
        "crystallization_id": crystallization.crystallization_id,
        "canonical": crystallization.status == "active",
        "labels": {
            "kind": crystallization.kind,
            "status": crystallization.status,
            "scope": list(crystallization.scope),
            "salience": crystallization.salience,
        },
        "links": [
            {
                "target_crystallization_id": link.target_crystallization_id,
                "relation": link.relation,
                "confidence": link.confidence,
            }
            for link in crystallization.links
        ],
    }


def record_graphiti_sync(
    crystallization: MemoryCrystallizationV1,
    *,
    episode_ids: list[str],
    entity_ids: list[str] | None = None,
    edge_ids: list[str] | None = None,
) -> CrystallizationProjectionRefsV1:
    """Return updated projection refs recording a Graphiti sync.

    Pure: the caller (governor/repository path) persists the updated refs.
    The Graphiti adapter never mutates the crystallization itself.
    """
    refs = crystallization.projection_refs
    return refs.model_copy(
        update={
            "graphiti_episode_ids": sorted(set(refs.graphiti_episode_ids) | set(episode_ids)),
            "graphiti_entity_ids": sorted(set(refs.graphiti_entity_ids) | set(entity_ids or [])),
            "graphiti_edge_ids": sorted(set(refs.graphiti_edge_ids) | set(edge_ids or [])),
            "synced_at": datetime.now(timezone.utc),
        }
    )
