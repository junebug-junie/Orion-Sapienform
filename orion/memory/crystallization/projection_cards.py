"""Projection: MemoryCrystallizationV1 -> MemoryCardV1 recall surface.

Projection rules (spec section 6.3 / 19.1):
- only active crystallizations project to active recall cards
- rejected/quarantined/proposed crystallizations must not project
- superseded crystallizations may project only with an explicit superseded marker
- projected cards carry subschema.crystallization_ref and can be rebuilt
"""

from __future__ import annotations

from datetime import datetime, timezone

from orion.core.contracts.memory_cards import MemoryCardCreateV1
from orion.schemas.memory_crystallization import MemoryCrystallizationV1

PROJECTABLE_STATUSES = frozenset({"active", "superseded"})

_KIND_PRIORITY: dict[str, str] = {
    "stance": "high_recall",
    "decision": "high_recall",
    "procedure": "high_recall",
    "contradiction": "high_recall",
    "failure_mode": "high_recall",
    "open_loop": "episodic_detail",
    "attractor": "episodic_detail",
    "semantic": "episodic_detail",
    "episode": "episodic_detail",
}


class ProjectionNotAllowed(Exception):
    """Raised when a crystallization's status forbids card projection."""


def crystallization_to_card(crystallization: MemoryCrystallizationV1) -> MemoryCardCreateV1:
    """Build a MemoryCardV1 create payload projecting this crystallization."""
    if crystallization.status not in PROJECTABLE_STATUSES:
        raise ProjectionNotAllowed(
            f"status {crystallization.status!r} must not project to recall cards"
        )

    superseded = crystallization.status == "superseded"
    title = crystallization.subject
    summary = crystallization.summary
    if superseded:
        title = f"[superseded] {title}"
        summary = f"SUPERSEDED — retained for history. {summary}"

    project = next(
        (s.split(":", 1)[1] for s in crystallization.scope if s.startswith("project:")),
        None,
    )

    return MemoryCardCreateV1(
        types=["crystallization", crystallization.kind],
        status="active" if not superseded else "superseded",
        confidence=crystallization.confidence,
        sensitivity=crystallization.governance.sensitivity,
        priority=_KIND_PRIORITY.get(crystallization.kind, "episodic_detail"),
        provenance="operator_distiller",
        project=project,
        title=title,
        summary=summary,
        tags=list(crystallization.tags) or None,
        subschema={
            "crystallization_ref": {
                "schema_version": "crystallization_ref.v1",
                "crystallization_id": crystallization.crystallization_id,
                "kind": crystallization.kind,
                "projection_role": "recall_surface",
                "superseded": superseded,
                "synced_at": datetime.now(timezone.utc).isoformat(),
            }
        },
    )
