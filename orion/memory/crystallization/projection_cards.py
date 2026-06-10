from __future__ import annotations

from datetime import datetime, timezone

from orion.core.contracts.memory_cards import MemoryCardCreateV1
from orion.memory.crystallization.schemas import CrystallizationRefV1, MemoryCrystallizationV1


def can_project_to_card(crystallization: MemoryCrystallizationV1) -> bool:
    """Only active crystallizations project to recall cards (superseded with explicit role)."""
    if crystallization.status == "active":
        return True
    if crystallization.status == "superseded":
        return True
    return False


def build_memory_card_projection(crystallization: MemoryCrystallizationV1) -> MemoryCardCreateV1 | None:
    if not can_project_to_card(crystallization):
        return None

    now = datetime.now(timezone.utc)
    role = "recall_surface" if crystallization.status == "active" else "superseded_surface"
    ref = CrystallizationRefV1(
        crystallization_id=crystallization.crystallization_id,
        kind=crystallization.kind,
        projection_role=role,
        synced_at=now,
    )

    priority = "high_recall" if crystallization.salience >= 0.7 else "episodic_detail"
    card_status = "active" if crystallization.status == "active" else "superseded"

    return MemoryCardCreateV1(
        types=["crystallization", crystallization.kind],
        status=card_status,
        confidence=crystallization.confidence,
        sensitivity=crystallization.governance.sensitivity,
        priority=priority,
        visibility_scope=list(crystallization.scope),
        provenance="operator_distiller",
        project=crystallization.scope[0] if crystallization.scope else "orion",
        title=crystallization.subject,
        summary=crystallization.summary,
        evidence=[
            {
                "source": f"{ev.source_kind}:{ev.source_id}",
                "excerpt": ev.excerpt,
            }
            for ev in crystallization.evidence[:5]
        ],
        subschema={"crystallization_ref": ref.model_dump(mode="json")},
    )
