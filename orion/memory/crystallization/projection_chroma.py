from __future__ import annotations

from datetime import datetime, timezone

from orion.memory.crystallization.schemas import MemoryCrystallizationV1
from orion.schemas.vector.schemas import VectorDocumentUpsertV1


DEFAULT_COLLECTION = "orion_memory_crystallizations"


def can_project_to_chroma(crystallization: MemoryCrystallizationV1) -> bool:
    return crystallization.status == "active"


def build_chroma_upsert(
    crystallization: MemoryCrystallizationV1,
    *,
    collection: str = DEFAULT_COLLECTION,
    embedding: list[float] | None = None,
) -> VectorDocumentUpsertV1 | None:
    if not can_project_to_chroma(crystallization):
        return None

    text = f"[{crystallization.kind}] {crystallization.subject} — {crystallization.summary}"
    doc_id = f"crys_{crystallization.crystallization_id}"

    return VectorDocumentUpsertV1(
        doc_id=doc_id,
        kind="memory.crystallization",
        text=text,
        collection=collection,
        embedding=embedding,
        metadata={
            "crystallization_id": crystallization.crystallization_id,
            "kind": crystallization.kind,
            "status": crystallization.status,
            "scope": ",".join(crystallization.scope),
            "salience": crystallization.salience,
            "confidence": crystallization.confidence,
            "synced_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def chroma_bus_envelope_kind() -> str:
    return "memory.vector.upsert.v1"
