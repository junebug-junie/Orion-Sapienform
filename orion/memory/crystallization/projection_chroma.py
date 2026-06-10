"""Projection: MemoryCrystallizationV1 -> Chroma via the existing vector path.

Emits VectorDocumentUpsertV1 payloads for the memory.vector.upsert.v1
envelope kind on orion:memory:vector:upsert (consumed by orion-vector-writer).
Chroma is a rebuildable semantic recall projection; Postgres remains truth.
"""

from __future__ import annotations

from orion.schemas.memory_crystallization import MemoryCrystallizationV1
from orion.schemas.vector.schemas import VectorDocumentUpsertV1

CRYSTALLIZATION_COLLECTION = "orion_memory_crystallizations"
CRYSTALLIZATION_VECTOR_KIND = "memory.crystallization"
VECTOR_UPSERT_ENVELOPE_KIND = "memory.vector.upsert.v1"
VECTOR_UPSERT_CHANNEL = "orion:memory:vector:upsert"


class ProjectionNotAllowed(Exception):
    """Raised when a crystallization's status forbids Chroma projection."""


def crystallization_doc_text(crystallization: MemoryCrystallizationV1) -> str:
    return f"[{crystallization.kind}] {crystallization.subject} — {crystallization.summary}"


def crystallization_to_vector_upsert(
    crystallization: MemoryCrystallizationV1,
    *,
    collection: str = CRYSTALLIZATION_COLLECTION,
    embedding: list[float] | None = None,
    embedding_model: str | None = None,
) -> VectorDocumentUpsertV1:
    """Build the vector upsert payload for an active crystallization."""
    if crystallization.status != "active":
        raise ProjectionNotAllowed(
            f"status {crystallization.status!r} must not project to Chroma"
        )

    return VectorDocumentUpsertV1(
        doc_id=crystallization.crystallization_id,
        kind=CRYSTALLIZATION_VECTOR_KIND,
        text=crystallization_doc_text(crystallization),
        collection=collection,
        metadata={
            "crystallization_id": crystallization.crystallization_id,
            "kind": crystallization.kind,
            "status": crystallization.status,
            "scope": ",".join(crystallization.scope),
            "salience": crystallization.salience,
            "confidence": crystallization.confidence,
        },
        embedding=embedding,
        embedding_model=embedding_model,
        embedding_dim=len(embedding) if embedding else None,
    )
