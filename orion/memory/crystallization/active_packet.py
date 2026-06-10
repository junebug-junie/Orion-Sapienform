from __future__ import annotations

from typing import Any

from orion.memory.crystallization.schemas import ActiveMemoryPacketV1, MemoryCrystallizationV1

KIND_TO_BUCKET = {
    "stance": "stance",
    "semantic": "project_state",
    "decision": "project_state",
    "procedure": "procedures",
    "open_loop": "open_loops",
    "contradiction": "contradictions",
    "attractor": "attractors",
    "failure_mode": "warnings",
    "episode": "project_state",
}


def _entry(crystallization: MemoryCrystallizationV1) -> dict[str, Any]:
    return {
        "crystallization_id": crystallization.crystallization_id,
        "kind": crystallization.kind,
        "summary": crystallization.summary,
        "planning_effects": list(crystallization.planning_effects),
        "salience": crystallization.salience,
        "confidence": crystallization.confidence,
    }


def build_active_packet(
    *,
    query: str,
    crystallizations: list[MemoryCrystallizationV1],
    card_refs: list[str] | None = None,
    task_type: str | None = None,
    project_id: str | None = None,
    session_id: str | None = None,
) -> ActiveMemoryPacketV1:
    active = [c for c in crystallizations if c.status == "active"]
    active.sort(key=lambda c: c.salience, reverse=True)

    packet = ActiveMemoryPacketV1(
        query=query,
        task_type=task_type,
        project_id=project_id,
        session_id=session_id,
        card_refs=list(card_refs or []),
        crystallization_refs=[c.crystallization_id for c in active],
        retrieval_trace={
            "strategy": "salience_ranked_active_crystallizations",
            "count": len(active),
            "task_type": task_type,
        },
    )

    buckets: dict[str, list[dict[str, Any]]] = {
        "stance": packet.stance,
        "project_state": packet.project_state,
        "procedures": packet.procedures,
        "open_loops": packet.open_loops,
        "contradictions": packet.contradictions,
        "warnings": packet.warnings,
        "attractors": packet.attractors,
    }

    for crys in active:
        bucket_key = KIND_TO_BUCKET.get(crys.kind, "project_state")
        buckets[bucket_key].append(_entry(crys))

    return packet
