from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from orion.memory.crystallization.dynamics import decayed_activation
from orion.memory.crystallization.recall_eligibility import eligible_for_recall
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


def _task_boost(kind: str, task_type: str | None) -> float:
    if not task_type:
        return 0.0
    planning_kinds = {"stance", "procedure", "decision", "open_loop", "contradiction", "attractor"}
    if task_type in ("planning", "architecture") and kind in planning_kinds:
        return 0.1
    return 0.0


def build_active_packet(
    *,
    query: str,
    crystallizations: list[MemoryCrystallizationV1],
    card_refs: list[str] | None = None,
    active_cards: list[dict[str, Any]] | None = None,
    task_type: str | None = None,
    project_id: str | None = None,
    session_id: str | None = None,
    now: datetime | None = None,
) -> ActiveMemoryPacketV1:
    ranking_time = now if now is not None else datetime.now(timezone.utc)
    active = [c for c in crystallizations if eligible_for_recall(c)]
    if project_id:
        active = [c for c in active if project_id in c.scope or c.scope == []]
    active.sort(
        key=lambda c: (
            decayed_activation(c, now=ranking_time) * (c.salience or 0.5) + _task_boost(c.kind, task_type)
        ),
        reverse=True,
    )

    cards = list(active_cards or [])
    card_ref_ids = list(card_refs or [])
    if not card_ref_ids and cards:
        card_ref_ids = [str(c.get("card_id")) for c in cards if c.get("card_id")]

    packet = ActiveMemoryPacketV1(
        query=query,
        task_type=task_type,
        project_id=project_id,
        session_id=session_id,
        card_refs=card_ref_ids,
        crystallization_refs=[c.crystallization_id for c in active],
        retrieval_trace={
            "strategy": "multi_rail_salience_ranked",
            "crystallization_count": len(active),
            "card_count": len(card_ref_ids),
            "task_type": task_type,
            "rails": ["postgres_crystallizations", "postgres_memory_cards"],
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
