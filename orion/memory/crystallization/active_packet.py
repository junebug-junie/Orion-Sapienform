"""Active memory packet assembly (spec section 15).

The retriever does not dump raw cards: it groups retrieved crystallizations
by kind into a compact, inspectable ActiveMemoryPacketV1.
"""

from __future__ import annotations

from typing import Any

from orion.schemas.memory_crystallization import (
    ActiveMemoryPacketV1,
    MemoryCrystallizationV1,
)

RETRIEVABLE_STATUSES = frozenset({"active"})

_KIND_TO_SECTION: dict[str, str] = {
    "stance": "stance",
    "semantic": "project_state",
    "decision": "project_state",
    "procedure": "procedures",
    "open_loop": "open_loops",
    "contradiction": "contradictions",
    "failure_mode": "warnings",
    "attractor": "attractors",
    "episode": "project_state",
}


def _packet_item(crystallization: MemoryCrystallizationV1) -> dict[str, Any]:
    item: dict[str, Any] = {
        "crystallization_id": crystallization.crystallization_id,
        "kind": crystallization.kind,
        "summary": crystallization.summary,
        "salience": crystallization.salience,
        "confidence": crystallization.confidence,
    }
    if crystallization.planning_effects:
        item["planning_effects"] = list(crystallization.planning_effects)
    return item


def build_active_packet(
    *,
    query: str,
    crystallizations: list[MemoryCrystallizationV1],
    task_type: str | None = None,
    project_id: str | None = None,
    session_id: str | None = None,
    card_refs: list[str] | None = None,
    chroma_refs: list[str] | None = None,
    graphiti_refs: list[str] | None = None,
    rdf_refs: list[str] | None = None,
    retrieval_trace: dict[str, Any] | None = None,
) -> ActiveMemoryPacketV1:
    """Group retrieved active crystallizations into a prompt-ready packet.

    Non-active crystallizations are dropped (and noted in the trace) rather
    than silently injected into cognition.
    """
    sections: dict[str, list[dict[str, Any]]] = {
        "stance": [],
        "project_state": [],
        "procedures": [],
        "open_loops": [],
        "contradictions": [],
        "warnings": [],
        "attractors": [],
    }
    included: list[str] = []
    excluded: list[dict[str, str]] = []

    ordered = sorted(crystallizations, key=lambda c: c.salience, reverse=True)
    for crystallization in ordered:
        if crystallization.status not in RETRIEVABLE_STATUSES:
            excluded.append(
                {
                    "crystallization_id": crystallization.crystallization_id,
                    "reason": f"status:{crystallization.status}",
                }
            )
            continue
        section = _KIND_TO_SECTION.get(crystallization.kind, "project_state")
        sections[section].append(_packet_item(crystallization))
        included.append(crystallization.crystallization_id)

    trace = dict(retrieval_trace or {})
    trace.setdefault("included_count", len(included))
    if excluded:
        trace["excluded"] = excluded

    return ActiveMemoryPacketV1(
        query=query,
        task_type=task_type,
        project_id=project_id,
        session_id=session_id,
        stance=sections["stance"],
        project_state=sections["project_state"],
        procedures=sections["procedures"],
        open_loops=sections["open_loops"],
        contradictions=sections["contradictions"],
        warnings=sections["warnings"],
        attractors=sections["attractors"],
        card_refs=list(card_refs or []),
        crystallization_refs=included,
        graphiti_refs=list(graphiti_refs or []),
        chroma_refs=list(chroma_refs or []),
        rdf_refs=list(rdf_refs or []),
        retrieval_trace=trace,
    )
