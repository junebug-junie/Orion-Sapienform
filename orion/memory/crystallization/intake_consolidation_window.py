from __future__ import annotations

from typing import Any

from orion.memory.consolidation_gate import ConsolidationGateResult
from orion.memory.crystallization.schemas import (
    CrystallizationEvidenceRefV1,
    CrystallizationGovernanceV1,
    MemoryCrystallizationV1,
    _utc_now,
    new_crystallization_id,
)

_PROPOSED_BY = "memory_consolidation_intake"
_POLICY = "consolidation_window_gate_v1"

_KIND_FOR_SHIFT = {
    "STANCE": "stance",
    "REPAIR": "open_loop",
    "TOPIC": "semantic",
}


def _window_summary(turns: list[dict[str, Any]]) -> str:
    for turn in reversed(turns):
        prompt = str(turn.get("prompt") or "").strip()
        if prompt:
            return prompt[:500]
    return "Consolidated chat window"


def _kind_for_gate(gate: ConsolidationGateResult, turns: list[dict[str, Any]]) -> str:
    if gate.dominant_shift:
        return _KIND_FOR_SHIFT.get(gate.dominant_shift, "episode")
    if len(turns) > 1:
        return "episode"
    return "episode"


def build_crystallization_from_window(
    *,
    memory_window_id: str,
    turns: list[dict[str, Any]],
    gate: ConsolidationGateResult,
) -> MemoryCrystallizationV1:
    now = _utc_now()
    summary = _window_summary(turns)
    evidence: list[CrystallizationEvidenceRefV1] = []
    for turn in turns:
        corr = str(turn.get("correlation_id") or "").strip()
        if not corr:
            continue
        excerpt = f"{turn.get('prompt', '')}\n{turn.get('response', '')}"[:2000] or None
        evidence.append(
            CrystallizationEvidenceRefV1(
                source_kind="chat_turn",
                source_id=corr,
                excerpt=excerpt,
                strength=0.75,
            )
        )
    grammar_ids = list(gate.grammar_event_ids)
    for event_id in grammar_ids:
        evidence.append(
            CrystallizationEvidenceRefV1(
                source_kind="grammar_event",
                source_id=event_id,
                strength=0.6,
            )
        )
    return MemoryCrystallizationV1(
        crystallization_id=new_crystallization_id(),
        kind=_kind_for_gate(gate, turns),
        subject=summary,
        summary=summary,
        status="proposed",
        scope=[f"memory_window:{memory_window_id}"],
        tags=["consolidation_window"],
        evidence=evidence,
        source_grammar_event_ids=grammar_ids,
        governance=CrystallizationGovernanceV1(
            proposed_by=_PROPOSED_BY,
            requires_manual_review=True,
            sensitivity="private",
            created_from_policy=_POLICY,
        ),
        provenance={
            "memory_window_id": memory_window_id,
            "dominant_shift": gate.dominant_shift,
            "window_novelty_max": gate.window_novelty_max,
            "window_significance_max": gate.window_significance_max,
            "gate_reasons": gate.reasons,
        },
        created_at=now,
        updated_at=now,
    )
