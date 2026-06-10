"""Proposal assembly helpers.

These helpers build well-formed MemoryCrystallizationV1 proposals from
existing artifacts (MemoryCardV1 rows, GrammarEventV1 traces, operator
notes). They never write canonical state: output always has status
"proposed" and an unvalidated governance block.

Local models may use these to shape proposals; the governor canonizes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from orion.schemas.memory_crystallization import (
    CrystallizationClaimV1,
    CrystallizationEvidenceRefV1,
    CrystallizationGovernanceV1,
    MemoryCrystallizationV1,
    MemoryGrammarEnvelopeV1,
    new_crystallization_id,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def build_proposal(
    *,
    kind: str,
    subject: str,
    summary: str,
    proposed_by: str,
    scope: list[str],
    evidence: list[CrystallizationEvidenceRefV1],
    claims: list[CrystallizationClaimV1] | None = None,
    confidence: str = "likely",
    tags: list[str] | None = None,
    planning_effects: list[str] | None = None,
    retrieval_affordances: list[str] | None = None,
    grammar_envelope: MemoryGrammarEnvelopeV1 | None = None,
    sensitivity: str = "private",
    created_from_policy: str | None = None,
) -> MemoryCrystallizationV1:
    """Assemble a crystallization proposal (status=proposed, unvalidated)."""
    now = _now()
    envelope = grammar_envelope or MemoryGrammarEnvelopeV1()
    return MemoryCrystallizationV1(
        crystallization_id=new_crystallization_id(),
        kind=kind,  # type: ignore[arg-type]
        subject=subject,
        summary=summary,
        status="proposed",
        confidence=confidence,  # type: ignore[arg-type]
        scope=list(scope),
        tags=list(tags or []),
        claims=list(claims or []),
        evidence=list(evidence),
        source_card_ids=[ev.source_id for ev in evidence if ev.source_kind == "memory_card"],
        source_grammar_event_ids=list(envelope.source_grammar_event_ids),
        source_atom_ids=list(envelope.source_atom_ids),
        grammar_envelope=envelope,
        planning_effects=list(planning_effects or []),
        retrieval_affordances=list(retrieval_affordances or []),
        governance=CrystallizationGovernanceV1(
            proposed_by=proposed_by,
            sensitivity=sensitivity,  # type: ignore[arg-type]
            created_from_policy=created_from_policy,
        ),
        created_at=now,
        updated_at=now,
    )


def evidence_from_cards(cards: list[dict[str, Any]], *, strength: float = 0.6) -> list[CrystallizationEvidenceRefV1]:
    """Build evidence refs from MemoryCardV1-shaped dicts (card_id + summary)."""
    refs: list[CrystallizationEvidenceRefV1] = []
    for card in cards:
        card_id = str(card.get("card_id") or "").strip()
        if not card_id:
            continue
        refs.append(
            CrystallizationEvidenceRefV1(
                source_kind="memory_card",
                source_id=card_id,
                excerpt=(card.get("summary") or None),
                strength=strength,
            )
        )
    return refs


def evidence_from_grammar_events(
    event_ids: list[str], *, strength: float = 0.5
) -> list[CrystallizationEvidenceRefV1]:
    """Build evidence refs from existing GrammarEventV1 event ids."""
    return [
        CrystallizationEvidenceRefV1(source_kind="grammar_event", source_id=event_id, strength=strength)
        for event_id in event_ids
        if event_id.strip()
    ]
