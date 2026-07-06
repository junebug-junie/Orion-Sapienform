from __future__ import annotations

from orion.journaler.schemas import JournalEntryWriteV1
from orion.memory.crystallization.schemas import (
    CrystallizationEvidenceRefV1,
    CrystallizationGovernanceV1,
    MemoryCrystallizationV1,
    _utc_now,
    new_crystallization_id,
)

_INTAKE_PROPOSED_BY = "autonomy_episode_intake"


def _episode_scope(journal_entry: JournalEntryWriteV1) -> list[str]:
    if journal_entry.source_ref:
        return [f"autonomy:episode:{journal_entry.source_ref}"]
    return ["autonomy:episode"]


def _episode_summary(journal_entry: JournalEntryWriteV1) -> str:
    if journal_entry.title and journal_entry.title.strip():
        return journal_entry.title.strip()
    body = (journal_entry.body or "").strip()
    if not body:
        return "Autonomy episode journal entry"
    first_line = body.splitlines()[0].strip()
    return first_line or "Autonomy episode journal entry"


def build_crystallization_from_episode(
    journal_entry: JournalEntryWriteV1,
    spawned_correlation_id: str,
    grammar_event_ids: list[str],
) -> MemoryCrystallizationV1:
    """Build a proposed episode crystallization from an autonomy journal entry."""
    now = _utc_now()
    episode_source_id = journal_entry.entry_id
    evidence = [
        CrystallizationEvidenceRefV1(
            source_kind="autonomy_episode",
            source_id=episode_source_id,
            excerpt=(journal_entry.body or "")[:2000] or None,
            strength=0.8,
            note=journal_entry.source_ref,
        ),
        *[
            CrystallizationEvidenceRefV1(
                source_kind="grammar_event",
                source_id=event_id,
                strength=0.6,
            )
            for event_id in grammar_event_ids
        ],
    ]
    return MemoryCrystallizationV1(
        crystallization_id=new_crystallization_id(),
        kind="episode",
        subject=_episode_summary(journal_entry),
        summary=_episode_summary(journal_entry),
        status="proposed",
        scope=_episode_scope(journal_entry),
        tags=["autonomy_episode"],
        evidence=evidence,
        source_grammar_event_ids=list(grammar_event_ids),
        governance=CrystallizationGovernanceV1(
            proposed_by=_INTAKE_PROPOSED_BY,
            requires_manual_review=True,
            sensitivity="private",
            created_from_policy="autonomy_episode_intake",
        ),
        provenance={
            "spawned_correlation_id": spawned_correlation_id,
            "source_kind": "autonomy_episode",
            "journal_entry_id": episode_source_id,
            "source_ref": journal_entry.source_ref,
            "correlation_id": journal_entry.correlation_id,
        },
        created_at=now,
        updated_at=now,
    )
