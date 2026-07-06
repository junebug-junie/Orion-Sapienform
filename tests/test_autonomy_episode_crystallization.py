from __future__ import annotations

from orion.journaler.schemas import JournalEntryWriteV1
from orion.memory.crystallization.intake_autonomy_episode import build_crystallization_from_episode


def test_crystallization_proposal_marks_autonomy_episode_source() -> None:
    entry = JournalEntryWriteV1(
        author="orion",
        mode="digest",
        title="Episode: GPU coverage gap",
        body="## Gap\nhardware_compute_gpu empty\n## Learnings\nFetched two articles.",
        source_kind="autonomy_episode",
        source_ref="goal-gap-gpu",
        correlation_id="wp-run-gap-gpu",
    )
    proposal = build_crystallization_from_episode(
        journal_entry=entry,
        spawned_correlation_id="wp-run-gap-gpu",
        grammar_event_ids=["gram-1", "gram-2"],
    )
    assert proposal.kind == "episode"
    assert proposal.status == "proposed"
    assert proposal.governance.proposed_by == "autonomy_episode_intake"
    assert any(ev.source_kind == "autonomy_episode" for ev in proposal.evidence)
    assert any(ev.source_kind == "grammar_event" for ev in proposal.evidence)
    assert proposal.source_grammar_event_ids == ["gram-1", "gram-2"]
    assert proposal.provenance.get("spawned_correlation_id") == "wp-run-gap-gpu"
    assert proposal.provenance.get("source_kind") == "autonomy_episode"
    assert proposal.scope
