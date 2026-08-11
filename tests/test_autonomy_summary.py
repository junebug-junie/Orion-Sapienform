from __future__ import annotations

from orion.autonomy.models import AutonomyGoalHeadlineV1, AutonomyStateV1
from orion.autonomy.summary import summarize_autonomy_state


def test_autonomy_summary_empty_state_safe_defaults() -> None:
    summary = summarize_autonomy_state(None)
    assert summary.dominant_drive is None
    assert summary.raw_state_present is False
    assert summary.proposal_headlines == []
    assert summary.response_hazards == []


def test_proposal_headlines_strip_chat_and_trace_suffix() -> None:
    # Renamed scope 2026-07-30 (chore/delete-orion-drives Wave 2a follow-up):
    # this test used to also assert on dominant_drive/drive_competition,
    # computed from AutonomyStateV1.drive_pressures/tension_kinds/
    # active_drives, all removed. summarize_autonomy_state() now always
    # returns dominant_drive=None/drive_competition=None for a non-empty
    # state (see orion/autonomy/summary.py's own comment) -- what remains
    # real and worth testing is proposal_headlines' chat/trace-suffix
    # stripping, which is independent of drives.
    state = AutonomyStateV1(
        subject="orion",
        model_layer="self-model",
        entity_id="orion",
        goal_headlines=[
            AutonomyGoalHeadlineV1(
                artifact_id="goal-1",
                goal_statement=(
                    "Clarify autonomy boundaries without executing any new action. · trace=abcd1234 · "
                    "eh, just testing user chat should not dominate the card"
                ),
                priority=0.8,
                cooldown_until=None,
                proposal_signature="sig-1",
            )
        ],
        source="graph",
    )
    summary = summarize_autonomy_state(state)
    assert summary.dominant_drive is None
    assert summary.proposal_headlines == ["Clarify autonomy boundaries without executing any new action."]
