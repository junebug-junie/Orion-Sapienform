from __future__ import annotations

from datetime import datetime, timezone

from orion.autonomy.models import AutonomyGoalHeadlineV1, AutonomyStateV1
from orion.autonomy.summary import summarize_autonomy_state


def test_active_goals_expose_planned_and_executing_qualifiers() -> None:
    state = AutonomyStateV1(
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        source="graph",
        generated_at=datetime(2026, 5, 21, 12, 0, tzinfo=timezone.utc),
        goal_headlines=[
            AutonomyGoalHeadlineV1(
                artifact_id="goal-planned",
                goal_statement="Plan the coherence review.",
                drive_origin="coherence",
                priority=0.85,
                proposal_signature="sig-planned",
                proposal_status="planned",
                planned_task_id="task-abc-123",
            ),
            AutonomyGoalHeadlineV1(
                artifact_id="goal-executing",
                goal_statement="Execute the coherence review.",
                drive_origin="predictive",
                priority=0.8,
                proposal_signature="sig-executing",
                proposal_status="executing",
                planned_task_id="task-def-456",
            ),
        ],
    )

    summary = summarize_autonomy_state(state)

    assert summary.goals_present is True
    assert len(summary.active_goals) == 2
    assert summary.active_goals[0].proposal_status == "planned"
    assert summary.active_goals[0].planned_task_id == "task-abc-123"
    assert summary.active_goals[1].proposal_status == "executing"
    assert summary.active_goals[1].planned_task_id == "task-def-456"
