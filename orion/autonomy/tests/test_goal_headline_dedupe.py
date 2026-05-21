from __future__ import annotations

from orion.autonomy.models import AutonomyGoalHeadlineV1
from orion.autonomy.summary import dedupe_goal_headlines_by_drive_origin


def _goal(artifact_id: str, drive_origin: str, priority: float) -> AutonomyGoalHeadlineV1:
    return AutonomyGoalHeadlineV1(
        artifact_id=artifact_id,
        goal_statement=f"Goal for {drive_origin}",
        drive_origin=drive_origin,
        priority=priority,
        proposal_signature=artifact_id,
    )


def test_dedupe_goal_headlines_keeps_highest_priority_per_drive_origin() -> None:
    goals = [
        _goal("goal-high", "autonomy", 0.9),
        _goal("goal-low", "autonomy", 0.1),
        _goal("goal-cont", "continuity", 0.5),
    ]
    out = dedupe_goal_headlines_by_drive_origin(goals, limit=3)
    assert [g.artifact_id for g in out] == ["goal-high", "goal-cont"]
