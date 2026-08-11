from __future__ import annotations

from orion.autonomy.models import AutonomyGoalHeadlineV1
from orion.autonomy.summary import _top_goal_headlines_by_priority


def _goal(artifact_id: str, priority: float) -> AutonomyGoalHeadlineV1:
    return AutonomyGoalHeadlineV1(
        artifact_id=artifact_id,
        goal_statement=f"Goal {artifact_id}",
        priority=priority,
        proposal_signature=artifact_id,
    )


def test_top_goal_headlines_by_priority_ranks_highest_first() -> None:
    goals = [
        _goal("goal-low", 0.1),
        _goal("goal-high", 0.9),
        _goal("goal-mid", 0.5),
    ]
    out = _top_goal_headlines_by_priority(goals, limit=3)
    assert [g.artifact_id for g in out] == ["goal-high", "goal-mid", "goal-low"]


def test_top_goal_headlines_by_priority_caps_at_limit() -> None:
    goals = [_goal("goal-a", 0.9), _goal("goal-b", 0.8), _goal("goal-c", 0.7)]
    out = _top_goal_headlines_by_priority(goals, limit=2)
    assert [g.artifact_id for g in out] == ["goal-a", "goal-b"]


def test_top_goal_headlines_by_priority_ties_break_on_artifact_id() -> None:
    goals = [_goal("goal-z", 0.5), _goal("goal-a", 0.5)]
    out = _top_goal_headlines_by_priority(goals, limit=2)
    assert [g.artifact_id for g in out] == ["goal-a", "goal-z"]
