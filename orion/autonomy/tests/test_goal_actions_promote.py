from __future__ import annotations

import pytest

from orion.autonomy.goal_actions import (
    GoalActionError,
    GoalActionResult,
    plan_promoted_goal,
    promote_goal,
    update_goal_planned_task_id,
)
from orion.autonomy.models import AutonomyGoalHeadlineV1


class _FakeGraphClient:
    def __init__(self) -> None:
        self.updates: list[str] = []
        self.select_rows: list[dict] = []

    def update(self, sparql: str) -> None:
        self.updates.append(sparql)

    def select(self, sparql: str) -> list[dict]:
        del sparql
        return list(self.select_rows)


def _goal_row(*, artifact_id: str = "goal-abc", status: str = "active") -> dict:
    return {
        "artifact_id": {"value": artifact_id},
        "goal_statement": {"value": "Stabilize coherence without auto execution."},
        "drive_origin": {"value": "coherence"},
        "priority": {"value": "0.72"},
        "proposal_signature": {"value": "deadbeef01"},
        "proposal_status": {"value": status},
        "subject_key": {"value": "orion"},
    }


def test_plan_promoted_goal_persists_task_id_with_planned_status(monkeypatch) -> None:
    monkeypatch.setenv("AUTONOMY_GOAL_EXECUTION_ENABLED", "true")
    client = _FakeGraphClient()
    goal = AutonomyGoalHeadlineV1(
        artifact_id="goal-abc",
        goal_statement="Stabilize coherence",
        drive_origin="coherence",
        priority=0.72,
        proposal_signature="deadbeef01",
        proposal_status="planned",
    )
    task_id = plan_promoted_goal(goal=goal, graph_client=client)
    assert task_id.startswith("goal-task-")
    assert len(client.updates) == 1
    assert "planned" in client.updates[0]
    assert task_id in client.updates[0]


def test_promote_goal_chains_planner_task_id(monkeypatch) -> None:
    monkeypatch.setenv("AUTONOMY_GOAL_EXECUTION_ENABLED", "true")
    client = _FakeGraphClient()
    client.select_rows = [_goal_row(status="active")]

    def _fake_promotion(**kwargs):
        from orion.core.schemas.reasoning_policy import PromotionEvaluationItemV1, PromotionEvaluationResultV1

        del kwargs
        return PromotionEvaluationResultV1(
            request_id="test-req",
            policy_version="phase3.v1",
            evaluated_count=1,
            items=[
                PromotionEvaluationItemV1(
                    artifact_id="goal-reasoning-goal-abc",
                    artifact_type="claim",
                    current_status="proposed",
                    target_status="canonical",
                    outcome="escalated_hitl",
                    reasons=["autonomy_goal_requires_hitl"],
                    risk_tier="medium",
                    human_review_required=True,
                )
            ],
        )

    monkeypatch.setattr("orion.autonomy.goal_actions.apply_operator_goal_reasoning_promotion", _fake_promotion)

    result = promote_goal(artifact_id="goal-abc", operator="operator-1", graph_client=client)
    assert isinstance(result, GoalActionResult)
    assert result.proposal_status == "planned"
    assert result.planned_task_id is not None
    assert result.planned_task_id.startswith("goal-task-")
    assert any("planned" in update for update in client.updates)


def test_plan_promoted_goal_requires_planned_status(monkeypatch) -> None:
    monkeypatch.setenv("AUTONOMY_GOAL_EXECUTION_ENABLED", "true")
    goal = AutonomyGoalHeadlineV1(
        artifact_id="goal-abc",
        goal_statement="Stabilize coherence",
        drive_origin="coherence",
        priority=0.72,
        proposal_signature="deadbeef01",
        proposal_status="active",
    )
    with pytest.raises(GoalActionError) as exc:
        plan_promoted_goal(goal=goal, graph_client=_FakeGraphClient())
    assert exc.value.code == "goal_not_planned"


def test_update_goal_planned_task_id_keeps_custom_status() -> None:
    client = _FakeGraphClient()
    update_goal_planned_task_id(client, "goal-xyz", "task-123", proposal_status="planned")
    assert "planned" in client.updates[0]
    assert "task-123" in client.updates[0]
