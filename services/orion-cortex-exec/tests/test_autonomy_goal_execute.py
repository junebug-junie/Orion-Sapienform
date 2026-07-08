from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.autonomy_goal_execute import (
    AUTONOMY_GOAL_EXECUTE_VERB,
    AutonomyGoalExecuteInputV1,
    execute_autonomy_goal_v1,
)
from orion.core.bus.bus_schemas import BaseEnvelope


@pytest.mark.asyncio
async def test_execute_autonomy_goal_v1_publishes_planned_event(monkeypatch: pytest.MonkeyPatch) -> None:
    published: list[tuple[str, BaseEnvelope]] = []

    class _Bus:
        async def publish(self, channel: str, env: BaseEnvelope) -> None:
            published.append((channel, env))

    goal = MagicMock()
    goal.proposal_status = "planned"
    goal.planned_task_id = "task-123"
    goal.goal_statement = "Clarify autonomy boundaries"

    monkeypatch.setattr(
        "app.autonomy_goal_execute.build_goal_graph_query_client",
        lambda: object(),
    )
    monkeypatch.setattr(
        "app.autonomy_goal_execute.fetch_goal_by_artifact_id",
        lambda _client, artifact_id: (goal, "orion") if artifact_id == "goal-abc" else None,
    )
    monkeypatch.setattr(
        "app.autonomy_goal_execute.update_goal_planned_task_id",
        lambda *_args, **_kwargs: None,
    )

    out = await execute_autonomy_goal_v1(
        AutonomyGoalExecuteInputV1(goal_artifact_id="goal-abc"),
        bus=_Bus(),
        correlation_id="00000000-0000-4000-8000-000000000099",
    )

    assert out.task_id == "task-123"
    assert out.goal_artifact_id == "goal-abc"
    assert len(published) == 1
    channel, env = published[0]
    assert channel == "orion:autonomy:goal:planned"
    assert env.kind == "autonomy.goal.planned.v1"
    assert env.payload["source_verb"] == AUTONOMY_GOAL_EXECUTE_VERB
