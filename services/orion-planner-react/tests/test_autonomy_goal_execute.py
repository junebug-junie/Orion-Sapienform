from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import Any

import pytest

SERVICE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO_ROOT = os.path.abspath(os.path.join(SERVICE_DIR, "..", ".."))
for path in (SERVICE_DIR, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from app.api import (  # noqa: E402
    AUTONOMY_GOAL_EXECUTE_VERB,
    AutonomyGoalExecuteInputV1,
    execute_autonomy_goal_v1,
    list_planner_verbs,
)
from orion.autonomy.models import AutonomyGoalHeadlineV1  # noqa: E402
from orion.core.bus.bus_schemas import BaseEnvelope  # noqa: E402


class _FakeGraphClient:
    def __init__(self) -> None:
        self.updates: list[str] = []

    def update(self, sparql: str) -> None:
        self.updates.append(sparql)


class _FakeBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, BaseEnvelope]] = []

    async def publish(self, channel: str, envelope: BaseEnvelope) -> None:
        self.published.append((channel, envelope))


def test_autonomy_goal_execute_verb_is_registered() -> None:
    assert AUTONOMY_GOAL_EXECUTE_VERB in list_planner_verbs()


def _planned_goal(*, artifact_id: str = "goal-abc-123", task_id: str | None = "task-existing") -> AutonomyGoalHeadlineV1:
    return AutonomyGoalHeadlineV1(
        artifact_id=artifact_id,
        goal_statement="Archive stale proposed goals older than retention window",
        drive_origin="autonomy",
        priority=0.8,
        proposal_signature="sig-abc",
        proposal_status="planned",
        planned_task_id=task_id,
    )


@pytest.mark.asyncio
async def test_autonomy_goal_execute_returns_task_id_and_publishes_supervisor_event(monkeypatch) -> None:
    fake_graph = _FakeGraphClient()
    fake_bus = _FakeBus()

    monkeypatch.setattr(
        "app.api.build_goal_graph_query_client",
        lambda: fake_graph,
    )
    monkeypatch.setattr(
        "app.api.fetch_goal_by_artifact_id",
        lambda _client, artifact_id: (_planned_goal(artifact_id=artifact_id), "orion"),
    )

    payload = AutonomyGoalExecuteInputV1(
        goal_artifact_id="goal-abc-123",
        goal_statement="Archive stale proposed goals older than retention window",
        drive_origin="autonomy",
    )

    result = await execute_autonomy_goal_v1(payload, bus=fake_bus)

    assert result.ok is True
    assert result.goal_artifact_id == "goal-abc-123"
    assert result.task_id == "task-existing"
    assert result.proposal_status == "executing"

    assert len(fake_graph.updates) == 1
    assert "goal-abc-123" in fake_graph.updates[0]
    assert "task-existing" in fake_graph.updates[0]
    assert "executing" in fake_graph.updates[0]

    assert len(fake_bus.published) == 1
    channel, env = fake_bus.published[0]
    assert channel == "orion:autonomy:goal:planned"
    assert env.kind == "autonomy.goal.planned.v1"
    assert env.payload["task_id"] == result.task_id
    assert env.payload["goal_artifact_id"] == "goal-abc-123"
    assert env.payload["drive_origin"] == "autonomy"


@pytest.mark.asyncio
async def test_autonomy_goal_execute_still_returns_task_id_when_graph_unconfigured(monkeypatch) -> None:
    fake_bus = _FakeBus()
    monkeypatch.setattr("app.api.build_goal_graph_query_client", lambda: None)
    monkeypatch.setattr(
        "app.api.fetch_goal_by_artifact_id",
        lambda _client, artifact_id: (_planned_goal(artifact_id=artifact_id, task_id=None), "orion"),
    )

    with pytest.raises(ValueError, match="graph_not_configured"):
        await execute_autonomy_goal_v1(
            AutonomyGoalExecuteInputV1(
                goal_artifact_id="goal-no-graph",
                goal_statement="Test goal",
                drive_origin="curiosity",
            ),
            bus=fake_bus,
        )
