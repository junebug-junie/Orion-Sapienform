from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from app.router import _autonomy_payload_from_ctx
from app.supervisor import Supervisor, _autonomy_goal_execution_allowed, _is_autonomy_goal_execute_action
from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.cortex.schemas import ExecutionPlan, PlanExecutionArgs, PlanExecutionRequest, StepExecutionResult

CORR_UNPROMOTED = "00000000-0000-4000-8000-000000000001"
CORR_PROMOTED = "00000000-0000-4000-8000-000000000002"


def _supervised_request() -> PlanExecutionRequest:
    plan = ExecutionPlan(
        verb_name="agent_runtime",
        label="agent_runtime",
        description="",
        category="x",
        priority="normal",
        interruptible=True,
        can_interrupt_others=False,
        timeout_ms=1000,
        max_recursion_depth=2,
        metadata={"mode": "agent", "execution_depth": "2"},
        steps=[],
    )
    return PlanExecutionRequest(
        plan=plan,
        args=PlanExecutionArgs(request_id="r-goal-e2e", extra={"mode": "agent", "supervised": True}),
        context={"mode": "agent", "messages": [{"role": "user", "content": "run promoted goal"}]},
    )


@pytest.mark.parametrize(
    ("proposal_status", "expected_mode"),
    [
        ("proposed", "hint_only"),
        ("planned", "planned"),
        ("executing", "executing"),
        ("completed", "none"),
    ],
)
def test_execution_mode_ladder_from_active_goals(proposal_status: str, expected_mode: str) -> None:
    ctx = {
        "chat_autonomy_execution_mode": "hint_only" if proposal_status == "proposed" else None,
        "chat_autonomy_summary": {
            "goals_present": True,
            "active_goals": [
                {
                    "artifact_id": "goal-abc",
                    "proposal_status": proposal_status,
                    "planned_task_id": "task-123" if proposal_status in {"planned", "executing"} else None,
                    "priority": 0.8,
                    "headline": "Clarify autonomy boundaries",
                }
            ],
        },
    }
    payload = _autonomy_payload_from_ctx(ctx)
    assert payload["autonomy_execution_mode"] == expected_mode


def test_unpromoted_goal_execute_action_is_detected() -> None:
    action = {
        "tool_id": "autonomy.goal.execute.v1",
        "input": {"goal_artifact_id": "goal-abc", "autonomy_goal_execute": True},
    }
    assert _is_autonomy_goal_execute_action(action) is True


def test_unpromoted_goal_execution_not_allowed_without_promote(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTONOMY_GOAL_EXECUTION_ENABLED", "true")
    ctx = {
        "chat_autonomy_summary": {
            "active_goals": [{"artifact_id": "goal-abc", "proposal_status": "proposed"}]
        }
    }
    action = {
        "tool_id": "autonomy.goal.execute.v1",
        "input": {"goal_artifact_id": "goal-abc", "autonomy_goal_execute": True},
    }
    assert _autonomy_goal_execution_allowed(ctx, action) is False


def test_promoted_goal_execution_allowed_after_operator_promote(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTONOMY_GOAL_EXECUTION_ENABLED", "true")
    ctx = {
        "chat_autonomy_summary": {
            "active_goals": [
                {
                    "artifact_id": "goal-abc",
                    "proposal_status": "planned",
                    "planned_task_id": "task-123",
                }
            ]
        }
    }
    action = {
        "tool_id": "autonomy.goal.execute.v1",
        "input": {"goal_artifact_id": "goal-abc", "autonomy_goal_execute": True},
    }
    assert _autonomy_goal_execution_allowed(ctx, action) is True


def test_supervisor_blocks_unpromoted_goal_execute(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTONOMY_GOAL_EXECUTION_ENABLED", "true")
    monkeypatch.setattr("app.settings.settings.context_exec_enabled", False)

    supervisor = Supervisor(object())
    monkeypatch.setattr(
        supervisor,
        "_execute_action",
        AsyncMock(side_effect=AssertionError("unpromoted goal must not execute")),
    )
    monkeypatch.setattr(
        "app.supervisor.run_recall_step",
        AsyncMock(
            return_value=(
                StepExecutionResult(
                    status="success",
                    verb_name="recall",
                    step_name="recall",
                    order=0,
                    result={},
                    latency_ms=1,
                    node="n",
                    logs=[],
                    error=None,
                ),
                {},
                None,
            )
        ),
    )

    ctx = {
        "mode": "agent",
        "messages": [{"role": "user", "content": "execute goal"}],
        "chat_autonomy_execution_mode": "hint_only",
        "chat_autonomy_summary": {
            "active_goals": [{"artifact_id": "goal-abc", "proposal_status": "proposed", "priority": 0.9}]
        },
    }
    result = asyncio.run(
        supervisor.execute(
            source=ServiceRef(name="x", version="0", node="n"),
            req=_supervised_request().plan,
            correlation_id=CORR_UNPROMOTED,
            ctx=ctx,
            recall_cfg={},
        )
    )
    assert result.status == "success"
    assert "operator promotes" in (result.final_text or "").lower()
    assert result.metadata["autonomy_execution_mode"] == "hint_only"


def test_supervisor_e2e_promote_plan_execute_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTONOMY_GOAL_EXECUTION_ENABLED", "true")
    monkeypatch.setattr("app.settings.settings.context_exec_enabled", False)
    execute_calls: list[str] = []

    async def _fake_execute_action(**kwargs):
        execute_calls.append(str((kwargs.get("action") or {}).get("tool_id")))
        return StepExecutionResult(
            status="success",
            verb_name="autonomy.goal.execute.v1",
            step_name="autonomy_goal_execute",
            order=0,
            result={"ContextExecService": {"text": "Goal task completed"}},
            latency_ms=1,
            node="n",
            logs=[],
            error=None,
        )

    supervisor = Supervisor(object())
    monkeypatch.setattr(supervisor, "_execute_action", _fake_execute_action)
    monkeypatch.setattr(
        "app.supervisor.run_recall_step",
        AsyncMock(
            return_value=(
                StepExecutionResult(
                    status="success",
                    verb_name="recall",
                    step_name="recall",
                    order=0,
                    result={},
                    latency_ms=1,
                    node="n",
                    logs=[],
                    error=None,
                ),
                {},
                None,
            )
        ),
    )

    ctx = {
        "mode": "agent",
        "messages": [{"role": "user", "content": "execute promoted goal"}],
        "chat_autonomy_summary": {
            "active_goals": [
                {
                    "artifact_id": "goal-abc",
                    "proposal_status": "planned",
                    "planned_task_id": "task-123",
                    "priority": 0.9,
                }
            ]
        },
    }
    planned_payload = _autonomy_payload_from_ctx(ctx)
    assert planned_payload["autonomy_execution_mode"] == "planned"

    result = asyncio.run(
        supervisor.execute(
            source=ServiceRef(name="x", version="0", node="n"),
            req=_supervised_request().plan,
            correlation_id=CORR_PROMOTED,
            ctx=ctx,
            recall_cfg={},
        )
    )
    assert execute_calls == ["autonomy.goal.execute.v1"]
    assert result.metadata["autonomy_execution_mode"] == "executing"

    ctx["chat_autonomy_summary"]["active_goals"][0]["proposal_status"] = "completed"
    ctx["chat_autonomy_summary"]["active_goals"][0]["planned_task_id"] = None
    completed_payload = _autonomy_payload_from_ctx(ctx)
    assert completed_payload["autonomy_execution_mode"] == "none"
