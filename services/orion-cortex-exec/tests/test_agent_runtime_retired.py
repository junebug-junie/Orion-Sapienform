"""orion-context-exec retirement (2026-10-10): depth-2 agent work fails fast.

Before: with CONTEXT_EXEC_ENABLED=true (the compose default), every mode=agent
supervised turn published a ContextExecRequestV1 to
orion:exec:request:ContextExecService -- a channel with no consumer -- waited
CONTEXT_EXEC_TIMEOUT_SEC (60s) and then returned "Insufficient grounding:
context-exec failed before acquiring evidence." Now the Supervisor never touches
the bus for that path and says plainly there is no depth-2 runtime.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

import app.clients as clients
import app.settings as app_settings
from app.supervisor import Supervisor
from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.cortex.schemas import ExecutionPlan, PlanExecutionArgs, PlanExecutionRequest, StepExecutionResult


class _NoPublishBus:
    """Any bus call fails the test: the retired path must not publish anything."""

    def __getattr__(self, name):  # pragma: no cover - only hit on regression
        raise AssertionError(f"agent_runtime path touched the bus: {name}")


def _agent_plan() -> ExecutionPlan:
    return ExecutionPlan(
        verb_name="agent_runtime",
        label="agent_runtime",
        description="",
        category="agentic",
        priority="normal",
        interruptible=True,
        can_interrupt_others=False,
        timeout_ms=1000,
        max_recursion_depth=1,
        metadata={"mode": "agent"},
        steps=[],
    )


def _recall_ok():
    return (
        StepExecutionResult(
            status="success", verb_name="recall", step_name="recall", order=0,
            result={}, latency_ms=1, node="n", logs=[], error=None,
        ),
        {},
        None,
    )


@pytest.mark.parametrize("options", [{}, {"agent_runtime_engine": "context_exec", "context_exec_mode": "trace_autopsy"}])
def test_agent_mode_fails_fast_without_publishing(monkeypatch: pytest.MonkeyPatch, options: dict) -> None:
    monkeypatch.setattr("app.supervisor.run_recall_step", AsyncMock(return_value=_recall_ok()))
    supervisor = Supervisor(_NoPublishBus())
    ctx = {
        "mode": "agent",
        "messages": [{"role": "user", "content": "what breaks if we replace recall?"}],
        "options": options,
    }
    result = asyncio.run(
        supervisor.execute(
            source=ServiceRef(name="x", version="0", node="n"),
            req=_agent_plan(),
            correlation_id="00000000-0000-4000-8000-0000000000ce",
            ctx=ctx,
            recall_cfg={},
        )
    )
    step_names = [s.step_name for s in result.steps]
    assert "agent_runtime_unavailable" in step_names
    assert "context_exec" not in step_names
    # recall succeeded, the agent step failed: the plan rolls up as "partial", never "success".
    assert result.status != "success"
    assert "no depth-2 agent runtime" in (result.final_text or "").lower()


def test_context_exec_client_and_settings_are_gone() -> None:
    assert not hasattr(clients, "ContextExecClient")
    s = app_settings.settings
    for attr in ("context_exec_enabled", "context_exec_timeout_sec", "channel_context_exec_intake",
                 "channel_context_exec_reply_prefix", "context_exec_depth2_default"):
        assert not hasattr(s, attr), attr
    assert not hasattr(Supervisor, "_context_exec_escalation")
