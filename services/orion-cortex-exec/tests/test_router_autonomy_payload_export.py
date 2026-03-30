from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from app.router import PlanRunner
from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.cortex.schemas import ExecutionPlan, ExecutionStep, PlanExecutionArgs, PlanExecutionRequest, StepExecutionResult


def _request() -> PlanExecutionRequest:
    plan = ExecutionPlan(
        verb_name="chat_general",
        label="chat_general",
        description="",
        category="x",
        priority="normal",
        interruptible=True,
        can_interrupt_others=False,
        timeout_ms=1000,
        max_recursion_depth=1,
        metadata={"mode": "brain"},
        steps=[
            ExecutionStep(
                verb_name="chat_general",
                step_name="llm_chat_general",
                description="",
                order=0,
                services=["LLMGatewayService"],
                requires_memory=False,
            )
        ],
    )
    return PlanExecutionRequest(
        plan=plan,
        args=PlanExecutionArgs(request_id="r1", extra={"mode": "brain"}),
        context={"mode": "brain", "raw_user_text": "hello"},
    )


def test_router_exports_autonomy_metadata_from_chat_stance_context(monkeypatch) -> None:
    runner = PlanRunner()
    fake_step = StepExecutionResult(
        status="success",
        verb_name="chat_general",
        step_name="llm_chat_general",
        order=0,
        result={"LLMGatewayService": {"content": "ok"}},
        latency_ms=1,
        node="n",
        logs=[],
        error=None,
    )
    monkeypatch.setattr("app.router.call_step_services", AsyncMock(return_value=fake_step))
    req = _request()
    ctx = {
        "mode": "brain",
        "raw_user_text": "hello",
        "chat_autonomy_summary": {
            "stance_hint": "favor synthesis and reduction",
            "top_drives": ["coherence"],
            "active_tensions": ["scope_sprawl"],
            "proposal_headlines": ["stabilize triage sequence"],
        },
        "chat_autonomy_debug": {"orion": {"availability": "available", "present": True}},
        "chat_autonomy_state": {
            "subject": "orion",
            "source": "graph",
            "dominant_drive": "coherence",
            "active_drives": ["coherence"],
            "tension_kinds": ["scope_sprawl"],
        },
    }
    result = asyncio.run(
        runner.run_plan(
            bus=object(),
            source=ServiceRef(name="x", version="0", node="n"),
            req=req,
            correlation_id="corr-a1",
            ctx=ctx,
        )
    )
    assert result.metadata["autonomy_summary"]["stance_hint"] == "favor synthesis and reduction"
    assert result.metadata["autonomy_debug"]["orion"]["availability"] == "available"
    assert result.metadata["autonomy_state_preview"]["dominant_drive"] == "coherence"
    assert result.metadata["autonomy_backend"] == "graph"
    assert result.metadata["autonomy_selected_subject"] == "orion"


def test_router_omits_autonomy_metadata_when_absent(monkeypatch) -> None:
    runner = PlanRunner()
    fake_step = StepExecutionResult(
        status="success",
        verb_name="chat_general",
        step_name="llm_chat_general",
        order=0,
        result={"LLMGatewayService": {"content": "ok"}},
        latency_ms=1,
        node="n",
        logs=[],
        error=None,
    )
    monkeypatch.setattr("app.router.call_step_services", AsyncMock(return_value=fake_step))
    req = _request()
    result = asyncio.run(
        runner.run_plan(
            bus=object(),
            source=ServiceRef(name="x", version="0", node="n"),
            req=req,
            correlation_id="corr-a2",
            ctx={"mode": "brain", "raw_user_text": "hello"},
        )
    )
    assert result.metadata == {}
