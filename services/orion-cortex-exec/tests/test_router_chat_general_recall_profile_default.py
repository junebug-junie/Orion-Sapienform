from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from app.router import PlanRunner
from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.cortex.schemas import (
    ExecutionPlan,
    ExecutionStep,
    PlanExecutionArgs,
    PlanExecutionRequest,
    StepExecutionResult,
)


def _request_with_inherited_reflect_profile() -> PlanExecutionRequest:
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
        metadata={"recall_profile": "chat.general.v1", "mode": "brain"},
        steps=[
            ExecutionStep(
                verb_name="chat_general",
                step_name="llm_chat_general",
                description="",
                order=0,
                services=["LLMGatewayService"],
                requires_memory=True,
            )
        ],
    )
    return PlanExecutionRequest(
        plan=plan,
        args=PlanExecutionArgs(
            request_id="r1",
            extra={"mode": "brain", "recall": {"enabled": True, "profile": "reflect.v1"}},
        ),
        context={"mode": "brain", "output_mode": "reflective_depth", "raw_user_text": "hello"},
    )


def test_router_prefers_chat_general_verb_profile_by_default(monkeypatch):
    runner = PlanRunner()
    captured: dict[str, str] = {}

    async def _fake_recall_step(*args, **kwargs):
        captured["profile"] = kwargs.get("recall_profile")
        return (
            StepExecutionResult(
                status="success",
                verb_name="chat_general",
                step_name="recall",
                order=-1,
                result={"RecallService": {"count": 1, "profile": kwargs.get("recall_profile")}},
                latency_ms=1,
                node="n",
                logs=[],
                error=None,
            ),
            {"count": 1, "profile": kwargs.get("recall_profile")},
            "",
        )

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

    monkeypatch.setattr("app.router.run_recall_step", _fake_recall_step)
    monkeypatch.setattr("app.router.call_step_services", AsyncMock(return_value=fake_step))

    req = _request_with_inherited_reflect_profile()
    result = asyncio.run(
        runner.run_plan(
            bus=object(),
            source=ServiceRef(name="x", version="0", node="n"),
            req=req,
            correlation_id="corr-1",
            ctx={"mode": "brain", "raw_user_text": "hello"},
        )
    )

    assert result.status == "success"
    assert captured["profile"] == "chat.general.v1"
    assert result.recall_debug.get("profile_source") == "verb"


def test_router_allows_intentional_explicit_override(monkeypatch):
    runner = PlanRunner()
    captured: dict[str, str] = {}

    async def _fake_recall_step(*args, **kwargs):
        captured["profile"] = kwargs.get("recall_profile")
        return (
            StepExecutionResult(
                status="success",
                verb_name="chat_general",
                step_name="recall",
                order=-1,
                result={"RecallService": {"count": 1, "profile": kwargs.get("recall_profile")}},
                latency_ms=1,
                node="n",
                logs=[],
                error=None,
            ),
            {"count": 1, "profile": kwargs.get("recall_profile")},
            "",
        )

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

    monkeypatch.setattr("app.router.run_recall_step", _fake_recall_step)
    monkeypatch.setattr("app.router.call_step_services", AsyncMock(return_value=fake_step))

    req = _request_with_inherited_reflect_profile()
    req.args.extra["recall"]["profile"] = "reflect.v1"
    req.args.extra["recall"]["profile_explicit"] = True

    result = asyncio.run(
        runner.run_plan(
            bus=object(),
            source=ServiceRef(name="x", version="0", node="n"),
            req=req,
            correlation_id="corr-2",
            ctx={"mode": "brain", "raw_user_text": "hello"},
        )
    )

    assert result.status == "success"
    assert captured["profile"] == "reflect.v1"
    assert result.recall_debug.get("profile_override_source") == "explicit"
