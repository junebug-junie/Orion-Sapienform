"""Contract tests: chat_quick uses a dedicated Recall bus RPC cap (both Quick UI variants share this verb)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from app import router
from app.router import PlanRunner
from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.cortex.schemas import (
    ExecutionPlan,
    ExecutionStep,
    PlanExecutionArgs,
    PlanExecutionRequest,
    StepExecutionResult,
)


def _chat_quick_plan() -> PlanExecutionRequest:
    plan = ExecutionPlan(
        verb_name="chat_quick",
        label="chat_quick",
        description="",
        category="x",
        priority="normal",
        interruptible=True,
        can_interrupt_others=False,
        timeout_ms=1000,
        max_recursion_depth=0,
        metadata={"recall_profile": "chat.general.v1", "mode": "brain"},
        steps=[
            ExecutionStep(
                verb_name="chat_quick",
                step_name="llm_chat_quick",
                description="",
                order=1,
                services=["LLMGatewayService"],
                requires_memory=True,
            )
        ],
    )
    return PlanExecutionRequest(
        plan=plan,
        args=PlanExecutionArgs(
            request_id="rq-quick",
            extra={"mode": "brain", "recall": {"enabled": True, "profile": "reflect.v1"}},
        ),
        context={"mode": "brain", "raw_user_text": "hi"},
    )


def _chat_general_plan() -> PlanExecutionRequest:
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
            request_id="rq-gen",
            extra={"mode": "brain", "recall": {"enabled": True, "profile": "reflect.v1"}},
        ),
        context={"mode": "brain", "raw_user_text": "hello"},
    )


def test_router_passes_none_recall_rpc_override_chat_quick(monkeypatch) -> None:
    """PlanRunner defers Recall bus wait to run_recall_step (min of STEP_TIMEOUT_MS and CHAT_QUICK_RECALL_TIMEOUT_SEC for chat_quick)."""
    captured: dict[str, float | None] = {"rpc_timeout_sec": None}

    async def _fake_recall_step(*args, **kwargs):
        captured["rpc_timeout_sec"] = kwargs.get("rpc_timeout_sec")
        return (
            StepExecutionResult(
                status="success",
                verb_name="chat_quick",
                step_name="recall",
                order=-1,
                result={"RecallService": {"count": 1}},
                latency_ms=1,
                node="n",
                logs=[],
                error=None,
            ),
            {"count": 1},
            "digest",
        )

    fake_llm = StepExecutionResult(
        status="success",
        verb_name="chat_quick",
        step_name="llm_chat_quick",
        order=1,
        result={"LLMGatewayService": {"content": "ok"}},
        latency_ms=1,
        node="n",
        logs=[],
        error=None,
    )

    monkeypatch.setattr(router, "run_recall_step", _fake_recall_step)
    monkeypatch.setattr(router, "call_step_services", AsyncMock(return_value=fake_llm))
    monkeypatch.setattr(router, "prepare_chat_quick_reply_context", lambda ctx: None)
    monkeypatch.setattr(router, "prepare_brain_reply_context", AsyncMock(return_value=None))

    runner = PlanRunner()
    result = asyncio.run(
        runner.run_plan(
            bus=object(),
            source=ServiceRef(name="x", version="0", node="n"),
            req=_chat_quick_plan(),
            correlation_id="corr-q",
            ctx={"mode": "brain", "raw_user_text": "hi"},
        )
    )

    assert result.status == "success"
    assert captured["rpc_timeout_sec"] is None


def test_router_passes_none_rpc_timeout_for_chat_general(monkeypatch) -> None:
    captured: dict[str, float | None] = {"rpc_timeout_sec": -1.0}

    async def _fake_recall_step(*args, **kwargs):
        captured["rpc_timeout_sec"] = kwargs.get("rpc_timeout_sec")
        return (
            StepExecutionResult(
                status="success",
                verb_name="chat_general",
                step_name="recall",
                order=-1,
                result={"RecallService": {"count": 1}},
                latency_ms=1,
                node="n",
                logs=[],
                error=None,
            ),
            {"count": 1},
            "",
        )

    fake_llm = StepExecutionResult(
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

    monkeypatch.setattr(router, "run_recall_step", _fake_recall_step)
    monkeypatch.setattr(router, "call_step_services", AsyncMock(return_value=fake_llm))
    monkeypatch.setattr(router, "prepare_brain_reply_context", AsyncMock(return_value=None))

    runner = PlanRunner()
    result = asyncio.run(
        runner.run_plan(
            bus=object(),
            source=ServiceRef(name="x", version="0", node="n"),
            req=_chat_general_plan(),
            correlation_id="corr-g",
            ctx={"mode": "brain", "raw_user_text": "hello"},
        )
    )

    assert result.status == "success"
    assert captured["rpc_timeout_sec"] is None
