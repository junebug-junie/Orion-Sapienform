from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from app.router import PlanRunner, _last_model_used
from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.cortex.schemas import (
    ExecutionPlan,
    ExecutionStep,
    PlanExecutionArgs,
    PlanExecutionRequest,
    StepExecutionResult,
)


def _request(*, verb_name: str = "chat_general", step_name: str = "llm_chat_general") -> PlanExecutionRequest:
    plan = ExecutionPlan(
        verb_name=verb_name,
        label=verb_name,
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
                verb_name=verb_name,
                step_name=step_name,
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


class TestLastModelUsed:
    """Unit coverage for the helper itself -- independent of reasoning_content,
    unlike _collect_metacog_traces (confirmed live: MetacognitiveTraceV1.
    model reads 'unknown' on essentially every real turn because that
    collector only fires when reasoning_content is non-empty)."""

    def test_reads_model_used_key(self) -> None:
        steps = [
            StepExecutionResult(
                status="success",
                verb_name="chat_general",
                step_name="llm_chat_general",
                order=0,
                result={"LLMGatewayService": {"content": "hi", "model_used": "qwen-36-instruct"}},
                latency_ms=1,
                node="n",
                logs=[],
                error=None,
            )
        ]
        assert _last_model_used(steps) == "qwen-36-instruct"

    def test_falls_back_to_model_key(self) -> None:
        steps = [
            StepExecutionResult(
                status="success",
                verb_name="chat_general",
                step_name="llm_chat_general",
                order=0,
                result={"LLMGatewayService": {"content": "hi", "model": "qwen-36-instruct"}},
                latency_ms=1,
                node="n",
                logs=[],
                error=None,
            )
        ]
        assert _last_model_used(steps) == "qwen-36-instruct"

    def test_takes_the_last_step_with_a_model(self) -> None:
        steps = [
            StepExecutionResult(
                status="success", verb_name="v", step_name="s1", order=0,
                result={"LLMGatewayService": {"model_used": "first-model"}},
                latency_ms=1, node="n", logs=[], error=None,
            ),
            StepExecutionResult(
                status="success", verb_name="v", step_name="s2", order=1,
                result={"LLMGatewayService": {"model_used": "second-model"}},
                latency_ms=1, node="n", logs=[], error=None,
            ),
        ]
        assert _last_model_used(steps) == "second-model"

    def test_none_when_no_llm_gateway_step(self) -> None:
        steps = [
            StepExecutionResult(
                status="success", verb_name="v", step_name="s1", order=0,
                result={"SomeOtherService": {"ok": True}},
                latency_ms=1, node="n", logs=[], error=None,
            )
        ]
        assert _last_model_used(steps) is None

    def test_none_on_empty_steps(self) -> None:
        assert _last_model_used([]) is None


def test_router_exports_model_in_metadata(monkeypatch) -> None:
    """Regression guard: metadata['model'] must actually be populated by
    run_plan, not just computable by the helper in isolation -- Hub's
    websocket_handler.py already reads gateway_meta.get('model') off this
    exact field at several chat_history call sites; before this fix nothing
    ever set it, so those reads were always None."""
    runner = PlanRunner()
    fake_step = StepExecutionResult(
        status="success",
        verb_name="chat_general",
        step_name="llm_chat_general",
        order=0,
        result={"LLMGatewayService": {"content": "ok", "model_used": "qwen-36-instruct"}},
        latency_ms=1,
        node="n",
        logs=[],
        error=None,
    )
    monkeypatch.setattr("app.router.call_step_services", AsyncMock(return_value=fake_step))
    monkeypatch.setattr("app.router.prepare_brain_reply_context", AsyncMock(return_value=None))
    result = asyncio.run(
        runner.run_plan(
            bus=object(),
            source=ServiceRef(name="x", version="0", node="n"),
            req=_request(),
            correlation_id="corr-model-export",
            ctx={"mode": "brain", "raw_user_text": "hello"},
        )
    )
    assert result.metadata["model"] == "qwen-36-instruct"


def test_router_omits_model_when_no_llm_gateway_step(monkeypatch) -> None:
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
    monkeypatch.setattr("app.router.prepare_brain_reply_context", AsyncMock(return_value=None))
    result = asyncio.run(
        runner.run_plan(
            bus=object(),
            source=ServiceRef(name="x", version="0", node="n"),
            req=_request(),
            correlation_id="corr-model-absent",
            ctx={"mode": "brain", "raw_user_text": "hello"},
        )
    )
    assert "model" not in result.metadata
