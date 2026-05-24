from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from app.executor import _forward_llm_uncertainty_metadata
from app.router import PlanRunner, _autonomy_payload_from_ctx
from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.cortex.schemas import ExecutionPlan, ExecutionStep, PlanExecutionArgs, PlanExecutionRequest, StepExecutionResult


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


def test_forward_llm_uncertainty_metadata_from_gateway_payload() -> None:
    ctx: dict = {}
    payload = {
        "content": "hello",
        "meta": {"llm_uncertainty": {"schema_version": "v1", "available": True, "mean_logprob": -0.4}},
    }
    _forward_llm_uncertainty_metadata(payload, ctx)
    assert ctx["metadata"]["llm_uncertainty"]["available"] is True


def test_autonomy_payload_exports_llm_uncertainty_from_ctx_metadata() -> None:
    md = _autonomy_payload_from_ctx(
        {
            "metadata": {
                "llm_uncertainty": {"schema_version": "v1", "available": True},
            }
        }
    )
    assert md["llm_uncertainty"]["available"] is True


def test_plan_result_metadata_includes_llm_uncertainty(monkeypatch) -> None:
    runner = PlanRunner()
    fake_step = StepExecutionResult(
        status="success",
        verb_name="chat_general",
        step_name="llm_chat_general",
        order=0,
        result={
            "LLMGatewayService": {
                "content": "ok",
                "meta": {"llm_uncertainty": {"schema_version": "v1", "available": True}},
            }
        },
        latency_ms=1,
        node="n",
        logs=[],
        error=None,
    )
    monkeypatch.setattr("app.router.call_step_services", AsyncMock(return_value=fake_step))
    monkeypatch.setattr("app.router.prepare_brain_reply_context", lambda _ctx: None)
    req = _request()
    result = asyncio.run(
        runner.run_plan(
            bus=object(),
            source=ServiceRef(name="x", version="0", node="n"),
            req=req,
            correlation_id="corr-llm-uncertainty",
            ctx={"mode": "brain", "raw_user_text": "hello"},
        )
    )
    assert result.metadata["llm_uncertainty"]["available"] is True
    gw_result = fake_step.result["LLMGatewayService"]
    assert isinstance(gw_result.get("meta"), dict)
    assert gw_result["meta"]["llm_uncertainty"]["available"] is True
