from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.verb_adapters import LegacyPlanVerb
from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.cortex.schemas import ExecutionPlan, ExecutionStep, PlanExecutionArgs, PlanExecutionRequest, PlanExecutionResult


def _request() -> PlanExecutionRequest:
    return PlanExecutionRequest(
        plan=ExecutionPlan(
            verb_name="chat_general",
            label="chat_general",
            steps=[ExecutionStep(verb_name="chat_general", step_name="llm_chat_general", order=0, services=["LLMService"])],
            metadata={"mode": "brain", "personality_file": "orion/cognition/personality/orion_identity.yaml"},
        ),
        args=PlanExecutionArgs(request_id="rid-1", extra={"mode": "brain"}),
        context={"messages": [{"role": "user", "content": "hello"}]},
    )


def test_legacy_plan_exec_context_receives_personality_file(monkeypatch) -> None:
    captured: dict = {}

    async def _fake_self_study_context(**_kwargs):
        return SimpleNamespace(used=False, consulted=False, consumer_name="legacy.plan", consumer_kind="internal", retrieval_mode=None, policy_reason="disabled", policy_decision=SimpleNamespace(model_dump=lambda mode="json": {}), notes=[], rendered="", result=None)

    async def _fake_run_plan(self, bus, *, source, req, correlation_id, ctx):
        captured.update(ctx)
        return PlanExecutionResult(verb_name=req.plan.verb_name, request_id=req.args.request_id, status="success")

    monkeypatch.setattr("app.verb_adapters._resolve_self_study_context", _fake_self_study_context)
    monkeypatch.setattr("app.verb_adapters._self_study_payload", lambda _ctx: {"rendered": ""})
    monkeypatch.setattr("app.verb_adapters.PlanRouter.run_plan", _fake_run_plan)

    verb = LegacyPlanVerb()
    ctx = SimpleNamespace(meta={"bus": object(), "source": ServiceRef(name="exec", version="0", node="n"), "correlation_id": "corr-1"})
    asyncio.run(verb.execute(ctx, _request()))

    assert captured.get("plan_metadata", {}).get("personality_file") == "orion/cognition/personality/orion_identity.yaml"
    assert captured.get("personality_file") == "orion/cognition/personality/orion_identity.yaml"
