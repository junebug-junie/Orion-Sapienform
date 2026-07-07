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
from orion.schemas.thought import GroundingCapsuleV1


def _request(*, verb_name: str, step_name: str) -> PlanExecutionRequest:
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
        context={"mode": "brain", "raw_user_text": "how are you?"},
    )


def _fake_step(*, verb_name: str, step_name: str) -> StepExecutionResult:
    return StepExecutionResult(
        status="success",
        verb_name=verb_name,
        step_name=step_name,
        order=0,
        result={"LLMGatewayService": {"content": '{"imperative":"Stay present.","tone":"warm"}'}},
        latency_ms=1,
        node="n",
        logs=[],
        error=None,
    )


def _capsule() -> GroundingCapsuleV1:
    return GroundingCapsuleV1(
        identity_summary=["I am Oríon."],
        relationship_summary=["Juniper is my collaborator."],
        response_policy_summary=["Speak plainly."],
        continuity_digest="We were mid-refactor.",
        belief_digest="Orion values continuity.",
        memory_digest="We were mid-refactor.",
        provenance={"identity_source": "configured_yaml", "pcr_ran": True},
    )


def test_router_attaches_grounding_capsule_to_metadata(monkeypatch) -> None:
    """stance_react turn: the router assembles the capsule and rides it on result metadata."""
    runner = PlanRunner()
    monkeypatch.setattr(
        "app.router.call_step_services",
        AsyncMock(return_value=_fake_step(verb_name="stance_react", step_name="llm_stance_react")),
    )
    monkeypatch.setattr("app.router.prepare_brain_reply_context", lambda _ctx: None)
    assemble = AsyncMock(return_value=_capsule())
    monkeypatch.setattr("app.router.assemble_stance_grounding", assemble)

    result = asyncio.run(
        runner.run_plan(
            bus=object(),
            source=ServiceRef(name="x", version="0", node="n"),
            req=_request(verb_name="stance_react", step_name="llm_stance_react"),
            correlation_id="corr-grounding",
            ctx={"mode": "brain", "raw_user_text": "how are you?"},
        )
    )

    assert assemble.await_count == 1
    capsule_md = result.metadata["grounding_capsule"]
    assert capsule_md["identity_summary"] == ["I am Oríon."]
    assert capsule_md["provenance"]["pcr_ran"] is True


def test_router_omits_grounding_capsule_when_assembler_returns_none(monkeypatch) -> None:
    """Flag-off / degraded assembly returns None => router leaves metadata untouched (no-op)."""
    runner = PlanRunner()
    monkeypatch.setattr(
        "app.router.call_step_services",
        AsyncMock(return_value=_fake_step(verb_name="stance_react", step_name="llm_stance_react")),
    )
    monkeypatch.setattr("app.router.prepare_brain_reply_context", lambda _ctx: None)
    monkeypatch.setattr("app.router.assemble_stance_grounding", AsyncMock(return_value=None))

    result = asyncio.run(
        runner.run_plan(
            bus=object(),
            source=ServiceRef(name="x", version="0", node="n"),
            req=_request(verb_name="stance_react", step_name="llm_stance_react"),
            correlation_id="corr-grounding-none",
            ctx={"mode": "brain", "raw_user_text": "how are you?"},
        )
    )

    assert "grounding_capsule" not in result.metadata


def test_router_does_not_assemble_grounding_for_non_stance_verb(monkeypatch) -> None:
    """Only stance_react turns assemble the capsule; other verbs never call the assembler."""
    runner = PlanRunner()
    monkeypatch.setattr(
        "app.router.call_step_services",
        AsyncMock(return_value=_fake_step(verb_name="chat_general", step_name="llm_chat_general")),
    )
    monkeypatch.setattr("app.router.prepare_brain_reply_context", lambda _ctx: None)
    assemble = AsyncMock(return_value=_capsule())
    monkeypatch.setattr("app.router.assemble_stance_grounding", assemble)

    result = asyncio.run(
        runner.run_plan(
            bus=object(),
            source=ServiceRef(name="x", version="0", node="n"),
            req=_request(verb_name="chat_general", step_name="llm_chat_general"),
            correlation_id="corr-grounding-nonstance",
            ctx={"mode": "brain", "raw_user_text": "how are you?"},
        )
    )

    assert assemble.await_count == 0
    assert "grounding_capsule" not in result.metadata
