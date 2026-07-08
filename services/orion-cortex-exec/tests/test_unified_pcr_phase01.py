from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from app.router import PlanRunner
from app.settings import settings
from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.cortex.schemas import (
    ExecutionPlan,
    ExecutionStep,
    PlanExecutionArgs,
    PlanExecutionRequest,
    StepExecutionResult,
)
from orion.schemas.recall_pcr import PcrChatMemoryV1
from orion.schemas.thought import GroundingCapsuleV1


def _stance_react_request(*, recall_enabled: bool = True) -> PlanExecutionRequest:
    plan = ExecutionPlan(
        verb_name="stance_react",
        label="stance_react",
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
                verb_name="stance_react",
                step_name="llm_stance_react",
                description="",
                order=0,
                services=["LLMGatewayService"],
                requires_memory=False,
            )
        ],
    )
    return PlanExecutionRequest(
        plan=plan,
        args=PlanExecutionArgs(
            request_id="r-stance",
            extra={"mode": "brain", "recall": {"enabled": recall_enabled}},
        ),
        context={
            "mode": "brain",
            "user_message": "what did we decide about the move?",
            "raw_user_text": "what did we decide about the move?",
            "surface_context": {"hub_chat_lane": "orion"},
        },
    )


def _stance_step_result() -> StepExecutionResult:
    return StepExecutionResult(
        status="success",
        verb_name="stance_react",
        step_name="llm_stance_react",
        order=0,
        result={
            "LLMGatewayService": {
                "content": (
                    '{"imperative":"Stay present.","tone":"warm",'
                    '"stance_harness_slice":{"task_mode":"reflective_dialogue",'
                    '"conversation_frame":"reflective"}}'
                )
            }
        },
        latency_ms=1,
        node="n",
        logs=[],
        error=None,
    )


@pytest.fixture(autouse=True)
def _enable_pcr_and_grounding(monkeypatch):
    monkeypatch.setattr(settings, "chat_pcr_enabled", True)
    monkeypatch.setattr(settings, "orion_unified_grounding_enabled", True)
    monkeypatch.setattr(settings, "chat_pcr_post_stance_recall", True)


@pytest.mark.asyncio
async def test_stance_react_runs_pcr_phase01_before_stance_step(monkeypatch) -> None:
    """Unified turn: continuity recall runs before llm_stance_react when PCR is enabled."""
    runner = PlanRunner()
    call_order: list[str] = []

    async def _fake_phase01(_bus, *, ctx, **_kwargs):
        call_order.append("phase01")
        ctx["continuity_digest"] = "recent thread continuity"
        pcr = PcrChatMemoryV1(
            phase="continuity",
            retrieval_intent="continuity",
            continuity_digest="recent thread continuity",
            belief_digest="",
            memory_digest="recent thread continuity",
            skip_reasons=[],
            recall_debug={"count": 1},
        )
        ctx["pcr_memory"] = pcr
        recall_step = StepExecutionResult(
            status="success",
            verb_name="recall",
            step_name="pcr_continuity_recall",
            order=-1,
            result={"RecallService": {"count": 1}},
            latency_ms=1,
            node="n",
            logs=[],
        )
        return pcr, recall_step, {"count": 1, "profile": "chat.continuity.v1"}

    async def _fake_call_step(*_args, **_kwargs):
        call_order.append("stance_step")
        return _stance_step_result()

    monkeypatch.setattr("app.router.run_pcr_phase0_and_1", _fake_phase01)
    monkeypatch.setattr("app.router.call_step_services", _fake_call_step)
    monkeypatch.setattr("app.router.prepare_brain_reply_context", lambda _ctx: None)
    monkeypatch.setattr(
        "app.router.assemble_stance_grounding",
        AsyncMock(
            return_value=GroundingCapsuleV1(
                identity_summary=["I am Oríon."],
                provenance={"identity_source": "configured_yaml", "pcr_ran": True},
            )
        ),
    )

    ctx = {
        "mode": "brain",
        "user_message": "what did we decide about the move?",
        "raw_user_text": "what did we decide about the move?",
        "surface_context": {"hub_chat_lane": "orion"},
    }
    result = await runner.run_plan(
        bus=object(),
        source=ServiceRef(name="x", version="0", node="n"),
        req=_stance_react_request(),
        correlation_id="corr-unified-pcr01",
        ctx=ctx,
    )

    assert result.status == "success"
    assert call_order == ["phase01", "stance_step"]
    assert ctx.get("continuity_digest") == "recent thread continuity"


@pytest.mark.asyncio
async def test_assemble_stance_grounding_skips_duplicate_phase01_when_router_ran(monkeypatch) -> None:
    """Grounding assembly runs phase3 only when router already completed phase 0+1."""
    from app import grounding_capsule as gc
    from app.settings import Settings

    phase01_calls: list[str] = []
    phase3_calls: list[str] = []

    async def _fake_phase01(*_args, **_kwargs):
        phase01_calls.append("phase01")
        return None, None, {}

    async def _fake_phase3(*_args, **_kwargs):
        phase3_calls.append("phase3")
        return None, None, {}

    monkeypatch.setattr(gc, "run_pcr_phase0_and_1", _fake_phase01)
    monkeypatch.setattr(gc, "run_pcr_phase3", _fake_phase3)
    cfg = Settings(ORION_UNIFIED_GROUNDING_ENABLED=True, CHAT_PCR_ENABLED=True)
    ctx: dict = {
        "orion_identity_summary": ["I am Oríon."],
        "identity_kernel_source": "configured_yaml",
        "continuity_digest": "already from router",
        "pcr_memory": PcrChatMemoryV1(
            phase="continuity",
            retrieval_intent="continuity",
            continuity_digest="already from router",
            belief_digest="",
            memory_digest="already from router",
            skip_reasons=[],
            recall_debug={},
        ),
    }

    capsule = await gc.assemble_stance_grounding(
        bus=None,
        source=None,
        ctx=ctx,
        correlation_id="corr-no-dup",
        recall_cfg={"enabled": True},
        stance_step_text='{"stance_harness_slice":{"task_mode":"reflective_dialogue","conversation_frame":"reflective"}}',
        exec_settings=cfg,
    )

    assert capsule is not None
    assert phase01_calls == []
    assert phase3_calls == ["phase3"]
    assert capsule.provenance["pcr_ran"] is True


def test_pcr_phase01_complete_detects_router_continuity() -> None:
    from app.pcr_chat_memory import pcr_phase01_complete

    ctx = {
        "continuity_digest": "router continuity",
        "pcr_memory": PcrChatMemoryV1(
            phase="continuity",
            retrieval_intent="continuity",
            continuity_digest="router continuity",
            belief_digest="",
            memory_digest="router continuity",
            skip_reasons=[],
            recall_debug={},
        ),
    }
    assert pcr_phase01_complete(ctx) is True
