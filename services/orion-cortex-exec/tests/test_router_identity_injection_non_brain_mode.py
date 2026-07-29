"""Regression: verbs that declare personality_file must get identity injected even when
dispatched with mode != "brain" (e.g. AI Town's embodiment worker dispatches chat_quick with
mode="quick" via EMBODIMENT_SPEECH_LANE=quick, per services/orion-embodiment/app/worker.py).

Before this fix, app/router.py only called prepare_chat_quick_reply_context /
prepare_brain_reply_context inside an `if mode == "brain"` gate, so chat_quick.yaml's declared
personality_file (orion/cognition/personality/orion_identity.yaml) never got loaded for any
non-brain-mode dispatch, and orion_identity_summary / juniper_relationship_summary /
response_policy_summary rendered empty in chat_quick.j2 for every AI Town speech turn.
"""

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

_IDENTITY_KEYS = ("orion_identity_summary", "juniper_relationship_summary", "response_policy_summary")


def _chat_quick_plan(*, mode: str) -> PlanExecutionRequest:
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
        metadata={
            "recall_profile": "chat.general.v1",
            "personality_file": "orion/cognition/personality/orion_identity.yaml",
        },
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
            request_id=f"rq-{mode}",
            extra={"mode": mode, "recall": {"enabled": False}},
        ),
        context={"mode": mode, "raw_user_text": "hi"},
    )


def _fake_llm_step() -> StepExecutionResult:
    return StepExecutionResult(
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


def _run(*, mode: str, monkeypatch, prepare_brain_reply_context_mock: AsyncMock | None = None) -> dict:
    monkeypatch.setattr(router, "call_step_services", AsyncMock(return_value=_fake_llm_step()))
    if prepare_brain_reply_context_mock is not None:
        monkeypatch.setattr(router, "prepare_brain_reply_context", prepare_brain_reply_context_mock)

    req = _chat_quick_plan(mode=mode)
    plan_metadata = req.plan.metadata if isinstance(req.plan.metadata, dict) else {}
    # Mirror the real ctx-building callers (app/main.py's exec handler and
    # app/verb_adapters.py's LegacyPlanVerb), both of which copy plan.metadata into
    # ctx["plan_metadata"]/ctx["personality_file"] before calling router.run_plan --
    # _inject_identity_context reads personality_file off ctx, not off plan.metadata directly.
    ctx: dict = {"mode": mode, "raw_user_text": "hi", "plan_metadata": plan_metadata}
    if "personality_file" in plan_metadata:
        ctx["personality_file"] = plan_metadata.get("personality_file")

    runner = PlanRunner()
    result = asyncio.run(
        runner.run_plan(
            bus=object(),
            source=ServiceRef(name="x", version="0", node="n"),
            req=req,
            correlation_id=f"corr-{mode}",
            ctx=ctx,
        )
    )
    assert result.status == "success"
    return ctx


def test_chat_quick_quick_mode_gets_identity_injected(monkeypatch) -> None:
    """AI Town's mode="quick" chat_quick dispatch must now receive real identity context."""
    ctx = _run(mode="quick", monkeypatch=monkeypatch)

    for key in _IDENTITY_KEYS:
        assert ctx.get(key), f"{key} should be populated for mode=quick chat_quick"
    assert ctx.get("identity_kernel_source") == "configured_yaml"


def test_chat_quick_brain_mode_unchanged(monkeypatch) -> None:
    """Existing mode="brain" chat_quick behavior (prepare_chat_quick_reply_context) is untouched."""
    prepare_brain_reply_context_mock = AsyncMock(return_value=None)
    ctx = _run(mode="brain", monkeypatch=monkeypatch, prepare_brain_reply_context_mock=prepare_brain_reply_context_mock)

    # brain-mode chat_quick (without chat_quick_full_stance) still goes through
    # prepare_chat_quick_reply_context directly, never prepare_brain_reply_context.
    prepare_brain_reply_context_mock.assert_not_called()
    for key in _IDENTITY_KEYS:
        assert ctx.get(key), f"{key} should still be populated for mode=brain chat_quick"
    assert ctx.get("identity_kernel_source") == "configured_yaml"
