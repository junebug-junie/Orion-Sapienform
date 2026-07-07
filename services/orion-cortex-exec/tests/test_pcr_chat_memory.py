from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from app.pcr_chat_memory import run_pcr_phase0_and_1, run_pcr_phase3
from app.settings import settings
from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.cortex.schemas import StepExecutionResult


@pytest.fixture(autouse=True)
def _enable_pcr(monkeypatch):
    monkeypatch.setattr(settings, "chat_pcr_enabled", True)
    monkeypatch.setattr(settings, "chat_pcr_skip_on_low_info", True)
    monkeypatch.setattr(settings, "chat_pcr_post_stance_recall", True)


def test_phase0_skips_recall_on_greeting(monkeypatch):
    recall_calls: list[dict] = []

    async def _fake_recall_step(*args, **kwargs):
        recall_calls.append(kwargs)
        return (
            StepExecutionResult(
                status="success",
                verb_name="chat_general",
                step_name="pcr_continuity_recall",
                order=-1,
                result={"RecallService": {"count": 1}},
                latency_ms=1,
                node="n",
                logs=[],
            ),
            {"count": 1},
            "continuity text",
        )

    monkeypatch.setattr("app.pcr_chat_memory.run_recall_step", _fake_recall_step)

    ctx = {
        "verb": "chat_general",
        "messages": [{"role": "user", "content": "hey Orion"}],
        "turn_change_appraisal": {"novelty_score": 0.1, "shift_kind": "NONE"},
    }

    pcr, recall_step, recall_debug = asyncio.run(
        run_pcr_phase0_and_1(
            object(),
            source=ServiceRef(name="x", version="0", node="n"),
            ctx=ctx,
            correlation_id="corr-greeting",
            recall_cfg={"enabled": True, "profile": "chat.general.v1"},
        )
    )

    assert len(recall_calls) == 0
    assert recall_step is None
    assert pcr.phase == "skip"
    assert pcr.retrieval_intent == "none"
    assert pcr.continuity_digest == ""
    assert pcr.belief_digest == ""
    assert "low_info_social" in pcr.skip_reasons
    assert ctx["continuity_digest"] == ""
    assert ctx["memory_digest"] == ""
    assert ctx["pcr_memory"] is pcr
    assert recall_debug.get("pcr_phase") == "skip"


def test_phase3_skipped_for_continuity_intent(monkeypatch):
    recall_calls: list[dict] = []

    async def _fake_recall_step(*args, **kwargs):
        recall_calls.append(kwargs)
        return (
            StepExecutionResult(
                status="success",
                verb_name="chat_general",
                step_name="pcr_belief_recall",
                order=-1,
                result={"RecallService": {"count": 1}},
                latency_ms=1,
                node="n",
                logs=[],
            ),
            {"count": 1},
            "belief text",
        )

    monkeypatch.setattr("app.pcr_chat_memory.run_recall_step", _fake_recall_step)

    ctx = {
        "verb": "chat_general",
        "user_message": "ok sounds good",
        "continuity_digest": "recent thread",
        "chat_stance_brief": {
            "task_mode": "direct_response",
            "interaction_regime": "instrumental",
        },
        "turn_change_appraisal": {"novelty_score": 0.1, "shift_kind": "NONE"},
    }

    pcr, recall_step, recall_debug = asyncio.run(
        run_pcr_phase3(
            object(),
            source=ServiceRef(name="x", version="0", node="n"),
            ctx=ctx,
            correlation_id="corr-continuity",
            recall_cfg={"enabled": True},
        )
    )

    assert len(recall_calls) == 0
    assert recall_step is None
    assert pcr.retrieval_intent == "continuity"
    assert pcr.belief_digest == ""
    assert pcr.memory_digest == "recent thread"
    assert recall_debug.get("pcr_phase") == "phase3_skipped"


def test_phase3_calls_recall_for_semantic_intent(monkeypatch):
    recall_calls: list[dict] = []

    async def _fake_recall_step(*args, **kwargs):
        recall_calls.append(kwargs)
        return (
            StepExecutionResult(
                status="success",
                verb_name="chat_general",
                step_name="pcr_belief_recall",
                order=-1,
                result={"RecallService": {"count": 1, "profile": kwargs.get("recall_profile")}},
                latency_ms=1,
                node="n",
                logs=[],
            ),
            {"count": 1, "profile": kwargs.get("recall_profile")},
            "approved belief",
        )

    monkeypatch.setattr("app.pcr_chat_memory.run_recall_step", _fake_recall_step)

    ctx = {
        "verb": "chat_general",
        "user_message": "what did we decide about the move?",
        "continuity_digest": "recent thread",
        "chat_stance_brief": {
            "task_mode": "instrumental",
            "interaction_regime": "instrumental",
        },
        "turn_change_appraisal": {"novelty_score": 0.72, "shift_kind": "TOPIC"},
    }

    pcr, recall_step, _ = asyncio.run(
        run_pcr_phase3(
            object(),
            source=ServiceRef(name="x", version="0", node="n"),
            ctx=ctx,
            correlation_id="corr-semantic",
            recall_cfg={"enabled": True},
        )
    )

    assert len(recall_calls) == 1
    assert recall_calls[0]["recall_phase"] == "purposeful"
    assert recall_calls[0]["retrieval_intent"] == "semantic"
    assert recall_calls[0]["recall_profile"] == "chat.belief.semantic.v1"
    assert recall_calls[0]["task_hints"]["rule_id"] == "topic_shift"
    assert recall_step is not None
    assert pcr.phase == "purposeful"
    assert pcr.retrieval_intent == "semantic"
    assert pcr.belief_digest == "approved belief"
    assert pcr.memory_digest == "recent thread\n\napproved belief"
    assert ctx["belief_digest"] == "approved belief"


def test_chat_pcr_disabled_uses_legacy_pre_recall(monkeypatch):
    from app.router import PlanRunner

    runner = PlanRunner()
    captured: dict[str, str] = {}
    recall_call_count = {"n": 0}

    async def _fake_recall_step(*args, **kwargs):
        recall_call_count["n"] += 1
        captured["profile"] = kwargs.get("recall_profile")
        captured["phase"] = kwargs.get("recall_phase")
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
            "legacy digest",
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

    monkeypatch.setattr(settings, "chat_pcr_enabled", False)
    monkeypatch.setattr("app.router.run_recall_step", _fake_recall_step)
    monkeypatch.setattr("app.router.call_step_services", AsyncMock(return_value=fake_step))

    plan = __import__(
        "orion.schemas.cortex.schemas",
        fromlist=[
            "ExecutionPlan",
            "ExecutionStep",
            "PlanExecutionArgs",
            "PlanExecutionRequest",
        ],
    )
    req = plan.PlanExecutionRequest(
        plan=plan.ExecutionPlan(
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
                plan.ExecutionStep(
                    verb_name="chat_general",
                    step_name="llm_chat_general",
                    description="",
                    order=0,
                    services=["LLMGatewayService"],
                    requires_memory=True,
                )
            ],
        ),
        args=plan.PlanExecutionArgs(
            request_id="r-legacy",
            extra={"mode": "brain", "recall": {"enabled": True, "profile": "chat.general.v1"}},
        ),
        context={"mode": "brain", "output_mode": "reflective_depth", "raw_user_text": "hello"},
    )

    result = asyncio.run(
        runner.run_plan(
            bus=object(),
            source=ServiceRef(name="x", version="0", node="n"),
            req=req,
            correlation_id="corr-legacy",
            ctx={"mode": "brain", "raw_user_text": "hello"},
        )
    )

    assert result.status == "success"
    assert recall_call_count["n"] == 1
    assert captured["profile"] == "chat.general.v1"
    assert captured.get("phase") is None
