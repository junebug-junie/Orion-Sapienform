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


def test_supervisor_pcr_phase3_after_stance(monkeypatch):
    from app.supervisor import Supervisor
    from orion.schemas.recall_pcr import PcrChatMemoryV1

    phase3_calls: list[dict] = []

    async def _fake_phase3(*args, **kwargs):
        phase3_calls.append(kwargs)
        pcr = PcrChatMemoryV1(
            phase="purposeful",
            retrieval_intent="semantic",
            continuity_digest="cont",
            belief_digest="belief",
            memory_digest="cont\n\nbelief",
        )
        return pcr, None, {"pcr_phase": "purposeful", "retrieval_intent": "semantic"}

    monkeypatch.setattr("app.supervisor.run_pcr_phase3", _fake_phase3)

    supervisor = Supervisor(object())
    ctx = {
        "chat_stance_brief": {"task_mode": "instrumental"},
        "continuity_digest": "cont",
        "turn_change_appraisal": {"shift_kind": "TOPIC", "novelty_score": 0.7},
        "messages": [{"role": "user", "content": "move logistics"}],
    }
    stance_step = StepExecutionResult(
        status="success",
        verb_name="chat_general",
        step_name="synthesize_chat_stance_brief",
        order=1,
        result={},
        latency_ms=1,
        node="n",
        logs=[],
    )
    step_results: list[StepExecutionResult] = []
    recall_debug: dict = {}

    ran = asyncio.run(
        supervisor._maybe_run_pcr_phase3_after_stance(
            source=ServiceRef(name="x", version="0", node="n"),
            ctx=ctx,
            correlation_id="corr-sup",
            recall_cfg={"enabled": True},
            verb_name="chat_general",
            stance_step=stance_step,
            step_results=step_results,
            recall_debug=recall_debug,
        )
    )

    assert ran is False  # no recall step returned (mock returns None step)
    assert len(phase3_calls) == 1
    assert phase3_calls[0]["correlation_id"] == "corr-sup"


# Live chat frames on 2026-10-06 always carried substrate prediction-error
# loops, and the old classifier turned any frame loop into open_loop: 630 of
# 630 purposeful recalls in 7 days used chat.belief.open_loop.v1.
_LIVE_FRAME = {"open_loops": [
    {"id": "open-loop-1", "target_type": "concept", "description": "Biometrics prediction error"},
]}


@pytest.mark.parametrize("signals, stance, profile, rule_id", [
    ([{"phrase": "Vincent", "type": "person", "confidence": 0.8}], {"task_mode": "direct_response"},
     "chat.belief.relational.v1", "turn_names_person"),
    ([{"phrase": "Hecate", "type": "concept"}], {"task_mode": "direct_response"},
     "chat.belief.semantic.v1", "turn_names_topic"),
    ([], {"task_mode": "technical_collaboration"}, "chat.belief.procedural.v1", "procedural_mode"),
    ([], {"task_mode": "direct_response"}, None, "continuity_only"),
])
def test_phase3_profile_follows_the_turn_not_the_frame(monkeypatch, signals, stance, profile, rule_id):
    """Same live-shaped frame, different turns -> different PCR recall (or none)."""
    recall_calls: list[dict] = []

    async def _fake_recall_step(*args, **kwargs):
        recall_calls.append(kwargs)
        return None, {"count": 0}, ""

    # Patch the globals run_pcr_phase3 actually reads: other tests in this
    # suite re-import `app.*`, so a dotted-string target can name a different
    # module object than the one imported at the top of this file.
    monkeypatch.setitem(run_pcr_phase3.__globals__, "run_recall_step", _fake_recall_step)
    ctx = {"verb": "chat_general", "user_message": "how is it going", "continuity_digest": "recent",
           "chat_stance_brief": stance, "chat_attention_frame": _LIVE_FRAME,
           "current_turn_llm_signals": signals}
    pcr, _, debug = asyncio.run(run_pcr_phase3(
        object(), source=ServiceRef(name="x", version="0", node="n"), ctx=ctx,
        correlation_id="corr-intent", recall_cfg={"enabled": True}))
    if profile is None:
        assert recall_calls == [] and pcr.phase == "continuity" and debug["rule_id"] == rule_id
    else:
        assert [c["recall_profile"] for c in recall_calls] == [profile], debug
        assert recall_calls[0]["task_hints"]["rule_id"] == rule_id
