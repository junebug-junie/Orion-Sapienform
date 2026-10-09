"""Phase 3 of the recall retrieval design (2026-09-29): what cortex-exec asks recall to search for.

Every cortex-exec recall goes through ``run_recall_step``; these tests capture the
RecallQueryV1 it actually sends (fake RecallClient, real run_recall_step) on each path:
direct, PCR phase 0+1 -> phase 3, stance_react grounding, and the router pre-recall for
reverie verbs.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from app import executor, router
from app.grounding_capsule import assemble_stance_grounding
from app.pcr_chat_memory import PCR_RETRIEVAL_QUERY_KEY, run_pcr_phase0_and_1, run_pcr_phase3
from app.router import PlanRunner
from app.settings import settings
from orion.cognition.plan_loader import build_plan_for_verb
from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.cortex.schemas import PlanExecutionArgs, PlanExecutionRequest, StepExecutionResult

SOURCE = ServiceRef(name="test", version="0", node="n")
STANDING_QUESTION = "What do I actually remember about the GPU pool cutover?"


class _CapturingRecallClient:
    sent: list = []
    timeouts: list = []

    def __init__(self, bus):
        self.bus = bus

    async def query(self, *, source, req, correlation_id, reply_to, timeout_sec):
        _CapturingRecallClient.sent.append(req)
        _CapturingRecallClient.timeouts.append(timeout_sec)

        class _Bundle:
            items: list = []
            rendered = "recalled"

            def model_dump(self, mode="json"):
                return {"items": [], "rendered": "recalled", "stats": {}}

        class _Res:
            bundle = _Bundle()
            debug: dict = {}

        return _Res()


@pytest.fixture
def capture(monkeypatch):
    _CapturingRecallClient.sent = []
    _CapturingRecallClient.timeouts = []
    monkeypatch.setattr(executor, "RecallClient", _CapturingRecallClient)
    return _CapturingRecallClient


def _run(ctx, *, recall_cfg=None, **kwargs):
    return asyncio.run(
        executor.run_recall_step(
            bus=object(),
            source=SOURCE,
            ctx=ctx,
            correlation_id="corr-1",
            recall_cfg=recall_cfg or {},
            recall_profile="reflect.v1",
            **kwargs,
        )
    )


def test_run_recall_step_sends_caller_retrieval_query_and_keeps_fragment(capture, monkeypatch):
    monkeypatch.setattr(executor.settings, "step_timeout_ms", 45_000)
    long_prompt = "You are Orion. Investigate. " * 1200  # the 30k-char instruction prompt shape
    ctx = {"verb": "stance_react", "user_message": long_prompt, "retrieval_query": f"  {STANDING_QUESTION}  "}

    step, _, _ = _run(ctx)

    assert step.status == "success"
    req = capture.sent[-1]
    assert req.retrieval_query == STANDING_QUESTION
    assert req.fragment == long_prompt  # provenance: still the turn text
    assert req.exclude["active_turn_text"] == long_prompt
    assert req.mode == "retrieve"
    # deadline_ms is the RPC wait actually used, in ms.
    assert req.deadline_ms == int(round(capture.timeouts[-1] * 1000))
    assert req.deadline_ms > 0


def test_run_recall_step_without_caller_query_leaves_it_to_recall(capture):
    step, _, _ = _run({"verb": "chat_general", "messages": [{"role": "user", "content": "hi there"}]})

    assert step.status == "success"
    req = capture.sent[-1]
    assert req.retrieval_query is None
    assert req.fragment == "hi there"
    assert req.deadline_ms is not None and req.deadline_ms > 0


def test_run_recall_step_caps_retrieval_query_at_contract_limit(capture):
    _run({"verb": "stance_react", "user_message": "x", "retrieval_query": "q" * 5000})

    assert len(capture.sent[-1].retrieval_query) == 1000


def test_run_recall_step_explicit_retrieval_query_beats_ctx(capture):
    _run({"verb": "chat_general", "user_message": "x", "retrieval_query": "from ctx"}, retrieval_query="explicit")

    assert capture.sent[-1].retrieval_query == "explicit"


def test_run_recall_step_context_only_mode_needs_no_fragment(capture):
    step, _, _ = _run({"verb": "reverie_narrate", "user_message": None}, recall_cfg={"query_mode": "context_only"})

    assert step.status == "success"
    req = capture.sent[-1]
    assert req.mode == "context_only"
    assert req.fragment == ""
    assert req.retrieval_query is None
    assert req.deadline_ms is not None


def test_run_recall_step_ignores_unknown_mode(capture):
    _run({"verb": "chat_general", "user_message": "x"}, recall_cfg={"query_mode": "bogus"})

    assert capture.sent[-1].mode == "retrieve"


def _phase3_ready_ctx(**extra):
    ctx = {
        "verb": "chat_general",
        "user_message": "what did we decide about the move?",
        "retrieval_query": STANDING_QUESTION,
        "chat_stance_brief": {"task_mode": "instrumental", "interaction_regime": "instrumental"},
        "turn_change_appraisal": {"novelty_score": 0.72, "shift_kind": "TOPIC"},
    }
    ctx.update(extra)
    return ctx


def test_pcr_phase3_reuses_phase01_retrieval_query(capture, monkeypatch):
    monkeypatch.setattr(settings, "chat_pcr_enabled", True)
    monkeypatch.setattr(settings, "chat_pcr_skip_on_low_info", False)
    ctx = _phase3_ready_ctx()

    asyncio.run(run_pcr_phase0_and_1(object(), source=SOURCE, ctx=ctx, correlation_id="c", recall_cfg={}))
    assert ctx[PCR_RETRIEVAL_QUERY_KEY] == STANDING_QUESTION
    # Something rewrites ctx between the phases: phase 3 must not re-derive.
    ctx["retrieval_query"] = "a different, later text"
    _, phase3_step, _ = asyncio.run(
        run_pcr_phase3(object(), source=SOURCE, ctx=ctx, correlation_id="c", recall_cfg={"enabled": True})
    )

    assert phase3_step is not None
    assert len(capture.sent) == 2
    phase01, phase3 = capture.sent
    assert phase01.recall_phase == "continuity"
    assert phase3.recall_phase == "purposeful"
    assert phase3.retrieval_query == phase01.retrieval_query == STANDING_QUESTION


def test_pcr_phase3_reuses_none_when_phase01_had_no_caller_query(capture, monkeypatch):
    monkeypatch.setattr(settings, "chat_pcr_enabled", True)
    monkeypatch.setattr(settings, "chat_pcr_skip_on_low_info", False)
    ctx = _phase3_ready_ctx(retrieval_query=None)

    asyncio.run(run_pcr_phase0_and_1(object(), source=SOURCE, ctx=ctx, correlation_id="c", recall_cfg={}))
    ctx["retrieval_query"] = "appeared later"
    asyncio.run(run_pcr_phase3(object(), source=SOURCE, ctx=ctx, correlation_id="c", recall_cfg={"enabled": True}))

    assert [r.retrieval_query for r in capture.sent] == [None, None]


def test_pcr_phase3_alone_falls_back_to_same_derivation(capture):
    ctx = _phase3_ready_ctx()
    asyncio.run(run_pcr_phase3(object(), source=SOURCE, ctx=ctx, correlation_id="c", recall_cfg={"enabled": True}))

    assert capture.sent[-1].retrieval_query == STANDING_QUESTION


def test_stance_grounding_phase3_query_equals_phase01_query(capture, monkeypatch):
    """stance_react: PCR 0+1 then phase 3 (grounding_capsule) send one search text."""
    monkeypatch.setattr(settings, "orion_unified_grounding_enabled", True)
    monkeypatch.setattr(settings, "chat_pcr_enabled", True)
    monkeypatch.setattr(settings, "chat_pcr_skip_on_low_info", False)
    ctx = _phase3_ready_ctx(verb="stance_react")
    ctx.pop("chat_stance_brief")

    capsule = asyncio.run(
        assemble_stance_grounding(
            object(),
            source=SOURCE,
            ctx=ctx,
            correlation_id="c",
            recall_cfg={"enabled": True},
            stance_step_text='{"task_mode": "instrumental", "interaction_regime": "instrumental"}',
        )
    )

    assert capsule is not None
    assert len(capture.sent) == 2, [r.recall_phase for r in capture.sent]
    assert capture.sent[0].retrieval_query == capture.sent[1].retrieval_query == STANDING_QUESTION
    assert all(r.deadline_ms for r in capture.sent)


def test_reverie_verb_yamls_declare_context_only():
    for verb in ("reverie_narrate", "reverie_expectation_judge"):
        assert build_plan_for_verb(verb).metadata["recall_query_mode_default"] == "context_only"
    assert build_plan_for_verb("chat_general").metadata["recall_query_mode_default"] == ""


@pytest.mark.parametrize("verb", ["reverie_narrate", "reverie_expectation_judge"])
def test_router_sends_context_only_for_reverie_verbs(capture, monkeypatch, verb):
    """The live log showed `verb=reverie_narrate recall_enabled=True recall_cfg={}`: the
    router pre-recall runs for reverie. It must now send mode=context_only, not an
    empty-query search."""
    fake_llm = StepExecutionResult(
        status="success",
        verb_name=verb,
        step_name="llm",
        order=0,
        result={"LLMGatewayService": {"content": "{}"}},
        latency_ms=1,
        node="n",
        logs=[],
        error=None,
    )
    monkeypatch.setattr(router, "call_step_services", AsyncMock(return_value=fake_llm))
    monkeypatch.setattr(router, "prepare_brain_reply_context", AsyncMock(return_value=None))
    ctx = {"user_message": None, "mode": "reverie", "metadata": {"mode": "reverie"}}
    req = PlanExecutionRequest(
        plan=build_plan_for_verb(verb, mode="brain"),
        args=PlanExecutionArgs(request_id="rq", extra={"mode": "brain"}),
        context=ctx,
    )

    asyncio.run(PlanRunner().run_plan(bus=object(), source=SOURCE, req=req, correlation_id="corr-r", ctx=ctx))

    assert capture.sent, "router did not recall for this verb -- the context_only seam is untested"
    assert all(r.mode == "context_only" for r in capture.sent)
    assert all(r.retrieval_query is None for r in capture.sent)


def _run_reverie_plan(monkeypatch, *, extra):
    fake_llm = StepExecutionResult(
        status="success", verb_name="reverie_narrate", step_name="llm", order=0,
        result={"LLMGatewayService": {"content": "{}"}}, latency_ms=1, node="n", logs=[], error=None,
    )
    monkeypatch.setattr(router, "call_step_services", AsyncMock(return_value=fake_llm))
    monkeypatch.setattr(router, "prepare_brain_reply_context", AsyncMock(return_value=None))
    ctx = {"user_message": None, "mode": "reverie"}
    req = PlanExecutionRequest(
        plan=build_plan_for_verb("reverie_narrate", mode="brain"),
        args=PlanExecutionArgs(request_id="rq", extra=extra),
        context=ctx,
    )
    asyncio.run(PlanRunner().run_plan(bus=object(), source=SOURCE, req=req, correlation_id="corr-r", ctx=ctx))


def test_router_caller_query_mode_beats_verb_default(capture, monkeypatch):
    _run_reverie_plan(monkeypatch, extra={"mode": "brain", "recall": {"query_mode": "retrieve"}})

    assert capture.sent and all(r.mode == "retrieve" for r in capture.sent)


def test_orch_routed_recall_directive_keeps_the_verb_context_only_default(capture, monkeypatch):
    """PR #2423 review: cortex-orch always sends recall=RecallDirective().model_dump(),
    whose "mode" is "hybrid". With the verb default stored under recall_cfg["mode"], that
    silently suppressed reverie's context_only. The default now has its own key."""
    from orion.schemas.cortex.contracts import RecallDirective

    directive = RecallDirective().model_dump()
    assert directive["mode"] == "hybrid"
    _run_reverie_plan(monkeypatch, extra={"mode": "brain", "recall": directive})

    assert capture.sent and all(r.mode == "context_only" for r in capture.sent)
