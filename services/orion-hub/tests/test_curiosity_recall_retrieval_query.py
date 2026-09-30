"""Self-initiated curiosity turns recall about their standing question, durably.

Recall retrieval design phase 3 (docs/superpowers/specs/2026-09-29-recall-retrieval-
query-architecture-design.md), Juniper's answer: self-inquiry and curiosity turns recall
about their standing question. It rides the durable run brief -> CuriosityTurnRequestV1 ->
``_generate`` -> ``execute_unified_turn(retrieval_query=...)``, so it survives a Hub restart
that empties the in-memory ``_mind_appraisal_by_run_id`` dict.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

# sys.path is arranged by tests/conftest.py (Hub root first). Do not insert paths here.
from orion.curiosity.self_inquiry import SELF_INQUIRY_TAG
from orion.schemas.durable_run import CuriosityTurnRequestV1, DurableRunRequestV1

from test_curiosity_investigation import _CortexBus, _FakeBus, _loop
from test_curiosity_self_inquiry import _DefinitionReader, _GrantConn, _self_loop
from scripts.curiosity_investigation import (
    INVESTIGATION_TAG,
    CuriosityInvestigation,
    _standing_question_from_view,
)

RUN = "abcdef123456"
QUESTION = "What in me keeps going when nobody is talking to me?"


def _durable_brief(bus) -> DurableRunRequestV1:
    assert len(bus.rpc_calls) == 1
    _channel, envelope, _reply = bus.rpc_calls[0]
    return DurableRunRequestV1.model_validate(envelope.payload["context"]["metadata"]["durable_run"])


def test_self_inquiry_durable_brief_carries_the_picked_question() -> None:
    bus = _CortexBus()
    loop = _self_loop(bus, reader=_DefinitionReader(), conn=_GrantConn(), kickoff_via_cortex=True)

    assert asyncio.run(loop.tick_self_inquiry()) == "dispatched"

    request = _durable_brief(bus)
    assert request.brief.line == "self_inquiry"
    assert request.brief.retrieval_query == loop._last_self_question.text
    assert request.brief.retrieval_query
    # The prompt is the long instruction text; the search text is not it.
    assert request.brief.retrieval_query != request.brief.prompt
    assert len(request.brief.retrieval_query) < len(request.brief.prompt)


def test_self_inquiry_in_process_turn_carries_the_picked_question() -> None:
    bus = _FakeBus()
    loop = _self_loop(bus, reader=_DefinitionReader(), conn=_GrantConn(), kickoff_via_cortex=False)

    assert asyncio.run(loop.tick_self_inquiry()) is None

    assert loop.seen_generate_kwargs.get("retrieval_query") == loop._last_self_question.text


def test_standing_question_is_the_continuation_note_only_when_continuing() -> None:
    cont = SimpleNamespace(continue_line=True, continue_note="  why does substrate.route have no edges?  ")
    assert _standing_question_from_view(SimpleNamespace(continuation=cont)) == (
        "why does substrate.route have no edges?"
    )
    stopped = SimpleNamespace(continue_line=False, continue_note="old note")
    assert _standing_question_from_view(SimpleNamespace(continuation=stopped)) is None
    assert _standing_question_from_view(SimpleNamespace(continuation=None)) is None


def test_investigation_kickoff_without_a_continuation_sends_no_brief_query() -> None:
    bus = _CortexBus()
    loop = _loop(bus, kickoff_via_cortex=True)

    assert asyncio.run(loop.tick()) == "dispatched"

    # Fresh kickoff: Orion has not chosen a question yet. None on the brief; the turn
    # then falls back to the Mind appraisal (tested below).
    assert _durable_brief(bus).brief.retrieval_query is None


def test_run_brief_caps_the_question() -> None:
    loop = _loop(_FakeBus())
    brief = loop._run_brief(prompt="p", material=None, retrieval_query="q" * 3000)
    assert len(brief.retrieval_query) == 1000


def _real_generate_turn(loop: CuriosityInvestigation, request: CuriosityTurnRequestV1) -> dict:
    """Serve one runner turn request through the REAL _generate, with the unified turn stubbed."""
    import orion.hub.turn_orchestrator as orchestrator

    captured: dict = {}
    original = orchestrator.execute_unified_turn

    async def _stub(**kwargs):
        captured.update(kwargs)
        return [{"type": "final", "llm_response": "found it", "harness_step_count": 9}]

    orchestrator.execute_unified_turn = _stub
    try:
        loop._generate = CuriosityInvestigation._generate.__get__(loop)
        asyncio.run(loop._turn_result_for(request, hold_lock=False))
    finally:
        orchestrator.execute_unified_turn = original
    return captured


def test_retrieval_query_survives_a_hub_restart() -> None:
    """A fresh Hub process (empty appraisal dict) serving the runner's turn request for a
    run dispatched before the restart still recalls about the standing question."""
    restarted = _loop(_CortexBus(), kickoff_via_cortex=True)
    assert restarted._mind_appraisal_by_run_id == {}

    captured = _real_generate_turn(
        restarted,
        CuriosityTurnRequestV1(
            run_id=RUN,
            correlation_id="corr-restart",
            prompt="A long self-inquiry instruction prompt. " * 200,
            timeout_sec=60.0,
            source_tag=SELF_INQUIRY_TAG,
            retrieval_query=QUESTION,
        ),
    )

    assert captured["retrieval_query"] == QUESTION
    assert captured["mind_appraisal_text"] is None  # the dict really was empty


def test_turn_without_a_standing_question_sends_none_not_the_boilerplate_appraisal() -> None:
    """PR #2423 review: a fresh kickoff's appraisal is build_investigation_subject
    boilerplate. Sending it would read as retrieval_query_source='caller' in recall
    telemetry while carrying no question. No standing question -> None."""
    from orion.curiosity.investigation_subject import build_investigation_subject

    loop = _loop(_CortexBus(), kickoff_via_cortex=True)
    loop._mind_appraisal_by_run_id[RUN] = build_investigation_subject(claim=None, continue_note=None)
    captured = _real_generate_turn(
        loop,
        CuriosityTurnRequestV1(
            run_id=RUN, correlation_id="corr-a", prompt="p", timeout_sec=60.0, source_tag=INVESTIGATION_TAG
        ),
    )
    assert captured["retrieval_query"] is None
    # The appraisal still reaches Mind/stance as before -- only recall's query changed.
    assert captured["mind_appraisal_text"] == loop._mind_appraisal_by_run_id[RUN]


def test_fresh_investigation_kickoff_sends_no_caller_query_end_to_end() -> None:
    """Kickoff with no continuation note, in-process: the turn gets None."""
    bus = _FakeBus()
    loop = _loop(bus, kickoff_via_cortex=False)
    assert asyncio.run(loop.tick()) is None
    assert "retrieval_query" not in loop.seen_generate_kwargs


def test_brief_query_wins_over_the_appraisal() -> None:
    loop = _loop(_CortexBus(), kickoff_via_cortex=True)
    loop._mind_appraisal_by_run_id[RUN] = "appraisal text"
    captured = _real_generate_turn(
        loop,
        CuriosityTurnRequestV1(
            run_id=RUN, correlation_id="corr-c", prompt="p", timeout_sec=60.0,
            source_tag=SELF_INQUIRY_TAG, retrieval_query=QUESTION,
        ),
    )
    assert captured["retrieval_query"] == QUESTION
    assert captured["mind_appraisal_text"] == "appraisal text"
