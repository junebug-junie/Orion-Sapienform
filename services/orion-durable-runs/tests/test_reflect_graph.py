"""self_study.reflect's own durable-run graph: one LLM call, no publish --
finding validation and journal/self_concept_history writes stay in
cortex-exec (they need the full snapshot/concepts for evidence-chain
construction). A transport failure is resumable at llm_call; a non-ok
result, empty text, or bad JSON shape is NOT a crash -- it's a valid
"llm_call_ok=False" completion so cortex-exec's synchronous waiter gets a
real answer instead of hanging until its own timeout.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

pytest.importorskip("langgraph")

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from app.reflect_graph import Deps, ReflectLlmCallFailed, build_reflect_graph, finish_detail, parse_reflect_findings  # noqa: E402


def _cfg(run_id: str) -> dict:
    return {"configurable": {"thread_id": run_id}}


def _brief() -> dict:
    return {
        "prompt": "reflect",
        "session_id": "self-study-reflect",
        "timeout_sec": 1500.0,
        "line": "reflect",
        "self_study_reflect_input": {"snapshot_id": "self-snapshot-1", "counts_by_kind": {}, "concepts": []},
        "llm_route": "agent",
    }


# --- parse_reflect_findings (pure) ------------------------------------------


def test_parse_reflect_findings_extracts_a_valid_list():
    text = '```json\n{"findings": [{"reflection_kind": "pattern", "title": "t"}]}\n```'
    assert parse_reflect_findings(text) == [{"reflection_kind": "pattern", "title": "t"}]


def test_parse_reflect_findings_strips_think_blocks():
    text = '<think>reasoning here</think>{"findings": []}'
    assert parse_reflect_findings(text) == []


def test_parse_reflect_findings_none_on_empty_text():
    assert parse_reflect_findings("") is None


def test_parse_reflect_findings_none_on_bad_json():
    assert parse_reflect_findings("not json") is None


def test_parse_reflect_findings_none_on_wrong_shape():
    assert parse_reflect_findings('{"not_findings": []}') is None
    assert parse_reflect_findings('{"findings": "not a list"}') is None


def test_parse_reflect_findings_drops_non_dict_items():
    assert parse_reflect_findings('{"findings": [{"a": 1}, "bad", 2]}') == [{"a": 1}]


# --- the graph ---------------------------------------------------------------


def test_a_successful_call_completes_with_findings():
    async def call_reflect_llm(reflect_input, llm_route):
        assert reflect_input == {"snapshot_id": "self-snapshot-1", "counts_by_kind": {}, "concepts": []}
        assert llm_route == "agent"
        return [{"reflection_kind": "pattern", "title": "t", "description": "d", "concept_kinds": ["x"]}]

    graph = build_reflect_graph(Deps(call_reflect_llm=call_reflect_llm), InMemorySaver())
    initial = {"run_id": "reflect-run-1", "correlation_id": "corr-1", "workflow": "self_study.reflect", "brief": _brief(), "attempt": 0}
    final = asyncio.run(graph.ainvoke(initial, _cfg("reflect-run-1")))

    assert final["status"] == "completed"
    assert final["llm_call_ok"] is True
    assert final["findings"] == [{"reflection_kind": "pattern", "title": "t", "description": "d", "concept_kinds": ["x"]}]
    detail = finish_detail(final)
    assert detail["line"] == "reflect" and detail["llm_call_ok"] is True and len(detail["findings"]) == 1


def test_a_none_result_completes_with_llm_call_ok_false_not_a_crash():
    """The LLM call succeeded at the transport level but returned nothing
    usable (non-ok status, empty text, bad JSON) -- a valid completion, not
    a node failure, so cortex-exec's waiter gets a real answer."""
    async def call_reflect_llm(reflect_input, llm_route):
        return None

    graph = build_reflect_graph(Deps(call_reflect_llm=call_reflect_llm), InMemorySaver())
    initial = {"run_id": "reflect-run-2", "correlation_id": "corr-2", "workflow": "self_study.reflect", "brief": _brief(), "attempt": 0}
    final = asyncio.run(graph.ainvoke(initial, _cfg("reflect-run-2")))

    assert final["status"] == "completed"
    assert final["llm_call_ok"] is False
    assert final["findings"] == []
    detail = finish_detail(final)
    assert detail["llm_call_ok"] is False and detail["llm_call_error"] == "llm_call_failed"


def test_a_transport_failure_leaves_the_thread_resumable_at_llm_call():
    calls = []

    async def call_reflect_llm(reflect_input, llm_route):
        calls.append(1)
        raise RuntimeError("rpc down")

    graph = build_reflect_graph(Deps(call_reflect_llm=call_reflect_llm), InMemorySaver())
    initial = {"run_id": "reflect-run-3", "correlation_id": "corr-3", "workflow": "self_study.reflect", "brief": _brief(), "attempt": 0}
    config = _cfg("reflect-run-3")
    # DurableRunner._drive normally catches this (see runner.py) and leaves
    # the thread resumable at its next node; driving the compiled graph
    # directly here (no runner wrapping it) means the exception surfaces
    # from ainvoke itself -- the checkpoint underneath is still intact.
    with pytest.raises(ReflectLlmCallFailed):
        asyncio.run(graph.ainvoke(initial, config))
    snap = asyncio.run(graph.aget_state(config))
    assert snap.next == ("llm_call",)
    assert len(calls) == 1
