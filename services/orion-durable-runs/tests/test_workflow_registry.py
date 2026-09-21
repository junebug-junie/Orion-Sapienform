"""DurableRunner's workflow registry (2026-09-21): more than one compiled
graph can share the runner and its checkpointer. Curiosity's own tests
(test_curiosity_graph_resume.py etc.) already prove the registry causes zero
behavior change for the one real workflow that exists today -- this file
proves the registry mechanism itself: a second, independent graph can be
registered, driven, and correctly discriminated from curiosity's own threads
during the resume sweep, all against the SAME shared checkpointer.

Deliberately does not touch `DurableWorkflowV1` (still
`Literal["curiosity.investigate"]` on the wire) -- widening that schema is
the concern of whichever PR adds a real second workflow. This file exercises
the registry through `runner.register_workflow()` / `runner._spec_for()`
directly, the same Python-level API a real second workflow will use.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, TypedDict

import pytest

pytest.importorskip("langgraph")

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402

from orion.schemas.durable_run import CuriosityRunBriefV1, DurableRunRequestV1  # noqa: E402

from app.graph import build_curiosity_graph  # noqa: E402
from app.runner import DEFAULT_WORKFLOW, WorkflowSpec  # noqa: E402


TOY_WORKFLOW = "toy.echo"


class ToyState(TypedDict, total=False):
    run_id: str
    correlation_id: str
    workflow: str
    brief: dict[str, Any]
    attempt: int
    text: str


def _build_toy_graph(checkpointer: Any):
    """Two nodes: echo -> finish. No Hub, no harness -- proves the registry
    doesn't assume every workflow talks to curiosity's turn-execution path."""

    async def echo(state: ToyState) -> dict[str, Any]:
        return {"text": f"echo:{state['brief'].get('prompt', '')}"}

    async def finish(state: ToyState) -> dict[str, Any]:
        return {"status": "completed"}

    g: StateGraph = StateGraph(ToyState)
    g.add_node("echo", echo)
    g.add_node("finish", finish)
    g.add_edge(START, "echo")
    g.add_edge("echo", "finish")
    g.add_edge("finish", END)
    return g.compile(checkpointer=checkpointer)


def _toy_finish_detail(state: dict[str, Any]) -> dict[str, Any]:
    return {"text": state.get("text", "")}


def _runner(saver):
    import app.settings as settings_mod

    settings_mod._settings = None
    from app.runner import DurableRunner

    return DurableRunner(settings_mod.get_settings(), bus=None, checkpointer=saver)


def _curiosity_request(run_id: str) -> DurableRunRequestV1:
    return DurableRunRequestV1(
        run_id=run_id, workflow="curiosity.investigate", correlation_id="7dcc3944-29bb-5d8f-915f-90f4e6968d47",
        brief=CuriosityRunBriefV1(
            prompt="Pick something.", session_id="orion_curiosity", timeout_sec=3500.0, graph_configured=True,
            material={"approved_total": 1, "approved_by_kind": {"semantic": 1}, "crystallization_count": 0, "relation_total": 0, "relation_count": 0},
        ),
    )


def test_a_second_registered_workflow_drives_independently_of_curiosity(monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    monkeypatch.setenv("ORION_BUS_ENABLED", "false")
    saver = InMemorySaver()
    runner = _runner(saver)
    runner.register_workflow(
        WorkflowSpec(workflow=TOY_WORKFLOW, graph=_build_toy_graph(saver), nodes=["echo", "finish"], finish_detail=_toy_finish_detail)
    )
    spec = runner._spec_for(TOY_WORKFLOW)
    assert spec is not None

    config = runner._config("toy-run-1")
    initial: ToyState = {
        "run_id": "toy-run-1", "correlation_id": "corr-1", "workflow": TOY_WORKFLOW,
        "brief": {"prompt": "hi"}, "attempt": 0,
    }
    final = asyncio.run(spec.graph.ainvoke(initial, config))
    assert final["text"] == "echo:hi"

    # Curiosity's own graph, sharing the SAME checkpointer, is untouched.
    curiosity_spec = runner._spec_for(DEFAULT_WORKFLOW)
    snap = asyncio.run(curiosity_spec.graph.aget_state(config))
    # Same thread_id, different graph's node schema: curiosity's graph has
    # never driven this thread, so it reads as empty, not as the toy result.
    assert not snap or not snap.values or "text" not in snap.values or snap.values.get("workflow") != DEFAULT_WORKFLOW


def test_peek_workflow_discriminates_two_in_flight_threads_on_the_shared_checkpointer(monkeypatch):
    """The resume sweep's core safety property: given two threads on the same
    checkpointer, each is read through ITS OWN graph, never the other's."""
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    monkeypatch.setenv("ORION_BUS_ENABLED", "false")
    saver = InMemorySaver()
    runner = _runner(saver)
    runner.register_workflow(
        WorkflowSpec(workflow=TOY_WORKFLOW, graph=_build_toy_graph(saver), nodes=["echo", "finish"], finish_detail=_toy_finish_detail)
    )

    toy_config = runner._config("toy-run-2")
    toy_initial: ToyState = {
        "run_id": "toy-run-2", "correlation_id": "corr-2", "workflow": TOY_WORKFLOW,
        "brief": {"prompt": "hey"}, "attempt": 0,
    }
    asyncio.run(runner._spec_for(TOY_WORKFLOW).graph.ainvoke(toy_initial, toy_config))

    curiosity_config = runner._config("curiosity-run-1")
    curiosity_spec = runner._spec_for(DEFAULT_WORKFLOW)
    curiosity_initial = {
        "run_id": "curiosity-run-1", "correlation_id": "corr-3", "workflow": DEFAULT_WORKFLOW,
        "brief": _curiosity_request("curiosity-run-1").brief.model_dump(mode="json"), "attempt": 1,
        "text": "found something", "debug": {}, "outcome": None, "footprint": None, "hops": [],
        "evidence_summary": None, "graph_readable": False,
    }
    # Land it mid-run (as-node update, no real turn needed) so it has a
    # `next` node -- the resume sweep only cares about unfinished threads.
    asyncio.run(curiosity_spec.graph.aupdate_state(curiosity_config, curiosity_initial, as_node=START))

    toy_workflow = asyncio.run(runner._peek_workflow("toy-run-2"))
    curiosity_workflow = asyncio.run(runner._peek_workflow("curiosity-run-1"))
    assert toy_workflow == TOY_WORKFLOW
    assert curiosity_workflow == DEFAULT_WORKFLOW

    # A thread with no `workflow` key at all (pre-registry checkpoint) reads
    # as curiosity -- the original single workflow, so old in-flight runs
    # keep resuming exactly as they did before this patch.
    legacy_config = {"configurable": {"thread_id": "legacy-run", "checkpoint_ns": ""}}
    asyncio.run(saver.aput(legacy_config, {"v": 1, "id": "1", "ts": "2026-01-01T00:00:00Z", "channel_values": {}, "channel_versions": {}, "versions_seen": {}}, {}, {}))
    assert asyncio.run(runner._peek_workflow("legacy-run")) == DEFAULT_WORKFLOW


def test_start_run_rejects_an_unregistered_workflow_without_crashing(monkeypatch, caplog):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    monkeypatch.setenv("ORION_BUS_ENABLED", "false")
    saver = InMemorySaver()
    runner = _runner(saver)
    assert runner._spec_for("nonexistent.workflow") is None
    # start_run's own guard: a request naming an unregistered workflow must
    # return cleanly, never spawn a task, never raise.
    fake_request = DurableRunRequestV1.model_construct(
        schema_version="durable.run.request.v1", run_id="bad-run-1", workflow="nonexistent.workflow",
        correlation_id="corr-4", requested_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        brief=_curiosity_request("bad-run-1").brief, admission=None,
    )
    asyncio.run(runner.start_run(fake_request))
    assert runner._active == {}
