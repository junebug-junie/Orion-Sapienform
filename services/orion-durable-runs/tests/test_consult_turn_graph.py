"""consult.orion_turn: one of Orion's consult turns as an admitted durable run (the single-step harness
graph shared with harness.echo_smoke). What must hold: the turn runs under the hold, in Orion's consult
session, with lookup NOT required (Orion may just reply to Claude), recall aimed at the question, a
per-turn source tag, and the full text checkpointed; a failed turn is retried by the shared policy."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("langgraph")

from langgraph.checkpoint.memory import InMemorySaver

from app.admitted_graph import AdmissionDeps
from app.harness_smoke_graph import build_consult_turn_graph, consult_turn_finish_detail
from app.harness_step import harness_step_correlation_id
from orion.schemas.consult_turn_run import (
    CONSULT_ORION_TURN_STEP_KEY, CONSULT_ORION_TURN_TAG, CONSULT_ORION_TURN_WORKFLOW, consult_turn_run_id,
)
from tests.test_harness_smoke_graph import REF, World

RUN_ID = consult_turn_run_id("9f138a849520", 3)
CFG = {"configurable": {"thread_id": RUN_ID}}
LONG = "I agree with Claude's first point. " * 200   # longer than the 2000-char state-row cap


def _graph(world, saver):
    return build_consult_turn_graph(
        world.turn,
        AdmissionDeps(world.register, world.lease, world.execute, world.release, world.event,
                      now=lambda: world.now, max_attempts=world.max_attempts, retry_base_seconds=10.0),
        saver)


def _initial():
    return {"run_id": RUN_ID, "correlation_id": "trace-consult", "workflow": CONSULT_ORION_TURN_WORKFLOW,
            "attempt": 0, "admission": {"resource": "llm.route.agent"},
            "brief": {"consult": "consult.orion_turn", "consult_id": "9f138a849520", "turn_no": 3,
                      "prompt": "Reply to Claude.", "session_id": "consult-9f138a849520",
                      "retrieval_query": "outward learning", "timeout_sec": 900}}


def test_run_id_is_one_per_turn():
    assert RUN_ID == "consult-9f138a849520-t3"


def test_consult_turn_runs_writing_only_in_its_session_and_keeps_the_full_text():
    async def scenario():
        world, saver = World(replies=[LONG]), InMemorySaver()
        world.grant()
        result = await asyncio.wait_for(_graph(world, saver).ainvoke(_initial(), CFG), 2)
        assert result["status"] == "completed"
        [req] = world.turns
        assert req.require_lookup is False and req.session_id == "consult-9f138a849520"
        assert req.retrieval_query == "outward learning" and req.source_tag == f"{CONSULT_ORION_TURN_TAG}:t3"
        assert req.gpu_lease.model_dump() == REF and req.timeout_sec == pytest.approx(870.0)
        assert req.model_dump(mode="json")["require_lookup"] is False   # on the wire, not omitted
        assert result["steps"][CONSULT_ORION_TURN_STEP_KEY]["text"] == LONG   # checkpoint keeps it all
        detail = consult_turn_finish_detail(result)
        assert detail["line"] == "consult_orion_turn" and detail["steps"]["turn"]["text_truncated"] is True
    asyncio.run(scenario())


def test_a_failed_consult_turn_is_retried_not_dropped():
    async def scenario():
        world, saver = World(replies=[("fail", "timeout"), LONG]), InMemorySaver()
        world.grant()
        await asyncio.wait_for(_graph(world, saver).ainvoke(_initial(), CFG), 2)
        snap = await _graph(world, saver).aget_state(CFG)
        assert snap.next == ("retry_wait",), "a timeout is an attempt, retried after backoff"
    asyncio.run(scenario())


def test_admission_runtime_registers_the_consult_turn():
    from app.admission_runtime import HARNESS_STEP_WORKFLOWS, WORK_NODES, AdmissionRuntime

    assert WORK_NODES[CONSULT_ORION_TURN_WORKFLOW] == {"harness_step"}
    assert HARNESS_STEP_WORKFLOWS[CONSULT_ORION_TURN_WORKFLOW] == CONSULT_ORION_TURN_STEP_KEY
    runtime = object.__new__(AdmissionRuntime)
    runtime.runner = SimpleNamespace(_publish=AsyncMock(return_value=True), _corr_for_admission=lambda v: v)
    detail = runtime._finish_detail_for(CONSULT_ORION_TURN_WORKFLOW, {
        "steps": {CONSULT_ORION_TURN_STEP_KEY: {"text": "ok", "turn_correlation_id": "c-1"}}})
    assert detail["line"] == "consult_orion_turn" and detail["turn_correlation_id"] == "c-1"
    state = {**_initial(), "lease": dict(REF), "hold_seq": 1}
    assert harness_step_correlation_id(state, CONSULT_ORION_TURN_STEP_KEY)
