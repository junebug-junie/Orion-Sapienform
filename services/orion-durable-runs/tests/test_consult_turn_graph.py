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

from datetime import datetime, timedelta

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.admitted_graph import AdmissionDeps
from app.harness_smoke_graph import (
    CUT_SHORT_CONTINUE_NOTE, CUT_SHORT_ERROR, build_consult_turn_graph, build_harness_smoke_graph,
    consult_turn_finish_detail,
)
from app.harness_step import harness_step_correlation_id
from orion.schemas.consult_turn_run import (
    CONSULT_ORION_TURN_STEP_KEY, CONSULT_ORION_TURN_TAG, CONSULT_ORION_TURN_WORKFLOW, consult_turn_run_id,
)
from orion.schemas.durable_run import CuriosityTurnResultV1
from tests.test_harness_smoke_graph import REF, World
from tests.test_harness_smoke_graph import CFG as SMOKE_CFG
from tests.test_harness_smoke_graph import initial as smoke_initial

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


CUT_TEXT = "[Cut short - not a finished answer.] I stopped after 83 steps. Found so far: ..."


class CutWorld(World):
    """``("cut", text)`` is a turn the motor stopped mid-work: ok, with text, grounding fcc_timeout."""

    async def turn(self, req):
        if self.replies and isinstance(self.replies[0], tuple) and self.replies[0][0] == "cut":
            self.turns.append(req)
            _, text = self.replies.pop(0)
            return CuriosityTurnResultV1(run_id=req.run_id, correlation_id=req.correlation_id, text=text, ok=True,
                                         debug={"harness_grounding_status": "fcc_timeout", "fcc_elapsed_sec": 540.0})
        return await super().turn(req)


async def _resume_after_backoff(world, saver):
    snap = await _graph(world, saver).aget_state(CFG)
    assert snap.next == ("retry_wait",)
    world.now = datetime.fromisoformat(snap.values["retry_at"]) + timedelta(seconds=1)
    world.grant()
    return await asyncio.wait_for(_graph(world, saver).ainvoke(Command(resume=True), CFG), 2)


def test_a_cut_short_turn_is_retried_in_the_same_session_and_told_to_continue():
    async def scenario():
        world, saver = CutWorld(replies=[("cut", CUT_TEXT), LONG]), InMemorySaver()
        world.grant()
        await asyncio.wait_for(_graph(world, saver).ainvoke(_initial(), CFG), 2)
        snap = await _graph(world, saver).aget_state(CFG)
        assert CUT_SHORT_ERROR in snap.values["last_error"] and "fcc_timeout" in snap.values["last_error"]
        assert CONSULT_ORION_TURN_STEP_KEY not in (snap.values.get("steps") or {}), "cut text is not the result"
        result = await _resume_after_backoff(world, saver)
        assert result["status"] == "completed"
        first, second = world.turns
        assert second.source_tag == first.source_tag and second.attempt == 2
        # The motor records its own stop as a finished session, so the retry is a NEW session: the cut
        # attempt's findings must be in the prompt itself, and the note must not claim otherwise.
        assert second.prompt.startswith(CUT_SHORT_CONTINUE_NOTE) and second.prompt.endswith("Reply to Claude.")
        assert CUT_TEXT in second.prompt and "new session" in CUT_SHORT_CONTINUE_NOTE
        assert "above" not in CUT_SHORT_CONTINUE_NOTE
        assert not first.prompt.startswith(CUT_SHORT_CONTINUE_NOTE)
        step = result["steps"][CONSULT_ORION_TURN_STEP_KEY]
        assert step["text"] == LONG and "cut_short" not in step
    asyncio.run(scenario())


def test_the_last_attempt_keeps_the_honest_cut_short_text():
    async def scenario():
        world, saver = CutWorld(replies=[("cut", CUT_TEXT), ("cut", CUT_TEXT)], max_attempts=2), InMemorySaver()
        world.grant()
        await asyncio.wait_for(_graph(world, saver).ainvoke(_initial(), CFG), 2)
        result = await _resume_after_backoff(world, saver)
        assert result["status"] == "completed", "never an empty failure when the turn found things"
        step = result["steps"][CONSULT_ORION_TURN_STEP_KEY]
        assert step["text"] == CUT_TEXT and step["cut_short"] == "fcc_timeout" and step["attempts"] == 2
        assert len(world.turns) == 2
    asyncio.run(scenario())


def test_a_cut_short_turn_is_kept_when_no_whole_attempt_fits_before_the_deadline():
    async def scenario():
        world, saver = CutWorld(replies=[("cut", CUT_TEXT)]), InMemorySaver()
        world.grant()
        state = _initial()
        state["admission"]["deadline_at"] = (world.now + timedelta(seconds=905)).isoformat()   # < one 900 s attempt after the 10 s backoff
        result = await asyncio.wait_for(_graph(world, saver).ainvoke(state, CFG), 2)
        assert result["status"] == "completed" and result["steps"]["turn"]["cut_short"] == "fcc_timeout"
    asyncio.run(scenario())


def test_harness_echo_smoke_does_not_retry_cut_short_turns():
    async def scenario():
        world, saver = CutWorld(replies=[("cut", CUT_TEXT)]), InMemorySaver()
        world.grant()
        result = await asyncio.wait_for(world.graph(saver).ainvoke(smoke_initial(), SMOKE_CFG), 2)
        assert result["status"] == "completed" and len(world.turns) == 1
    asyncio.run(scenario())


def test_every_cut_attempts_findings_reach_the_next_attempt():
    async def scenario():
        world, saver = CutWorld(replies=[("cut", "found A"), ("cut", "found B"), LONG]), InMemorySaver()
        world.grant()
        await asyncio.wait_for(_graph(world, saver).ainvoke(_initial(), CFG), 2)
        await _resume_after_backoff(world, saver)
        result = await _resume_after_backoff(world, saver)
        assert result["status"] == "completed"
        third = world.turns[2].prompt
        assert third.index("Attempt 1 (cut short) ---\nfound A") < third.index("Attempt 2 (cut short) ---\nfound B")
    asyncio.run(scenario())


def test_carried_findings_are_capped_keeping_the_newest():
    from app.harness_smoke_graph import CUT_SHORT_CARRY_MAX_CHARS, carry_cut_short

    carried = carry_cut_short("x" * CUT_SHORT_CARRY_MAX_CHARS, "newest finding", 2)
    assert len(carried) == CUT_SHORT_CARRY_MAX_CHARS and carried.endswith("newest finding")


def test_the_motor_ends_a_timed_out_turn_as_finished_so_a_retry_is_a_fresh_session():
    """Why the findings ride in the prompt: pins the motor rule the retry design depends on."""
    import inspect

    from orion.harness import fcc_motor

    src = inspect.getsource(fcc_motor._record_session_end)
    assert "STATE_COMPLETED" in src and "timeout, stall" in src
