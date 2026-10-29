"""harness.echo_smoke: the durable-run -> harness step seam (app/harness_step.py) under real
LangGraph checkpoints, a fake pool and a fake Hub turn.

What must hold: the step runs only under a granted hold, its final text and turn correlation id are
checkpointed as the step's result, a refusal / empty reply / transport failure is a retried step
failure (never an empty success), and a checkpointed step is never re-run.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("langgraph")

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parents[1])]

from app.admitted_graph import GRANTED, WAITING, AdmissionDeps
from app.harness_smoke_graph import build_harness_smoke_graph, finish_detail
from app.harness_step import HarnessStepFailed, harness_step_correlation_id, run_harness_step, step_detail
from orion.schemas.durable_run import CuriosityTurnRequestV1, CuriosityTurnResultV1
from orion.schemas.harness_step_run import (
    HARNESS_ECHO_SMOKE_SESSION_ID, HARNESS_ECHO_SMOKE_STEP_KEY, HARNESS_ECHO_SMOKE_TAG, HARNESS_ECHO_SMOKE_WORKFLOW,
)

RUN_ID = "harness-echo-smoke-test-001"
REF = {"lease_id": "hold-hs", "generation": 1, "role": "agent", "holder": f"durable-runs:{RUN_ID}"}
CFG = {"configurable": {"thread_id": RUN_ID}}
REPLY = "harness echo smoke ok 2026-10-29T00:00:00Z"


class World:
    """One hold: ``queued`` until ``grant()``. ``replies`` is consumed one per turn: a str is the
    turn's text, an Exception is raised from the turn RPC, ``("fail", error)`` is a refusal."""

    def __init__(self, replies=None, max_attempts=3):
        self.now = datetime(2026, 10, 29, tzinfo=timezone.utc)
        self.granted = False
        self.replies = list(replies or [REPLY])
        self.turns: list[CuriosityTurnRequestV1] = []
        self.executes: list[dict] = []
        self.releases: list[str] = []
        self.max_attempts = max_attempts

    def grant(self):
        self.granted = True

    async def turn(self, req: CuriosityTurnRequestV1) -> CuriosityTurnResultV1:
        self.turns.append(req)
        reply = self.replies.pop(0) if self.replies else REPLY
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, tuple):
            return CuriosityTurnResultV1(run_id=req.run_id, correlation_id=req.correlation_id,
                                         ok=False, error=reply[1])
        return CuriosityTurnResultV1(run_id=req.run_id, correlation_id=req.correlation_id, text=reply,
                                     ok=bool(reply), debug={"harness_step_count": 4, "elapsed_sec": 12.5,
                                                            "prompt": "never kept"})

    async def register(self, state):
        return {"status": "waiting_resource", "hold": {"request_id": f"{RUN_ID}:1", "lease_id": "hold-hs"},
                "hold_seq": 1}

    async def lease(self, state):
        if self.granted:
            return GRANTED, {"status": "admitted", "lease": dict(REF), "hold": state.get("hold")}
        return WAITING, {"status": "waiting_resource", "lease": None}

    async def execute(self, state, node):
        self.executes.append({"step_correlation_id": state.get("step_correlation_id")})
        if not self.granted or state.get("lease") != REF:
            raise RuntimeError("gpu_hold_lost")
        return await node(state)

    async def release(self, state, reason, keep_requeued=False):
        if not state.get("hold") and not state.get("lease"):
            return {"lease": None, "hold": None}
        self.releases.append(reason)
        self.granted = False
        return {"lease": None, "hold": None}

    async def event(self, *args):
        pass

    def graph(self, saver):
        return build_harness_smoke_graph(
            self.turn,
            AdmissionDeps(self.register, self.lease, self.execute, self.release, self.event,
                          now=lambda: self.now, max_attempts=self.max_attempts, retry_base_seconds=10.0),
            saver)


def initial(**extra):
    return {"run_id": RUN_ID, "correlation_id": "trace-hs", "workflow": HARNESS_ECHO_SMOKE_WORKFLOW,
            "attempt": 0, "admission": {"resource": "llm.route.agent"},
            "brief": {"smoke": "harness.echo_smoke", "prompt": "Run date -u, reply one line.", "timeout_sec": 300},
            **extra}


async def _resume_after_backoff(world, saver):
    snap = await world.graph(saver).aget_state(CFG)
    assert snap.next == ("retry_wait",)
    world.now = datetime.fromisoformat(snap.values["retry_at"]) + timedelta(seconds=1)
    world.grant()
    return await world.graph(saver).ainvoke(Command(resume=True), CFG)


def test_waits_for_the_hold_then_checkpoints_the_turn_text_and_its_correlation_id():
    async def scenario():
        world, saver = World(), InMemorySaver()
        await asyncio.wait_for(world.graph(saver).ainvoke(initial(), CFG), 2)
        snap = await world.graph(saver).aget_state(CFG)
        assert snap.next == ("resource_wait",) and world.turns == []   # waiting is never a turn
        world.grant()
        result = await asyncio.wait_for(world.graph(saver).ainvoke(Command(resume={}), CFG), 2)
        assert result["status"] == "completed" and world.releases == ["completed"]
        [req] = world.turns
        # The existing Hub turn contract, under the hold, in the smoke's own clean session.
        assert req.gpu_lease is not None and req.gpu_lease.model_dump() == REF
        assert req.session_id == HARNESS_ECHO_SMOKE_SESSION_ID and req.source_tag == HARNESS_ECHO_SMOKE_TAG
        assert req.prompt == "Run date -u, reply one line." and req.attempt == 1
        # Hub's limit leaves the reply grace inside the node timer.
        assert req.timeout_sec == pytest.approx(270.0)
        record = result["steps"][HARNESS_ECHO_SMOKE_STEP_KEY]
        assert record["text"] == REPLY and record["final_len"] == len(REPLY) and record["attempts"] == 1
        assert record["turn_correlation_id"] == req.correlation_id == world.executes[0]["step_correlation_id"]
        assert record["debug"] == {"harness_step_count": 4, "elapsed_sec": 12.5}   # bounded, no prompt
        assert record["harness_elapsed_sec"] >= 0 and record["harness_started_at"]
        detail = finish_detail(result)
        assert detail["turn_correlation_id"] == req.correlation_id
        assert detail["steps"][HARNESS_ECHO_SMOKE_STEP_KEY]["text"] == REPLY
    asyncio.run(scenario())


@pytest.mark.parametrize("first", ["", "   ", ("fail", "timeout"), ("fail", "no_lookup"), ConnectionError("rpc")])
def test_an_empty_refused_or_lost_turn_is_a_retried_failure_then_the_retry_lands(first):
    async def scenario():
        world, saver = World(replies=[first, REPLY]), InMemorySaver()
        world.grant()
        result = await asyncio.wait_for(world.graph(saver).ainvoke(initial(), CFG), 2)
        # Not an empty success: retrying, nothing checkpointed as the step's result.
        assert result["status"] == "retrying" and result["attempt"] == 1
        assert not (result.get("steps") or {})
        assert world.releases == ["attempt_failed"]
        assert result["harness_turn_meta"]["turn_correlation_id"] == world.turns[0].correlation_id
        done = await asyncio.wait_for(_resume_after_backoff(world, saver), 2)
        assert done["status"] == "completed"
        record = done["steps"][HARNESS_ECHO_SMOKE_STEP_KEY]
        assert record["text"] == REPLY and record["attempts"] == 2
        # The retry is a fresh turn identity (Hub's result cache cannot hand it the failed one).
        assert world.turns[0].correlation_id != world.turns[1].correlation_id
        assert record["turn_correlation_id"] == world.turns[1].correlation_id
        assert world.turns[1].attempt == 2
    asyncio.run(scenario())


def test_past_the_retry_budget_the_run_fails_with_the_turns_error_never_completes():
    async def scenario():
        world, saver = World(replies=[("fail", "timeout")] * 5, max_attempts=2), InMemorySaver()
        world.grant()
        first = await asyncio.wait_for(world.graph(saver).ainvoke(initial(), CFG), 2)
        assert first["status"] == "retrying"
        done = await asyncio.wait_for(_resume_after_backoff(world, saver), 2)
        assert done["status"] == "failed" and len(world.turns) == 2
        assert "harness_step_failed:echo:timeout" in done["last_error"]
        assert not (done.get("steps") or {})
        # The failed run still names the turn that failed (AdmissionRuntime._terminal_detail_for).
        assert done["harness_turn_meta"]["turn_correlation_id"] == world.turns[-1].correlation_id
    asyncio.run(scenario())


def test_a_checkpointed_step_is_never_run_again():
    async def scenario():
        world, saver = World(), InMemorySaver()
        world.grant()
        done_step = {"step_key": HARNESS_ECHO_SMOKE_STEP_KEY, "text": REPLY, "turn_correlation_id": "c-1"}
        result = await asyncio.wait_for(
            world.graph(saver).ainvoke(initial(steps={HARNESS_ECHO_SMOKE_STEP_KEY: done_step}), CFG), 2)
        assert result["status"] == "completed" and world.turns == []
        assert finish_detail(result)["turn_correlation_id"] == "c-1"
    asyncio.run(scenario())


def test_a_reply_for_another_turn_is_a_step_failure():
    async def scenario():
        async def wrong(req):
            return CuriosityTurnResultV1(run_id=req.run_id, correlation_id="someone-else", text=REPLY)

        state = {**initial(), "lease": dict(REF)}
        with pytest.raises(HarnessStepFailed, match="identity_mismatch"):
            await run_harness_step(wrong, state, step_key="echo", prompt="p", correlation_id="c-1",
                                   timeout_sec=10, source_tag="t", session_id="s")
    asyncio.run(scenario())


def test_step_correlation_id_changes_with_every_replay_counter_and_step():
    base = {**initial(), "hold_seq": 1}
    first = harness_step_correlation_id(base, "echo")
    assert first == harness_step_correlation_id(dict(base), "echo")   # derivable from the checkpoint
    for change in ({"attempt": 1}, {"turn_fence": 1}, {"hold_seq": 2}, {"hold_takebacks": 1}):
        assert harness_step_correlation_id({**base, **change}, "echo") != first
    assert harness_step_correlation_id(base, "hop:what_it_was") != first


def test_step_detail_caps_the_text_for_the_state_row():
    record = {"step_key": "echo", "text": "x" * 5000, "turn_correlation_id": "c", "debug": {"a": 1}}
    detail = step_detail(record)
    assert len(detail["text"]) == 2000 and detail["text_truncated"] is True and "debug" not in detail


def test_admission_runtime_wires_the_smoke_and_cancels_its_pending_step_turn():
    from app.admission_runtime import WORK_NODES, AdmissionRuntime

    assert WORK_NODES[HARNESS_ECHO_SMOKE_WORKFLOW] == {"harness_step"}
    runtime = object.__new__(AdmissionRuntime)
    runtime.runner = SimpleNamespace(_publish=AsyncMock(return_value=True), _corr_for_admission=lambda v: v)
    state = {**initial(), "lease": dict(REF), "hold_seq": 1}
    asyncio.run(runtime._cancel_harness(state, "worker_recovery"))
    cancelled = {call.args[2].correlation_id for call in runtime.runner._publish.await_args_list}
    assert harness_step_correlation_id(state, HARNESS_ECHO_SMOKE_STEP_KEY) in cancelled
    detail = runtime._finish_detail_for(HARNESS_ECHO_SMOKE_WORKFLOW, {
        "steps": {HARNESS_ECHO_SMOKE_STEP_KEY: {"text": REPLY, "turn_correlation_id": "c-9"}}})
    assert detail["line"] == "harness_smoke" and detail["turn_correlation_id"] == "c-9"


def _smoke_script():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "harness_echo_smoke.py"
    spec = importlib.util.spec_from_file_location("harness_echo_smoke_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_trigger_script_builds_a_valid_admitted_smoke_request():
    from orion.schemas.durable_run import DurableRunRequestV1
    from orion.schemas.harness_step_run import HarnessEchoSmokeBriefV1

    script = _smoke_script()
    request = DurableRunRequestV1.model_validate(script.build_request())
    assert request.workflow == HARNESS_ECHO_SMOKE_WORKFLOW and isinstance(request.brief, HarnessEchoSmokeBriefV1)
    assert request.admission is not None and request.admission.resource == "llm.route.agent"
    assert request.admission.deadline_at is not None and script.MARKER in request.brief.prompt


def test_the_smoke_request_contract_refuses_mismatches():
    from pydantic import ValidationError

    from orion.schemas.durable_run import CuriosityRunBriefV1, DurableRunRequestV1

    good = _smoke_script().build_request()
    with pytest.raises(ValidationError, match="durable resource admission"):
        DurableRunRequestV1.model_validate({**good, "admission": None})
    with pytest.raises(ValidationError):   # a curiosity brief cannot ride the smoke workflow
        DurableRunRequestV1.model_validate({**good, "brief": {"prompt": "p", "session_id": "s", "timeout_sec": 5}})
    with pytest.raises(ValidationError):   # nor a smoke brief another workflow
        DurableRunRequestV1.model_validate({**good, "workflow": "journal.compose"})
    # A curiosity brief is never mistaken for the smoke's (the smoke's literal is required).
    curiosity = DurableRunRequestV1.model_validate({"run_id": "curio-001", "workflow": "curiosity.investigate",
                                                    "correlation_id": "c", "brief": {"prompt": "p", "session_id": "s",
                                                                                     "timeout_sec": 5}})
    assert isinstance(curiosity.brief, CuriosityRunBriefV1)


def test_the_trigger_script_verdict_never_passes_an_empty_or_failed_run():
    script = _smoke_script()
    ok = {"status": "completed", "harness_steps": {"echo": {"text": "Harness echo smoke ok 2026-10-29T01:00:00Z"}}}
    assert script.verdict(ok)[0] is True
    assert script.verdict({"status": "failed", "error": "harness_step_failed:echo:timeout"})[0] is False
    assert script.verdict({"status": "completed", "harness_steps": {"echo": {"text": ""}}})[0] is False
    assert script.verdict({"status": "completed", "harness_steps": {"echo": {"text": "hello"}}})[0] is False
