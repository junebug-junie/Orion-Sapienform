"""pr.field_note (app/pr_field_note_graph.py) under real LangGraph checkpoints, a fake pool, a fake Hub
turn and a real ledger file.

What must hold: a PR that already has a record never takes a hold or a turn; the turn runs under the
hold on the harness step seam with a lean, fixed prompt; the run completes ONLY when the reply carries
this PR's save receipt AND the ledger holds that id as a note (verify_receipt); anything else is a
retried step failure and, past the budget, a failed run -- never an empty or unproven success.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip("langgraph")

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.admitted_graph import GRANTED, WAITING, AdmissionDeps
from app.harness_step import HarnessStepFailed
from app.pr_field_note_graph import TOOL_CALLS_MAX, build_prompt, build_pr_field_note_graph, finish_detail, verify_receipt
from app.pr_field_note_ledger import FieldNoteLedger
from orion.schemas.durable_run import CuriosityTurnRequestV1, CuriosityTurnResultV1
from orion.schemas.pr_field_note_run import (
    PR_FIELD_NOTE_SESSION_ID, PR_FIELD_NOTE_STEP_KEY, PR_FIELD_NOTE_TAG, PR_FIELD_NOTE_WORKFLOW, PrFieldNoteBriefV1,
)

PR = 3105
RUN_ID = f"pr-field-note-{PR}"
REF = {"lease_id": "hold-fn", "generation": 1, "role": "agent", "holder": f"durable-runs:{RUN_ID}"}
CFG = {"configurable": {"thread_id": RUN_ID}}
NOTE_ID = f"fn:{PR}:0123456789"


def note_line(pr, note_id, *, note="[pr_text] A real note about the change in my body here."):
    """A ledger line in the PrFieldNoteV1 shape (written as JSON: tests outside the atlas and dig do
    not import the ledger module: the repo-wide partition test under tests/)."""
    return json.dumps({"schema_version": "pr_field_note.v1", "note_id": note_id, "pr_number": pr,
                       "merge_sha": "a" * 40, "merged_at": "2026-11-05T10:00:00+00:00", "title": "t",
                       "flags": {}, "body_map": [], "note": note, "written_at": "2026-11-05T11:00:00+00:00"})


class World:
    def __init__(self, ledger_dir: Path, replies=None, *, write_on_turn=True, max_attempts=3):
        self.now = datetime(2026, 11, 5, tzinfo=timezone.utc)
        self.granted = False
        self.ledger = FieldNoteLedger(ledger_dir)
        self.replies = list(replies or [f"field-note #{PR} saved {NOTE_ID}"])
        self.write_on_turn = write_on_turn
        self.turns: list[CuriosityTurnRequestV1] = []
        self.registers = 0
        self.releases: list[str] = []
        self.max_attempts = max_attempts

    def write(self, pr, note_id, **kw):
        with self.ledger.path.open("a") as fh:
            fh.write(note_line(pr, note_id, **kw) + "\n")

    def grant(self):
        self.granted = True

    async def turn(self, req: CuriosityTurnRequestV1) -> CuriosityTurnResultV1:
        self.turns.append(req)
        reply = self.replies.pop(0) if self.replies else f"field-note #{PR} saved {NOTE_ID}"
        if isinstance(reply, Exception):
            raise reply
        if self.write_on_turn and "saved" in reply and reply.endswith(NOTE_ID):
            self.write(PR, NOTE_ID)   # the dig write the turn ran
        return CuriosityTurnResultV1(run_id=req.run_id, correlation_id=req.correlation_id, text=reply, ok=bool(reply),
                                     debug={"harness_step_count": 9, "elapsed_sec": 80.0})

    async def register(self, state):
        self.registers += 1
        return {"status": "waiting_resource", "hold": {"request_id": f"{RUN_ID}:1", "lease_id": "hold-fn"}, "hold_seq": 1}

    async def lease(self, state):
        if self.granted:
            return GRANTED, {"status": "admitted", "lease": dict(REF), "hold": state.get("hold")}
        return WAITING, {"status": "waiting_resource", "lease": None}

    async def execute(self, state, node):
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
        return build_pr_field_note_graph(
            self.turn,
            AdmissionDeps(self.register, self.lease, self.execute, self.release, self.event,
                          now=lambda: self.now, max_attempts=self.max_attempts, retry_base_seconds=10.0),
            saver, ledger=self.ledger)


def initial():
    return {"run_id": RUN_ID, "correlation_id": "trace-fn", "workflow": PR_FIELD_NOTE_WORKFLOW, "attempt": 0,
            "admission": {"resource": "llm.route.agent"},
            "brief": {"field_note": "pr.field_note", "pr_number": PR, "merged_at": "2026-11-05T10:00:00+00:00",
                      "title": "fix(vision-retina): parse dh out of FCM headers", "timeout_sec": 420}}


async def start_and_grant(world, saver):
    await asyncio.wait_for(world.graph(saver).ainvoke(initial(), CFG), 2)
    snap = await world.graph(saver).aget_state(CFG)
    assert snap.next == ("resource_wait",) and world.turns == []
    world.grant()
    return await asyncio.wait_for(world.graph(saver).ainvoke(Command(resume={}), CFG), 2)


async def after_backoff(world, saver):
    snap = await world.graph(saver).aget_state(CFG)
    assert snap.next == ("retry_wait",)
    world.now = datetime.fromisoformat(snap.values["retry_at"]) + timedelta(seconds=1)
    world.grant()
    return await asyncio.wait_for(world.graph(saver).ainvoke(Command(resume=True), CFG), 2)


def test_a_receipt_backed_by_the_ledger_completes_the_run(tmp_path):
    async def scenario():
        world, saver = World(tmp_path), InMemorySaver()
        result = await start_and_grant(world, saver)
        assert result["status"] == "completed" and result["outcome"] == "noted" and result["note_id"] == NOTE_ID
        [req] = world.turns
        assert req.gpu_lease is not None and req.session_id == PR_FIELD_NOTE_SESSION_ID
        assert req.source_tag == PR_FIELD_NOTE_TAG and req.timeout_sec == pytest.approx(390.0)
        record = result["steps"][PR_FIELD_NOTE_STEP_KEY]
        assert record["note_id"] == NOTE_ID and record["attempts"] == 1
        assert record["debug"]["harness_step_count"] == 9    # real step use, measured per note
        detail = finish_detail(result)
        assert detail["line"] == "pr_field_note" and detail["pr_number"] == PR and detail["note_id"] == NOTE_ID
        assert detail["turn_correlation_id"] == req.correlation_id
        assert world.releases == ["completed"]
    asyncio.run(scenario())


def test_the_prompt_is_lean_and_spells_out_every_command():
    p = build_prompt(PrFieldNoteBriefV1.model_validate(initial()["brief"]))
    assert f"python -P -m orion.dig field-note {PR}`" in p
    assert f"python -P -m orion.dig field-note {PR} write" in p
    assert f"field-note #{PR} saved <note_id>" in p
    assert "ONLY when a sentence claims something about code that the packet does not state" in p
    assert f"never more than {TOOL_CALLS_MAX}" in p and len(p) < 2600


def test_a_pr_with_a_record_takes_no_hold_and_no_turn(tmp_path):
    async def scenario():
        world, saver = World(tmp_path), InMemorySaver()
        world.write(PR, "fn:3105:aaaaaaaaaa", note=None)    # e.g. the poller's code record
        result = await asyncio.wait_for(world.graph(saver).ainvoke(initial(), CFG), 2)
        assert result["status"] == "completed" and result["outcome"] == "already_recorded"
        assert result["prior_record"]["has_note"] is False
        assert world.turns == [] and world.registers == 0 and world.releases == []
    asyncio.run(scenario())


def test_a_reply_without_a_receipt_is_retried_never_completed(tmp_path):
    async def scenario():
        world, saver = World(tmp_path, replies=["I wrote the note, all done!", f"field-note #{PR} saved {NOTE_ID}"]), InMemorySaver()
        r1 = await start_and_grant(world, saver)
        assert r1["status"] == "retrying" and "field_note_no_receipt" in r1["last_error"]
        assert PR_FIELD_NOTE_STEP_KEY not in (r1.get("steps") or {})
        r2 = await after_backoff(world, saver)
        assert r2["status"] == "completed" and r2["steps"][PR_FIELD_NOTE_STEP_KEY]["attempts"] == 2
        assert world.turns[1].attempt == 2
    asyncio.run(scenario())


def test_a_receipt_the_ledger_does_not_hold_fails_after_the_budget(tmp_path):
    async def scenario():
        fake = f"field-note #{PR} saved fn:{PR}:ffffffffff"
        world, saver = World(tmp_path, replies=[fake] * 3, write_on_turn=False), InMemorySaver()
        r = await start_and_grant(world, saver)
        assert r["status"] == "retrying" and "receipt_not_in_ledger" in r["last_error"]
        r = await after_backoff(world, saver)
        r = await after_backoff(world, saver)
        assert r["status"] == "failed" and "receipt_not_in_ledger" in r["last_error"]
        assert not r.get("steps") and r.get("outcome") is None
    asyncio.run(scenario())


def test_an_unmounted_ledger_fails_the_run_before_any_hold(tmp_path):
    async def scenario():
        world, saver = World(tmp_path / "absent"), InMemorySaver()
        result = await asyncio.wait_for(world.graph(saver).ainvoke(initial(), CFG), 2)
        assert result["status"] == "failed" and "ledger_unreadable" in result["last_error"]
        assert world.registers == 0 and world.turns == []
    asyncio.run(scenario())


def test_verify_receipt_checks_the_pr_and_the_ledger(tmp_path):
    ok = lambda pr, nid: (pr, nid) == (PR, NOTE_ID)  # noqa: E731
    assert verify_receipt(f"done.\nfield-note #{PR} saved {NOTE_ID}\n", PR, ok) == NOTE_ID
    with pytest.raises(HarnessStepFailed, match="no_receipt"):
        verify_receipt(f"field-note #3999 saved fn:3999:0123456789", PR, ok)   # another PR's receipt
    with pytest.raises(HarnessStepFailed, match="no_receipt"):
        verify_receipt("", PR, ok)
    with pytest.raises(HarnessStepFailed, match="not_in_ledger"):
        verify_receipt(f"field-note #{PR} saved fn:{PR}:9999999999", PR, ok)


def test_ledger_receipt_ok_wants_a_note_for_this_pr(tmp_path):
    world = World(tmp_path)
    world.write(PR, NOTE_ID)
    world.write(3200, "fn:3200:1111111111", note=None)
    assert world.ledger.receipt_ok(PR, NOTE_ID) is True
    assert world.ledger.receipt_ok(3999, NOTE_ID) is False
    assert world.ledger.receipt_ok(3200, "fn:3200:1111111111") is False   # a code record is not a note


def test_a_retry_after_the_note_was_saved_runs_no_second_turn(tmp_path):
    """Attempt 1 saved the note but replied without a usable receipt. The retry looks at the ledger
    first and finishes: no resumed turn that could only repeat the receipt (too few steps for Hub)."""
    async def scenario():
        world, saver = World(tmp_path, replies=["Saved it."]), InMemorySaver()
        real_turn = world.turn

        async def writes_then_mumbles(req):
            world.write(PR, NOTE_ID)
            return await real_turn(req)
        world.turn = writes_then_mumbles
        r1 = await start_and_grant(world, saver)
        assert r1["status"] == "retrying" and len(world.turns) == 1
        snap = await world.graph(saver).aget_state(CFG)
        world.now = datetime.fromisoformat(snap.values["retry_at"]) + timedelta(seconds=1)
        r2 = await asyncio.wait_for(world.graph(saver).ainvoke(Command(resume=True), CFG), 2)
        assert r2["status"] == "completed" and r2["outcome"] == "already_recorded" and len(world.turns) == 1
        assert r2["prior_record"]["note_id"] == NOTE_ID and r2["prior_record"]["has_note"] is True
    asyncio.run(scenario())
