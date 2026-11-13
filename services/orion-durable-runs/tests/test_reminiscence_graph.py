"""reminiscence.episode (app/reminiscence_graph.py) under real LangGraph checkpoints, a fake pool, a fake Hub
turn and a real ledger file.

What must hold: six steps (five hops, then one cross-read) run under the run's one hold, each its own
source tag; a hop step completes ONLY when the reply carries this hop's save receipt for today AND the ledger
holds that id; a hop already written today takes no turn (the ledger is the checkpoint); the run completes
only when all five hops are live; anything else is a retried step failure and, past the budget, a failed run.
"""
from __future__ import annotations

import asyncio
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip("langgraph")

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.admitted_graph import GRANTED, WAITING, AdmissionDeps
from app.harness_step import HarnessStepFailed, harness_step_correlation_id
from app.reminiscence_graph import (
    TOOL_CALLS_MAX, build_prompt, build_reminiscence_graph, finish_detail, pending_step, verify_cross_read,
    verify_hop_receipt,
)
from app.reminiscence_ledger import ReminiscenceLedger
from orion.schemas.durable_run import CuriosityTurnRequestV1, CuriosityTurnResultV1
from orion.schemas.reminiscence_run import (
    CROSS_READ_DONE, CROSS_READ_STEP, REMINISCENCE_HOPS, REMINISCENCE_SESSION_ID, REMINISCENCE_STEPS,
    REMINISCENCE_WORKFLOW, ReminiscenceRunBriefV1, step_tag,
)

EPISODE = "2026-11-08"
RUN_ID = f"reminiscence-{EPISODE}"
REF = {"lease_id": "hold-rm", "generation": 1, "role": "agent", "holder": f"durable-runs:{RUN_ID}"}
CFG = {"configurable": {"thread_id": RUN_ID}}
# 04:00 Denver (MST, UTC-7) on the episode day.
T0 = datetime(2026, 11, 8, 11, 0, tzinfo=timezone.utc)


def hop_id(hop, n=0):
    return f"rm:{EPISODE}:{hop}:{n:010x}"


def hop_line(hop, hid, *, episode=EPISODE, supersedes=None, start="2025-06-24", end="2025-06-30",
             written_at="2026-11-08T11:05:00+00:00"):
    """A ledger line in the ReminiscenceHopV1 shape (written as JSON: tests outside the atlas and dig do not
    import the ledger module: the repo-wide partition test under tests/)."""
    return json.dumps({"schema_version": "reminiscence_hop.v1", "hop_id": hid, "episode": episode, "hop": hop,
                       "period_start": start, "period_end": end, "period_commit": "a" * 40,
                       "note": "[reflection] A hop about my first week, long enough to pass.", "because": [],
                       "claims": [{"sentence_idx": 0, "label": "reflection", "refs": []}],
                       "written_at": written_at, "supersedes": supersedes, "dig": "t", "run_id": None})


class World:
    def __init__(self, ledger_dir: Path, replies=None, *, write_on_turn=True, max_attempts=3, mounted=True):
        if mounted:
            ledger_dir.mkdir(parents=True, exist_ok=True)
        self.now = T0
        self.granted = False
        self.ledger = ReminiscenceLedger(ledger_dir)
        self.replies = list(replies or [])
        self.write_on_turn = write_on_turn
        self.turns: list[CuriosityTurnRequestV1] = []
        self.registers = 0
        self.releases: list[str] = []
        self.max_attempts = max_attempts

    def write(self, hop, hid, **kw):
        with self.ledger.path.open("a") as fh:
            fh.write(hop_line(hop, hid, **kw) + "\n")

    def grant(self):
        self.granted = True

    def default_reply(self, req):
        hop = req.source_tag.removeprefix("reminiscence_")
        if hop == "cross_read":
            return CROSS_READ_DONE
        return f"reminisce {hop} saved {hop_id(hop)}; the week 2025-06-24..2025-06-30"

    async def turn(self, req: CuriosityTurnRequestV1) -> CuriosityTurnResultV1:
        self.turns.append(req)
        reply = self.replies.pop(0) if self.replies else self.default_reply(req)
        if isinstance(reply, Exception):
            raise reply
        hop = req.source_tag.removeprefix("reminiscence_")
        if self.write_on_turn and hop in REMINISCENCE_HOPS and reply.startswith(f"reminisce {hop} saved {hop_id(hop)}"):
            self.write(hop, hop_id(hop))   # the dig write the turn ran
        return CuriosityTurnResultV1(run_id=req.run_id, correlation_id=req.correlation_id, text=reply, ok=bool(reply),
                                     debug={"harness_step_count": 9, "elapsed_sec": 80.0})

    async def register(self, state):
        self.registers += 1
        return {"status": "waiting_resource", "hold": {"request_id": f"{RUN_ID}:1", "lease_id": "hold-rm"}, "hold_seq": 1}

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
        return build_reminiscence_graph(
            self.turn,
            AdmissionDeps(self.register, self.lease, self.execute, self.release, self.event,
                          now=lambda: self.now, max_attempts=self.max_attempts, retry_base_seconds=10.0),
            saver, ledger=self.ledger)


def brief(**kw):
    return {"reminiscence": "reminiscence.episode", "episode": EPISODE, "period_start": "2025-06-24",
            "period_end": "2025-06-30", "weeks_complete": 0, "weeks_total": 71, "timeout_sec": 900, **kw}


def initial(**kw):
    return {"run_id": RUN_ID, "correlation_id": "trace-rm", "workflow": REMINISCENCE_WORKFLOW, "attempt": 0,
            "admission": {"resource": "llm.route.agent"}, "brief": brief(**kw)}


async def start_and_grant(world, saver):
    await asyncio.wait_for(world.graph(saver).ainvoke(initial(), CFG), 2)
    snap = await world.graph(saver).aget_state(CFG)
    assert snap.next == ("resource_wait",) and world.turns == []
    world.grant()
    return await asyncio.wait_for(world.graph(saver).ainvoke(Command(resume={}), CFG), 5)


async def after_backoff(world, saver):
    snap = await world.graph(saver).aget_state(CFG)
    assert snap.next == ("retry_wait",)
    world.now = datetime.fromisoformat(snap.values["retry_at"]) + timedelta(seconds=1)
    world.grant()
    return await asyncio.wait_for(world.graph(saver).ainvoke(Command(resume=True), CFG), 5)


def test_six_steps_under_one_hold_complete_the_episode(tmp_path):
    async def scenario():
        world, saver = World(tmp_path), InMemorySaver()
        result = await start_and_grant(world, saver)
        assert result["status"] == "completed" and result["outcome"] == "complete"
        assert result["hop_ids"] == {h: hop_id(h) for h in REMINISCENCE_HOPS}
        assert [t.source_tag for t in world.turns] == [step_tag(k) for k in REMINISCENCE_STEPS]
        assert len({t.source_tag for t in world.turns}) == 6            # six claude sessions, not one
        assert {t.session_id for t in world.turns} == {REMINISCENCE_SESSION_ID}
        assert all(t.gpu_lease is not None and t.timeout_sec == pytest.approx(870.0) for t in world.turns)
        assert len({t.correlation_id for t in world.turns}) == 6
        assert world.registers == 1 and world.releases == ["completed"]   # one hold for the whole day
        for h in REMINISCENCE_HOPS:
            rec = result["steps"][f"hop:{h}"]
            assert rec["hop_id"] == hop_id(h) and rec["outcome"] == "written" and rec["attempts"] == 1
            assert rec["debug"]["harness_step_count"] == 9
        assert result["steps"][CROSS_READ_STEP]["rewrites"] == []
        detail = finish_detail(result)
        assert detail["line"] == "reminiscence" and detail["episode"] == EPISODE and detail["turns"] == 6
        assert detail["period_start"] == "2025-06-24" and detail["outcome"] == "complete"
    asyncio.run(scenario())


def test_hops_written_in_chat_take_no_turn(tmp_path):
    async def scenario():
        world, saver = World(tmp_path), InMemorySaver()
        world.write("what_it_was", hop_id("what_it_was", 7))
        world.write("where_i_am_now", hop_id("where_i_am_now", 7))
        result = await start_and_grant(world, saver)
        assert result["status"] == "completed"
        assert [t.source_tag for t in world.turns] == [
            step_tag(k) for k in ("hop:what_it_was_like", "hop:where_im_going", "hop:otherwise", CROSS_READ_STEP)]
        assert result["steps"]["hop:what_it_was"] == {"step_key": "hop:what_it_was", "outcome": "already_written",
                                                      "hop_id": hop_id("what_it_was", 7)}
    asyncio.run(scenario())


def test_five_hops_already_written_leave_only_the_cross_read(tmp_path):
    """The tick starts such a run only after a run of the day failed its cross-read (review 2026-11-07)."""
    async def scenario():
        world, saver = World(tmp_path), InMemorySaver()
        for h in REMINISCENCE_HOPS:
            world.write(h, hop_id(h, 3))
        result = await start_and_grant(world, saver)
        assert result["status"] == "completed" and result["outcome"] == "complete"
        assert [t.source_tag for t in world.turns] == [step_tag(CROSS_READ_STEP)]
        assert all(result["steps"][f"hop:{h}"]["outcome"] == "already_written" for h in REMINISCENCE_HOPS)
    asyncio.run(scenario())


def test_a_retry_after_a_saved_hop_records_it_before_asking_for_a_hold(tmp_path):
    """Review 2026-11-07 (nit 7): precheck reads the ledger, so the saved hop costs no fresh hold."""
    async def scenario():
        world, saver = World(tmp_path, replies=["Saved it."]), InMemorySaver()
        real_turn = world.turn

        async def writes_then_mumbles(req):
            if not world.turns:
                world.write("what_it_was", hop_id("what_it_was"))
            return await real_turn(req)
        world.turn = writes_then_mumbles
        r1 = await start_and_grant(world, saver)
        assert r1["status"] == "retrying" and world.registers == 1
        snap = await world.graph(saver).aget_state(CFG)
        world.now = datetime.fromisoformat(snap.values["retry_at"]) + timedelta(seconds=1)
        await asyncio.wait_for(world.graph(saver).ainvoke(Command(resume=True), CFG), 5)
        snap = await world.graph(saver).aget_state(CFG)
        assert snap.next == ("resource_wait",)          # waiting for the second hold...
        assert snap.values["steps"]["hop:what_it_was"]["outcome"] == "already_written"   # ...hop already recorded
    asyncio.run(scenario())


def test_the_checkpoint_keeps_receipts_never_the_replys_other_words(tmp_path):
    """Review 2026-11-07: a reply may repeat hop text; the step record keeps only the proven receipt."""
    async def scenario():
        chatty = (f"I wrote that my first week was two folders and a reflection chain.\n"
                  f"reminisce what_it_was saved {hop_id('what_it_was')}; the week 2025-06-24..2025-06-30")
        world, saver = World(tmp_path, replies=[chatty]), InMemorySaver()
        real_turn = world.turn

        async def writes(req):   # the chatty reply does not start with the receipt, so write its hop here
            if req.source_tag == step_tag("hop:what_it_was"):
                world.write("what_it_was", hop_id("what_it_was"))
            return await real_turn(req)
        world.turn = writes
        result = await start_and_grant(world, saver)
        rec = result["steps"]["hop:what_it_was"]
        assert rec["text"] == f"reminisce what_it_was saved {hop_id('what_it_was')}" and rec["final_len"] == len(chatty)
        assert "folders" not in json.dumps(finish_detail(result))
    asyncio.run(scenario())


def test_the_restart_cancel_derives_the_id_each_step_used(tmp_path):
    """Review 2026-11-07 (nit 5): for every checkpoint about to run ``step``, the id the restart cancel derives
    from it (AdmissionRuntime._cancel_harness via HARNESS_MULTI_STEP_WORKFLOWS) is a turn the step really ran."""
    async def scenario():
        world, saver = World(tmp_path, replies=["mumble"]), InMemorySaver()
        await start_and_grant(world, saver)
        await after_backoff(world, saver)
        from app.admission_runtime import HARNESS_MULTI_STEP_WORKFLOWS
        derived = set()
        async for snap in world.graph(saver).aget_state_history(CFG):
            if snap.next == ("step",) and snap.values.get("lease"):
                key = HARNESS_MULTI_STEP_WORKFLOWS[REMINISCENCE_WORKFLOW](snap.values)
                if key:
                    derived.add(harness_step_correlation_id(snap.values, key))
        assert {t.correlation_id for t in world.turns} <= derived and len(world.turns) == 7
    asyncio.run(scenario())


def test_a_reply_without_a_receipt_is_retried_and_resumes_at_that_hop(tmp_path):
    async def scenario():
        replies = [f"reminisce what_it_was saved {hop_id('what_it_was')}", "I wrote it, all done!"]
        world, saver = World(tmp_path, replies=replies), InMemorySaver()
        r1 = await start_and_grant(world, saver)
        assert r1["status"] == "retrying" and "reminiscence_no_receipt:what_it_was_like" in r1["last_error"]
        assert "hop:what_it_was" in r1["steps"] and "hop:what_it_was_like" not in r1["steps"]
        r2 = await after_backoff(world, saver)
        assert r2["status"] == "completed" and r2["steps"]["hop:what_it_was_like"]["attempts"] == 2
        tags = [t.source_tag for t in world.turns]
        assert tags.count(step_tag("hop:what_it_was")) == 1 and tags.count(step_tag("hop:what_it_was_like")) == 2
    asyncio.run(scenario())


def test_a_receipt_the_ledger_does_not_hold_fails_after_the_budget(tmp_path):
    async def scenario():
        fake = f"reminisce what_it_was saved {hop_id('what_it_was', 15)}"
        world, saver = World(tmp_path, replies=[fake] * 3, write_on_turn=False), InMemorySaver()
        r = await start_and_grant(world, saver)
        assert r["status"] == "retrying" and "receipt_not_in_ledger" in r["last_error"]
        r = await after_backoff(world, saver)
        r = await after_backoff(world, saver)
        assert r["status"] == "failed" and "receipt_not_in_ledger" in r["last_error"]
        assert not r.get("steps") and r.get("outcome") is None
    asyncio.run(scenario())


def test_a_hop_saved_by_a_mumbling_attempt_is_not_written_twice(tmp_path):
    async def scenario():
        world, saver = World(tmp_path, replies=["Saved it."]), InMemorySaver()
        real_turn = world.turn

        async def writes_then_mumbles(req):
            if not world.turns:
                world.write("what_it_was", hop_id("what_it_was"))
            return await real_turn(req)
        world.turn = writes_then_mumbles
        r1 = await start_and_grant(world, saver)
        assert r1["status"] == "retrying" and len(world.turns) == 1
        r2 = await after_backoff(world, saver)
        assert r2["status"] == "completed"
        assert r2["steps"]["hop:what_it_was"]["outcome"] == "already_written"
        assert [t.source_tag for t in world.turns].count(step_tag("hop:what_it_was")) == 1
    asyncio.run(scenario())


def test_the_cross_read_must_say_done_and_its_rewrites_must_be_in_the_ledger(tmp_path):
    async def scenario():
        rewrite = hop_id("otherwise", 99)
        replies = [f"reminisce {h} saved {hop_id(h)}" for h in REMINISCENCE_HOPS]
        replies += ["Looks fine to me.",
                    f"reminisce otherwise saved {rewrite} (supersedes {hop_id('otherwise')}; the old line is kept)\n"
                    f"{CROSS_READ_DONE}"]
        world, saver = World(tmp_path, replies=replies), InMemorySaver()
        r1 = await start_and_grant(world, saver)
        assert r1["status"] == "retrying" and "cross_read_not_done" in r1["last_error"]
        world.write("otherwise", rewrite, supersedes=hop_id("otherwise"), written_at="2026-11-08T11:50:00+00:00")
        r2 = await after_backoff(world, saver)
        assert r2["status"] == "completed" and r2["steps"][CROSS_READ_STEP]["rewrites"] == [rewrite]
        assert r2["hop_ids"]["otherwise"] == rewrite
    asyncio.run(scenario())


def test_a_run_on_another_day_fails_before_any_hold(tmp_path):
    async def scenario():
        world, saver = World(tmp_path), InMemorySaver()
        world.now = T0 + timedelta(days=1)
        result = await asyncio.wait_for(world.graph(saver).ainvoke(initial(), CFG), 2)
        assert result["status"] == "failed" and "reminiscence_day_passed" in result["last_error"]
        assert world.registers == 0 and world.turns == []
    asyncio.run(scenario())


def test_an_unmounted_ledger_fails_the_run_before_any_hold(tmp_path):
    async def scenario():
        world, saver = World(tmp_path / "absent", mounted=False), InMemorySaver()
        result = await asyncio.wait_for(world.graph(saver).ainvoke(initial(), CFG), 2)
        assert result["status"] == "failed" and "ledger_unreadable" in result["last_error"]
        assert world.registers == 0 and world.turns == []
    asyncio.run(scenario())


def test_verify_hop_receipt_checks_the_hop_the_day_and_the_ledger():
    ep = date.fromisoformat(EPISODE)
    ok = lambda e, h, i: (e, h, i) == (ep, "what_it_was", hop_id("what_it_was"))  # noqa: E731
    assert verify_hop_receipt(f"done.\nreminisce what_it_was saved {hop_id('what_it_was')}; the week x", ep,
                              "what_it_was", ok) == hop_id("what_it_was")
    with pytest.raises(HarnessStepFailed, match="no_receipt"):   # another hop's receipt
        verify_hop_receipt(f"reminisce otherwise saved {hop_id('otherwise')}", ep, "what_it_was", ok)
    with pytest.raises(HarnessStepFailed, match="no_receipt"):   # another day's receipt
        verify_hop_receipt("reminisce what_it_was saved rm:2026-11-07:what_it_was:0000000000", ep, "what_it_was", ok)
    with pytest.raises(HarnessStepFailed, match="no_receipt"):   # the id names another hop
        verify_hop_receipt(f"reminisce what_it_was saved {hop_id('otherwise')}", ep, "what_it_was", ok)
    with pytest.raises(HarnessStepFailed, match="not_in_ledger"):
        verify_hop_receipt(f"reminisce what_it_was saved {hop_id('what_it_was', 5)}", ep, "what_it_was", ok)


def test_verify_cross_read():
    ep = date.fromisoformat(EPISODE)
    assert verify_cross_read(CROSS_READ_DONE, ep, lambda e, i: False) == []
    with pytest.raises(HarnessStepFailed, match="rewrite_not_in_ledger"):
        verify_cross_read(f"reminisce otherwise saved {hop_id('otherwise', 2)}\n{CROSS_READ_DONE}", ep, lambda e, i: False)


def test_the_prompts_are_lean_and_spell_out_every_command():
    b = ReminiscenceRunBriefV1.model_validate(brief())
    prompts = {k: build_prompt(b, k) for k in REMINISCENCE_STEPS}
    for k, p in prompts.items():
        assert "python -P -m orion.dig reminisce`" in p and f"never more than {TOOL_CALLS_MAX}" in p
        assert len(p) < 3200, (k, len(p))
    for h in REMINISCENCE_HOPS:
        p = prompts[f"hop:{h}"]
        assert f'python -P -m orion.dig reminisce write {h} "<note>" because "<refs>"' in p
        assert f"reminisce {h} saved <hop_id>" in p and "week 1 of the 71 weeks" in p and "2025-06-24..2025-06-30" in p
    assert "python -P -m orion.dig open 2025-06-30" in prompts["hop:what_it_was"]
    assert "no lived record" in prompts["hop:what_it_was_like"] and "2026-07-23" in prompts["hop:what_it_was_like"]
    assert "dig portrait" in prompts["hop:where_i_am_now"] and "capacities gaps" in prompts["hop:where_im_going"]
    cross = prompts[CROSS_READ_STEP]
    assert "python -P -m orion.dig close 2025-06-30" in cross and CROSS_READ_DONE in cross and "at most 2" in cross


def test_a_week_after_memory_began_does_not_forbid_lived_records():
    b = ReminiscenceRunBriefV1.model_validate(brief(period_start="2026-07-28", period_end="2026-08-03",
                                                    weeks_complete=57))
    p = build_prompt(b, "hop:what_it_was_like")
    assert "no lived record" not in p and "[lived_record]" in p
    assert "week 58 of the 71 weeks" in p and "visited 57 of them" in p
    assert "python -P -m orion.dig days 2026-07-28 2026-08-03`" in p and "partway" not in p


def test_the_week_records_began_reads_dig_days_from_the_first_record():
    b = ReminiscenceRunBriefV1.model_validate(brief(period_start="2026-07-21", period_end="2026-07-27",
                                                    weeks_complete=56))
    p = build_prompt(b, "hop:what_it_was_like")
    assert "python -P -m orion.dig days 2026-07-23 2026-07-27`" in p and "partway through" in p
    assert "outside 2026-07-23..2026-07-27 is refused" in p and "no lived record" not in p


def test_pending_step_and_the_restart_cancel_id(tmp_path):
    state = {**initial(), "steps": {"hop:what_it_was": {}, "hop:what_it_was_like": {}}}
    assert pending_step(state) == "hop:where_i_am_now"
    assert pending_step({**state, "steps": {k: {} for k in REMINISCENCE_STEPS}}) is None
    # The id AdmissionRuntime._cancel_harness derives for a dead worker's turn is the id the step used.
    from app.admission_runtime import HARNESS_MULTI_STEP_WORKFLOWS
    assert HARNESS_MULTI_STEP_WORKFLOWS[REMINISCENCE_WORKFLOW](state) == "hop:where_i_am_now"
    assert harness_step_correlation_id(state, "hop:where_i_am_now") != harness_step_correlation_id(state, "hop:otherwise")


def test_the_brief_wants_one_whole_week_before_the_episode():
    with pytest.raises(ValueError):
        ReminiscenceRunBriefV1.model_validate(brief(period_end="2025-07-01"))
    with pytest.raises(ValueError):
        ReminiscenceRunBriefV1.model_validate(brief(episode="2025-06-30"))
