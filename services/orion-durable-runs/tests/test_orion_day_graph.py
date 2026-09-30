"""orion_day.letter admitted graph: wait for the pool's hold (never an attempt), write the note,
write the carry-forward (hold released right after), persist once, journal the NOTE only.

Resume per node, idempotent persist, and note/carry-forward separation are the contract.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip("langgraph")

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parents[1])]

from app.admitted_graph import GRANTED, WAITING, AdmissionDeps, HoldLost
from app.orion_day_graph import (
    MIN_NOTE_CHARS, OrionDayDeps, build_orion_day_graph, finish_detail, slim_brief,
)
from orion.schemas.orion_day import (
    ORION_DAY_CARRY_FORWARD_VERB, ORION_DAY_NOTE_VERB, OrionDayLlmViewV1, OrionDayMaterialV1, OrionDayRunBriefV1,
    orion_day_journal_entry_id,
)

REF = {"lease_id": "hold-od", "generation": 1, "role": "agent", "holder": "durable-runs:orion-day-2026-09-29-1"}
RUN = "orion-day-2026-09-29-1"
CFG = {"configurable": {"thread_id": RUN}}
NOTE = "Yesterday I spent most of my attention on the hop written_at prior. " * 20
CARRY = "- [curiosity:ab61e4ccd47b] Test whether written_at is ever populated on hop nodes."


class Crash(BaseException):
    """A process death mid-node: escapes every except Exception, leaves the checkpoint behind."""


def make_brief() -> OrionDayRunBriefV1:
    start = datetime(2026, 9, 29, 6, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    material = OrionDayMaterialV1(letter_date=date(2026, 9, 29), window_start=start, window_end=end,
                                  gathered_at=end + timedelta(hours=8))
    view = OrionDayLlmViewV1(digest_md="# My day: 2026-09-29\n\n### [curiosity:ab61e4ccd47b] Curiosity\nfound x",
                             approx_tokens=20, budget_tokens=70000, chars_per_token=3.5,
                             included_refs=["curiosity:ab61e4ccd47b"])
    return OrionDayRunBriefV1(letter_date=date(2026, 9, 29), window_start=start, window_end=end,
                              material=material, llm_view=view, timeout_sec=900.0)


class World:
    def __init__(self, *, max_attempts=3):
        self.now = datetime(2026, 9, 30, 14, tzinfo=timezone.utc)
        self.granted = False
        self.brief = make_brief()
        self.calls: list[tuple[str, dict, dict]] = []
        self.replies: dict[str, list] = {ORION_DAY_NOTE_VERB: [], ORION_DAY_CARRY_FORWARD_VERB: []}
        self.releases: list[str] = []
        self.registers = 0
        self.execute_lost: list[bool] = []
        self.rows: dict[date, dict] = {}
        self.persist_fail = 0
        self.journals = []
        self.journal_fail = 0
        self.max_attempts = max_attempts

    # --- deps -------------------------------------------------------------------------------
    async def call_verb_text(self, verb, metadata, llm_route, *, gpu_lease, timeout_sec, user_text):
        self.calls.append((verb, metadata, {"route": llm_route, "lease": gpu_lease, "timeout": timeout_sec}))
        reply = self.replies[verb].pop(0) if self.replies[verb] else (NOTE if verb == ORION_DAY_NOTE_VERB else CARRY)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    async def persist_letter(self, row):
        self.hold_at_persist = self.granted
        if self.persist_fail:
            self.persist_fail -= 1
            raise RuntimeError("db down")
        existing = self.rows.setdefault(row["letter_date"], row)
        return {k: existing[k] for k in ("run_id", "note_md", "created_at")}

    async def publish_journal(self, entry):
        if self.journal_fail:
            self.journal_fail -= 1
            return None
        self.journals.append(entry)
        return entry.entry_id

    async def load_brief(self, state):
        return self.brief

    # --- admission --------------------------------------------------------------------------
    async def register(self, state):
        self.registers += 1
        return {"status": "waiting_resource", "hold": {"request_id": f"{RUN}:1", "lease_id": "hold-od"}, "hold_seq": 1}

    async def lease(self, state):
        if self.granted:
            return GRANTED, {"status": "admitted", "lease": dict(REF), "hold": state.get("hold")}
        return WAITING, {"status": "waiting_resource", "lease": None}

    async def execute(self, state, node):
        if not self.granted or state.get("lease") != REF:
            raise RuntimeError("gpu_hold_missing")
        if self.execute_lost and self.execute_lost.pop(0):
            raise HoldLost("gpu_hold_lost:hold-od")
        return await node(state)

    async def release(self, state, reason, keep_requeued=False):
        self.releases.append(reason)
        if state.get("lease") or state.get("hold"):
            self.granted = False
        return {"lease": None, "hold": None}

    async def event(self, *args):
        pass

    def graph(self, saver):
        return build_orion_day_graph(
            OrionDayDeps(call_verb_text=self.call_verb_text, persist_letter=self.persist_letter,
                         publish_journal=self.publish_journal, load_brief=self.load_brief),
            AdmissionDeps(self.register, self.lease, self.execute, self.release, self.event,
                          now=lambda: self.now, max_attempts=self.max_attempts,
                          retry_base_seconds=30.0, retry_max_seconds=300.0),
            saver,
        )


def initial():
    return {"run_id": RUN, "correlation_id": RUN, "attempt": 0, "workflow": "orion_day.letter",
            "admission": {"resource": "llm.route.agent"},
            "brief": slim_brief(make_brief().model_dump(mode="json"))}


async def drive(graph, world, first=True, grant_after=0):
    """Run until the graph ends or parks; grant the hold after `grant_after` parked waits."""
    parks = 0
    payload = initial() if first else None
    while True:
        await graph.ainvoke(payload, CFG)
        snap = await graph.aget_state(CFG)
        if not snap.next:
            return snap
        if snap.next == ("resource_wait",):
            parks += 1
            if parks > grant_after:
                world.granted = True
        elif snap.next == ("retry_wait",):
            world.now = datetime.fromisoformat(snap.values["retry_at"]) + timedelta(seconds=1)
        payload = Command(resume=True)


def test_waits_without_spending_attempts_then_writes_note_carry_forward_and_letter():
    async def scenario():
        world, saver = World(), InMemorySaver()
        snap = await drive(world.graph(saver), world, grant_after=3)
        values = snap.values
        assert values["status"] == "completed"
        assert values["llm_attempts"] == {"write_note": 1, "write_carry_forward": 1}  # waiting never counted
        assert values["note_md"] == NOTE.strip() and values["carry_forward_md"] == CARRY
        assert [c[0] for c in world.calls] == [ORION_DAY_NOTE_VERB, ORION_DAY_CARRY_FORWARD_VERB]
        for _, _, extra in world.calls:
            assert extra["route"] == "agent" and extra["timeout"] == 870.0
            assert extra["lease"].lease_id == "hold-od"  # attached to the run's hold
        # Hold handed back as soon as the carry-forward text was checkpointed, before persist.
        assert world.releases[0] == "completed" and world.hold_at_persist is False
        row = world.rows[date(2026, 9, 29)]
        assert row["run_id"] == RUN and row["note_md"] == NOTE.strip() and row["carry_forward_md"] == CARRY
        assert row["journal_entry_id"] == orion_day_journal_entry_id("2026-09-29")
        assert row["carry_forward_expires_at"] == world.now + timedelta(hours=48)
        detail = finish_detail(values)
        assert detail["persisted"] is True and detail["persist_outcome"] == "written"
        assert "note_md" not in detail and "carry_forward_md" not in detail  # bounded: no texts
    asyncio.run(scenario())


def test_note_and_carry_forward_stay_separate():
    async def scenario():
        world, saver = World(), InMemorySaver()
        await drive(world.graph(saver), world)
        (_, note_meta, _), (_, cf_meta, _) = world.calls
        assert set(note_meta["orion_day_input"]) == {"letter_date", "timezone", "digest_md"}
        assert "note_md" not in note_meta["orion_day_input"]
        assert cf_meta["orion_day_input"]["note_md"] == NOTE.strip()
        assert note_meta["orion_day_input"]["digest_md"] == cf_meta["orion_day_input"]["digest_md"]
        # Journal: the NOTE only, under the stable id and the new trigger kind.
        (entry,) = world.journals
        assert entry.body == NOTE.strip() and CARRY not in entry.body
        assert entry.entry_id == orion_day_journal_entry_id("2026-09-29")
        assert entry.trigger_kind == "orion_day_letter" and entry.source_kind == "orion_day"
        assert entry.mode == "daily"
    asyncio.run(scenario())


def test_checkpoint_holds_only_the_slim_brief():
    async def scenario():
        world, saver = World(), InMemorySaver()
        snap = await drive(world.graph(saver), world)
        brief = snap.values["brief"]
        assert "material" not in brief and "llm_view" not in brief
        assert brief["letter_date"] == "2026-09-29" and brief["timeout_sec"] == 900.0
    asyncio.run(scenario())


def test_restart_after_the_note_resumes_at_carry_forward_without_regenerating_the_note():
    async def scenario():
        world, saver = World(), InMemorySaver()
        world.replies[ORION_DAY_CARRY_FORWARD_VERB] = [Crash()]
        with pytest.raises(Crash):
            await drive(world.graph(saver), world)
        # A new driver process: fresh graph object on the same saver, recovery as AdmissionRuntime._recover
        # does for a WORK_NODES node (drop the lease, back to resource_request).
        graph = world.graph(saver)
        snap = await graph.aget_state(CFG)
        assert snap.next == ("write_carry_forward",) and snap.values["note_md"] == NOTE.strip()
        await graph.aupdate_state(CFG, {"lease": None, "turn_fence": 1, "status": "waiting_resource"},
                                  as_node="resource_request")
        world.granted = True
        snap = await drive(graph, world, first=False)
        assert snap.values["status"] == "completed"
        assert [c[0] for c in world.calls] == [ORION_DAY_NOTE_VERB, ORION_DAY_CARRY_FORWARD_VERB,
                                                ORION_DAY_CARRY_FORWARD_VERB]
        assert snap.values["llm_attempts"]["write_note"] == 1
    asyncio.run(scenario())


def test_restart_during_persist_is_idempotent():
    async def scenario():
        world, saver = World(), InMemorySaver()
        real = world.publish_journal

        async def crash_after_insert(entry):
            world.publish_journal = real
            raise Crash()

        world.publish_journal = crash_after_insert
        with pytest.raises(Crash):
            await drive(world.graph(saver), world)
        assert date(2026, 9, 29) in world.rows and not world.journals
        graph = world.graph(saver)  # new process picks up at persist; no GPU needed
        snap = await drive(graph, world, first=False)
        assert snap.values["status"] == "completed" and snap.values["persisted"] is True
        assert len(world.rows) == 1 and len(world.journals) == 1
        assert len(world.calls) == 2  # neither LLM call repeated
    asyncio.run(scenario())


def test_a_second_run_for_the_same_day_is_a_no_op():
    async def scenario():
        world, saver = World(), InMemorySaver()
        first_at = datetime(2026, 9, 30, 13, tzinfo=timezone.utc)
        world.rows[date(2026, 9, 29)] = {"run_id": "orion-day-2026-09-29-0", "letter_date": date(2026, 9, 29),
                                         "note_md": "The earlier run's note. " * 30, "created_at": first_at}
        snap = await drive(world.graph(saver), world)
        assert snap.values["status"] == "completed"
        assert snap.values["persisted"] is False and snap.values["persist_outcome"] == "already_written"
        assert snap.values["existing_run_id"] == "orion-day-2026-09-29-0"
        # It (re)publishes the STORED row's journal entry -- identical id, body and timestamp -- so a
        # first run that wrote the row but never got its journal out is healed, and nothing differs.
        (entry,) = world.journals
        assert entry.body == world.rows[date(2026, 9, 29)]["note_md"] and entry.created_at == first_at
        assert entry.entry_id == orion_day_journal_entry_id("2026-09-29")
    asyncio.run(scenario())


def test_persist_failure_backs_off_and_retries():
    async def scenario():
        world, saver = World(), InMemorySaver()
        world.persist_fail, world.journal_fail = 1, 1
        snap = await drive(world.graph(saver), world)
        assert snap.values["status"] == "completed" and snap.values["persisted"] is True
        assert snap.values["tail_attempts"] == {"persist": 1, "journal": 1}
        assert snap.values["journal_published"] is True
    asyncio.run(scenario())


def test_journal_trouble_never_fails_a_run_whose_letter_is_written():
    async def scenario():
        world, saver = World(max_attempts=2), InMemorySaver()
        world.journal_fail = 100
        snap = await drive(world.graph(saver), world)
        assert snap.values["status"] == "completed"
        assert snap.values["persisted"] is True and snap.values["journal_published"] is False
        assert world.rows and not world.journals
        assert finish_detail(snap.values)["journal_published"] is False
    asyncio.run(scenario())


def test_journal_entry_is_stamped_with_the_rows_created_at_on_every_replay():
    async def scenario():
        world, saver = World(), InMemorySaver()
        world.journal_fail = 1
        snap = await drive(world.graph(saver), world)
        row_at = world.rows[date(2026, 9, 29)]["created_at"]
        (entry,) = world.journals
        assert entry.created_at == row_at  # not the (later) replay's clock
        assert snap.values["status"] == "completed"
    asyncio.run(scenario())


def test_persist_gives_up_after_the_attempt_budget():
    async def scenario():
        world, saver = World(max_attempts=2), InMemorySaver()
        world.persist_fail = 5
        snap = await drive(world.graph(saver), world)
        assert snap.values["status"] == "failed" and "db down" in snap.values["last_error"]
    asyncio.run(scenario())


def test_a_lost_hold_replays_the_node_and_is_not_an_attempt():
    async def scenario():
        world, saver = World(max_attempts=1), InMemorySaver()
        world.execute_lost = [True, False, False]  # note loses its hold once
        snap = await drive(world.graph(saver), world)
        assert snap.values["status"] == "completed"
        assert snap.values["hold_takebacks"] == 1
        assert snap.values["llm_attempts"] == {"write_note": 1, "write_carry_forward": 1}
        assert world.registers == 2  # queued again for the hold
    asyncio.run(scenario())


def test_empty_or_short_completion_is_an_attempt_never_a_success():
    async def scenario():
        world, saver = World(max_attempts=3), InMemorySaver()
        world.replies[ORION_DAY_NOTE_VERB] = ["", "too short", NOTE]
        snap = await drive(world.graph(saver), world)
        assert snap.values["status"] == "completed"
        assert snap.values["llm_attempts"]["write_note"] == 3
        assert len(snap.values["note_md"]) >= MIN_NOTE_CHARS
    asyncio.run(scenario())


def test_a_failed_attempt_waits_out_a_backoff_before_the_next_one():
    async def scenario():
        world, saver = World(max_attempts=3), InMemorySaver()
        world.replies[ORION_DAY_NOTE_VERB] = [RuntimeError("cortex restarting")]
        graph = world.graph(saver)
        world.granted = True
        await graph.ainvoke(initial(), CFG)
        snap = await graph.aget_state(CFG)
        assert snap.next == ("retry_wait",)  # parked, not straight back into another call
        assert datetime.fromisoformat(snap.values["retry_at"]) == world.now + timedelta(seconds=30)
        assert len(world.calls) == 1
        # AdmissionRuntime._drive only resumes a parked retry_wait once now >= retry_at (the same
        # gate as curiosity's retry_wait); drive() advances the clock to retry_at the same way.
        snap = await drive(graph, world, first=False)
        assert snap.values["status"] == "completed" and snap.values["llm_attempts"]["write_note"] == 2
    asyncio.run(scenario())


def test_carry_forward_without_a_list_item_is_an_attempt():
    async def scenario():
        world, saver = World(max_attempts=3), InMemorySaver()
        world.replies[ORION_DAY_CARRY_FORWARD_VERB] = ["A paragraph with no list at all, just prose about threads.", CARRY]
        snap = await drive(world.graph(saver), world)
        assert snap.values["status"] == "completed"
        assert snap.values["llm_attempts"]["write_carry_forward"] == 2
        assert snap.values["carry_forward_refs"] == {"valid": 1, "unknown": 0}
    asyncio.run(scenario())


def test_a_release_error_after_the_carry_forward_keeps_the_text():
    async def scenario():
        world, saver = World(), InMemorySaver()
        real = world.release

        async def flaky_release(state, reason, keep_requeued=False):
            if reason == "completed" and state.get("carry_forward_md") and not state.get("letter_written"):
                world.release = real
                raise RuntimeError("store down")
            return await real(state, reason, keep_requeued)

        world.release = flaky_release
        snap = await drive(world.graph(saver), world)
        assert snap.values["status"] == "completed"
        assert [c[0] for c in world.calls].count(ORION_DAY_CARRY_FORWARD_VERB) == 1
    asyncio.run(scenario())


def test_rpc_timeout_sits_under_the_node_budget():
    async def scenario():
        world, saver = World(), InMemorySaver()
        await drive(world.graph(saver), world)
        assert all(extra["timeout"] == 900.0 - 30.0 for _, _, extra in world.calls)
    asyncio.run(scenario())


def test_repeated_transport_failure_fails_the_run_without_a_letter():
    async def scenario():
        world, saver = World(max_attempts=2), InMemorySaver()
        world.replies[ORION_DAY_NOTE_VERB] = [RuntimeError("rpc timeout"), RuntimeError("rpc timeout")]
        snap = await drive(world.graph(saver), world)
        assert snap.values["status"] == "failed" and "rpc timeout" in snap.values["last_error"]
        assert world.rows == {} and world.journals == []
        assert "carry_forward_md" not in snap.values or not snap.values["carry_forward_md"]
    asyncio.run(scenario())
