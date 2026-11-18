"""orion_day.letter admitted graph (living Prime #3224 Part 3): wait for the pool's hold (never an
attempt), write the letter as ONE unified turn (hold released right after its checkpoint), persist
once, journal the NOTE only, hand Prime the threads.

Resume per node, idempotent persist, and letter/threads separation are the contract.
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
from app.harness_step import harness_step_correlation_id
from app.orion_day_graph import (
    LETTER_STEP_KEY, MIN_NOTE_CHARS, OrionDayDeps, build_orion_day_graph, finish_detail, slim_brief,
)
from orion.orion_day.letter_turn import MAX_THREADS, NO_THREADS_MD, THREADS_HEADING
from orion.schemas.durable_run import CuriosityTurnRequestV1, CuriosityTurnResultV1
from orion.schemas.orion_day import (
    ORION_DAY_LETTER_SESSION_ID, ORION_DAY_LETTER_SOURCE_TAG, MissedLetterV1, OrionDayLetterSourcesV1,
    OrionDayLlmViewV1, OrionDayMaterialV1, OrionDayRunBriefV1, orion_day_journal_entry_id,
)

REF = {"lease_id": "hold-od", "generation": 1, "role": "agent", "holder": "durable-runs:orion-day-2026-09-29-1"}
RUN = "orion-day-2026-09-29-1"
CFG = {"configurable": {"thread_id": RUN}}
NOTE = ("Yesterday I spent most of my attention on the hop written_at prior. " * 20).strip()
THREAD = "[curiosity:ab61e4ccd47b] Test whether written_at is ever populated on hop nodes."
CARRY = f"- {THREAD}"
LETTER = f"{NOTE}\n\n{THREADS_HEADING}\n\n{CARRY}\n"
DIGEST = "# My day: 2026-09-29\n\n### [curiosity:ab61e4ccd47b] Curiosity\nfound x"
PRIME_ACT = "act:prime:make:2026-09-29"


class Crash(BaseException):
    """A process death mid-node: escapes every except Exception, leaves the checkpoint behind."""


def make_brief(**extra) -> OrionDayRunBriefV1:
    start = datetime(2026, 9, 29, 6, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    material = OrionDayMaterialV1(letter_date=date(2026, 9, 29), window_start=start, window_end=end,
                                  gathered_at=end + timedelta(hours=8))
    view = OrionDayLlmViewV1(digest_md=DIGEST, approx_tokens=20, budget_tokens=70000, chars_per_token=3.5,
                             included_refs=["curiosity:ab61e4ccd47b"], source_chars={"curiosity": 40},
                             source_shares={"curiosity": 1.0})
    return OrionDayRunBriefV1(letter_date=date(2026, 9, 29), window_start=start, window_end=end,
                              material=material, llm_view=view, timeout_sec=900.0, prime_act_id=PRIME_ACT,
                              missed_letters=[MissedLetterV1(letter_date=date(2026, 9, 28),
                                                             reason="the make bid was never funded")], **extra)


class World:
    """One hold: ``granted`` toggles it. ``replies`` is consumed one per turn: a str is the turn's text,
    an Exception (or BaseException) is raised from the turn RPC, ``("fail", error)`` is a refusal."""

    def __init__(self, *, max_attempts=3):
        self.now = datetime(2026, 9, 30, 14, tzinfo=timezone.utc)
        self.granted = False
        self.brief = make_brief()
        self.turns: list[CuriosityTurnRequestV1] = []
        self.replies: list = []
        self.releases: list[str] = []
        self.registers = 0
        self.execute_lost: list[bool] = []
        self.rows: dict[date, dict] = {}
        self.persist_fail = 0
        self.journals = []
        self.journal_fail = 0
        self.max_attempts = max_attempts

    # --- deps -------------------------------------------------------------------------------
    async def run_turn(self, req: CuriosityTurnRequestV1) -> CuriosityTurnResultV1:
        self.turns.append(req)
        reply = self.replies.pop(0) if self.replies else LETTER
        if isinstance(reply, BaseException):
            raise reply
        if isinstance(reply, tuple):
            return CuriosityTurnResultV1(run_id=req.run_id, correlation_id=req.correlation_id, ok=False, error=reply[1])
        return CuriosityTurnResultV1(run_id=req.run_id, correlation_id=req.correlation_id, text=reply,
                                     ok=bool(reply), debug={"harness_step_count": 3, "elapsed_sec": 400.0})

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
            OrionDayDeps(run_turn=self.run_turn, persist_letter=self.persist_letter,
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


def test_waits_without_spending_attempts_then_writes_one_letter_turn_and_persists_it():
    async def scenario():
        world, saver = World(), InMemorySaver()
        snap = await drive(world.graph(saver), world, grant_after=3)
        values = snap.values
        assert values["status"] == "completed"
        (turn,) = world.turns   # ONE unified turn; waiting never counted
        assert values["steps"][LETTER_STEP_KEY]["attempts"] == 1
        assert values["note_md"] == NOTE and values["carry_forward_md"] == CARRY and values["threads"] == [THREAD]
        # Hold handed back as soon as the letter was checkpointed, before persist.
        assert world.releases[0] == "completed" and world.hold_at_persist is False
        row = world.rows[date(2026, 9, 29)]
        assert row["run_id"] == RUN and row["note_md"] == NOTE and row["carry_forward_md"] == CARRY
        assert row["journal_entry_id"] == orion_day_journal_entry_id("2026-09-29")
        assert row["carry_forward_expires_at"] is None   # the curiosity offer is retired
        sources = OrionDayLetterSourcesV1.model_validate(row["sources"])
        assert sources.turn_correlation_id == turn.correlation_id
        assert sources.prime_act_id == PRIME_ACT
        assert [m.letter_date for m in sources.missed_letters] == [date(2026, 9, 28)]
        assert sources.input_shares == {"curiosity": 1.0}
        detail = finish_detail(values)
        assert detail["persisted"] is True and detail["persist_outcome"] == "written"
        assert detail["turn_correlation_id"] == turn.correlation_id and detail["prime_act_id"] == PRIME_ACT
        assert detail["threads"] == [THREAD] and detail["harness_step_count"] == 3
        assert "note_md" not in detail and "carry_forward_md" not in detail  # bounded: no letter text
    asyncio.run(scenario())


def test_the_turn_request_is_the_letter_turn():
    async def scenario():
        world, saver = World(), InMemorySaver()
        await drive(world.graph(saver), world)
        (turn,) = world.turns
        assert turn.session_id == ORION_DAY_LETTER_SESSION_ID and turn.source_tag == ORION_DAY_LETTER_SOURCE_TAG
        assert turn.require_lookup is False
        assert turn.retrieval_query == "my day 2026-09-29"
        assert DIGEST in turn.prompt and THREADS_HEADING in turn.prompt
        assert "2026-09-28: the make bid was never funded" in turn.prompt   # the missed day, told once
        assert turn.timeout_sec == 900.0 - 30.0   # the turn limit sits under the node timer
        assert turn.gpu_lease is not None and turn.gpu_lease.lease_id == "hold-od"   # attached to the run's hold
        assert turn.run_id == RUN and turn.attempt == 1
        # The turn's id is the one the letter row carries: the letter is findable in cockpit_turn_sighting.
        (row,) = world.rows.values()
        assert row["sources"]["turn_correlation_id"] == turn.correlation_id
    asyncio.run(scenario())


def test_journal_carries_the_letter_only():
    async def scenario():
        world, saver = World(), InMemorySaver()
        await drive(world.graph(saver), world)
        (entry,) = world.journals
        assert entry.body == NOTE and THREAD not in entry.body and THREADS_HEADING not in entry.body
        assert entry.entry_id == orion_day_journal_entry_id("2026-09-29")
        assert entry.trigger_kind == "orion_day_letter" and entry.source_kind == "orion_day"
        assert entry.mode == "daily"
    asyncio.run(scenario())


def test_checkpoint_holds_only_the_slim_brief_and_the_full_letter_step():
    async def scenario():
        world, saver = World(), InMemorySaver()
        snap = await drive(world.graph(saver), world)
        brief = snap.values["brief"]
        assert "material" not in brief and "llm_view" not in brief
        assert brief["letter_date"] == "2026-09-29" and brief["timeout_sec"] == 900.0
        assert brief["prime_act_id"] == PRIME_ACT
        step = snap.values["steps"][LETTER_STEP_KEY]
        assert step["text"] == LETTER and step["turn_correlation_id"] == world.turns[0].correlation_id
        assert snap.values["harness_turn_meta"]["turn_correlation_id"] == step["turn_correlation_id"]
    asyncio.run(scenario())


def test_a_letter_with_no_threads_keeps_its_letter_and_hands_prime_none():
    async def scenario():
        world, saver = World(), InMemorySaver()
        world.replies = [NOTE]
        snap = await drive(world.graph(saver), world)
        assert snap.values["status"] == "completed" and len(world.turns) == 1
        assert snap.values["carry_forward_md"] == NO_THREADS_MD and snap.values["threads"] == []
        assert finish_detail(snap.values)["threads"] == []
    asyncio.run(scenario())


def test_finish_detail_hands_prime_at_most_max_threads():
    async def scenario():
        world, saver = World(), InMemorySaver()
        bullets = "\n".join(f"- thread number {i} still open" for i in range(MAX_THREADS + 3))
        world.replies = [f"{NOTE}\n\n{THREADS_HEADING}\n{bullets}\n"]
        snap = await drive(world.graph(saver), world)
        assert len(snap.values["threads"]) == MAX_THREADS + 3   # the stored list keeps every one
        detail = finish_detail(snap.values)
        assert detail["threads"] == [f"thread number {i} still open" for i in range(MAX_THREADS)]
        assert detail["threads_total"] == MAX_THREADS + 3
    asyncio.run(scenario())


def test_restart_after_the_letter_checkpoint_never_reruns_the_turn():
    async def scenario():
        world, saver = World(), InMemorySaver()

        async def crash_on_persist(row):
            raise Crash()

        real = world.persist_letter
        world.persist_letter = crash_on_persist
        with pytest.raises(Crash):
            await drive(world.graph(saver), world)
        world.persist_letter = real
        graph = world.graph(saver)   # a new process
        snap = await graph.aget_state(CFG)
        assert snap.next == ("persist",) and snap.values["note_md"] == NOTE
        snap = await drive(graph, world, first=False)
        assert snap.values["status"] == "completed" and snap.values["persisted"] is True
        assert len(world.turns) == 1   # the letter was never regenerated
    asyncio.run(scenario())


def test_a_crash_inside_the_turn_replays_it_after_recovery():
    async def scenario():
        world, saver = World(), InMemorySaver()
        world.replies = [Crash()]
        with pytest.raises(Crash):
            await drive(world.graph(saver), world)
        graph = world.graph(saver)
        snap = await graph.aget_state(CFG)
        assert snap.next == ("write_letter",) and not snap.values.get("note_md")
        # Recovery as AdmissionRuntime._recover does for a WORK_NODES node: fence, back to resource_request.
        await graph.aupdate_state(CFG, {"lease": None, "turn_fence": 1, "status": "waiting_resource"},
                                  as_node="resource_request")
        world.granted = True
        snap = await drive(graph, world, first=False)
        assert snap.values["status"] == "completed" and len(world.turns) == 2
        assert world.turns[0].correlation_id != world.turns[1].correlation_id   # fresh turn id after the fence
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
        assert len(world.turns) == 1  # the turn was not repeated
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
        assert snap.values["journal_published"] is True and len(world.turns) == 1
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


def test_a_hold_take_back_replays_the_turn_without_spending_an_attempt():
    async def scenario():
        world, saver = World(max_attempts=1), InMemorySaver()
        world.execute_lost = [True, False]  # the letter loses its hold once
        snap = await drive(world.graph(saver), world)
        assert snap.values["status"] == "completed"
        assert snap.values["hold_takebacks"] == 1
        assert snap.values["steps"][LETTER_STEP_KEY]["attempts"] == 1
        assert world.registers == 2  # queued again for the hold
    asyncio.run(scenario())


def test_a_too_short_letter_is_a_retried_attempt_never_a_success():
    async def scenario():
        world, saver = World(max_attempts=3), InMemorySaver()
        world.replies = [f"Too short.\n\n{THREADS_HEADING}\n{CARRY}", LETTER]
        snap = await drive(world.graph(saver), world)
        assert snap.values["status"] == "completed" and len(world.turns) == 2
        assert snap.values["steps"][LETTER_STEP_KEY]["attempts"] == 2
        assert len(snap.values["note_md"]) >= MIN_NOTE_CHARS
        assert world.turns[0].correlation_id != world.turns[1].correlation_id   # a fresh turn, not a cached one
        assert world.turns[1].attempt == 2
    asyncio.run(scenario())


def test_a_failed_attempt_waits_out_a_backoff_before_the_next_one():
    async def scenario():
        world, saver = World(max_attempts=3), InMemorySaver()
        world.replies = [("fail", "harness_timeout")]
        graph = world.graph(saver)
        world.granted = True
        await graph.ainvoke(initial(), CFG)
        snap = await graph.aget_state(CFG)
        assert snap.next == ("retry_wait",)  # parked, not straight back into another turn
        assert datetime.fromisoformat(snap.values["retry_at"]) == world.now + timedelta(seconds=30)
        assert len(world.turns) == 1 and "harness_timeout" in snap.values["last_error"]
        snap = await drive(graph, world, first=False)
        assert snap.values["status"] == "completed" and snap.values["steps"][LETTER_STEP_KEY]["attempts"] == 2
    asyncio.run(scenario())


def test_refusals_and_empty_replies_fail_the_run_past_the_budget_with_no_letter():
    async def scenario():
        world, saver = World(max_attempts=3), InMemorySaver()
        world.replies = [("fail", "refused"), "", RuntimeError("rpc timeout")]
        snap = await drive(world.graph(saver), world)
        assert snap.values["status"] == "failed" and "rpc timeout" in snap.values["last_error"]
        assert len(world.turns) == 3 and snap.values["attempt"] == 3
        assert world.rows == {} and world.journals == []
        assert not snap.values.get("note_md") and LETTER_STEP_KEY not in (snap.values.get("steps") or {})
        # The failed turn's identity stays on the state for the failed-run detail and cancel.
        assert snap.values["harness_turn_meta"]["turn_correlation_id"] == world.turns[-1].correlation_id
    asyncio.run(scenario())


def test_a_release_error_after_the_letter_keeps_the_letter():
    async def scenario():
        world, saver = World(), InMemorySaver()
        real = world.release

        async def flaky_release(state, reason, keep_requeued=False):
            if reason == "completed" and state.get("note_md") and not state.get("letter_written"):
                world.release = real
                raise RuntimeError("store down")
            return await real(state, reason, keep_requeued)

        world.release = flaky_release
        snap = await drive(world.graph(saver), world)
        assert snap.values["status"] == "completed" and len(world.turns) == 1
    asyncio.run(scenario())
