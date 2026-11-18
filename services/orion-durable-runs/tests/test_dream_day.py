"""The day dream Patch 2 (dream.day): the answer parse, the ref check, the 5-memory cap, idempotent ids, the Denver
windows, the grading, the night tick, the graph under real LangGraph checkpoints, and the act's grade.

What must hold: an item with any unresolved ref is never written (dropped and counted); at most 5 memories and 1
new concern; memory ids depend on (day, index) only; the run reads the day only through `dig dream-day` (no
receipt = a failed attempt) and needs an answer block (none = a failed attempt); a night with no memory grades
0 with "no lasting trace"; one run a night, only inside 01:00-03:00 Denver, for the day that just closed.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.dream_day_graph import (
    DreamDayDeps, build_dream_day_graph, build_prompt, check_receipt, finish_detail,
)
from app.dream_day_poll import DreamDayTicker, dreamt_day, next_run_id
from app.dream_day_store import concerns_from_state
from app.harness_step import HarnessStepFailed
from app.run_acts import closed_event, run_act_family
from orion.memory.dream_day import (
    NO_TRACE_NOTE, ParsedAnswer, answer_refs, build_memories, canonical_ref, check_answer, day_window,
    dream_memory_id, experience_cites, grade, night_window, parse_answer,
)
from orion.schemas.dream_day import (
    DREAM_DAY_STEP_KEY, DREAM_DAY_WORKFLOW, DreamDayAnswerV1, DreamDayBriefV1, DreamDayConcernV1,
    DreamDayMaterialV1, DreamDayNowItemV1, DreamDayReadV1,
)
from orion.schemas.durable_run import CuriosityTurnRequestV1, CuriosityTurnResultV1, DurableRunRequestV1
from orion.schemas.registry import resolve

DAY = date(2026, 11, 17)
RUN_ID = "dream-day-2026-11-17"
# 2026-11-18 01:30 Denver (MST, UTC-7): inside the night window for 2026-11-17.
T0 = datetime(2026, 11, 18, 8, 30, tzinfo=timezone.utc)
EXP = "experience:reverie_chain:11300caf-50b1-4e01-8117-7e9280302ead"
ACT = "act:act_9c473ebd88447b8b6efbe3e6"
CYCLE = "dream_cycle:dc-145755756e5d"
RESOLVED = {EXP: "reverie_chain: reverie chain: open-loop-e91fa7e1c61e",
            ACT: "hub.world_pulse_read look reading:x: stage1_completed: a paper about spiking networks",
            CYCLE: "sleep pressure completed, 0 links: pairs dream=3 control=1"}


def answer_json(memories=None, plan=None, day=None) -> str:
    import json
    body = {"memories": memories if memories is not None else [
        {"statement": "I read a paper about spiking networks and it pulled at my reverie that afternoon.",
         "purpose": "happened", "refs": [ACT, EXP]}],
        "plan": plan or [], "day": day or {"text": "A quiet day of reading and one sleep that linked nothing.",
                                           "refs": [CYCLE]}}
    return "I read the day.\n```json\n" + json.dumps(body) + "\n```"


# --------------------------------------------------------------------------- parse


def test_parse_takes_the_last_json_block_and_counts_malformed_items():
    text = ("```json\n{\"memories\": []}\n```\nthen\n```json\n"
            '{"memories": [{"statement": "ok statement here", "refs": ["act:act_1"]}, {"refs": 3}],'
            ' "plan": [{"action": "explode"}], "day": "A day."}\n```')
    parsed = parse_answer(text)
    assert parsed is not None and len(parsed.answer.memories) == 1 and parsed.malformed == 2
    assert parsed.answer.day.text == "A day."


def test_parse_reads_a_bare_object_and_refuses_no_answer():
    assert parse_answer('Done. {"memories": [], "plan": [], "day": {"text": "x", "refs": []}}') is not None
    assert parse_answer("I slept on it and have nothing to say.") is None
    assert parse_answer('```json\n{"unrelated": 1}\n```') is None


def test_canonical_refs_fold_chat_and_refuse_other_kinds():
    assert canonical_ref("[chat:abc]") == "chat_turn:abc"
    assert canonical_ref(EXP) == EXP
    assert canonical_ref("journal:123") is None and canonical_ref("not a ref") is None


# --------------------------------------------------------------------------- the ref check


def test_unresolved_or_unsupported_ref_drops_the_whole_item():
    parsed = parse_answer(answer_json(memories=[
        {"statement": "A memory with one good and one missing record behind it.", "refs": [ACT, "act:act_missing"]},
        {"statement": "A memory citing a kind the dream may not cite at all.", "refs": ["journal:9"]},
        {"statement": "A memory with no record behind it at all, said anyway.", "refs": []},
        {"statement": "A memory whose records all resolve, so it is kept.", "refs": [ACT]},
    ]))
    check = check_answer(parsed, RESOLVED, [])
    assert [m.refs for m in check.memories] == [[ACT]]
    assert check.items_dropped == 3 and check.refs_dropped == 2
    assert "memory_without_ref" in check.reasons and "memory_unresolved_ref" in check.reasons


def test_five_memory_cap_and_one_new_concern():
    mems = [{"statement": f"Memory number {i} of a busy day, each with its record.", "refs": [ACT]} for i in range(7)]
    plan = [{"action": "new", "gist": "look at the sleep", "reason": "it linked nothing", "refs": [CYCLE]},
            {"action": "new", "gist": "a second new one", "reason": "greedy", "refs": [CYCLE]},
            {"action": "keep", "concern_id": "c1", "reason": "still open"},
            {"action": "drop", "concern_id": "c2", "reason": "done", "refs": [EXP]},
            {"action": "drop", "concern_id": "unknown", "reason": "not held"},
            {"action": "drop", "concern_id": "c1", "reason": "repeated"},
            {"action": "keep", "concern_id": "c2", "reason": ""}]
    check = check_answer(parse_answer(answer_json(memories=mems, plan=plan)), RESOLVED, ["c1", "c2"])
    assert len(check.memories) == 5 and check.memories_over_cap == 2
    assert (check.kept, check.dropped, check.new) == (1, 1, 1) and check.plan_rejected == 4
    assert check.plan[0] == {"action": "new", "gist": "look at the sleep", "reason": "it linked nothing",
                             "refs": [CYCLE]}


def test_day_paragraph_is_cut_to_1200_and_dropped_on_a_bad_ref():
    long = " ".join(["word"] * 400)
    check = check_answer(parse_answer(answer_json(day={"text": long, "refs": [CYCLE]})), RESOLVED, [])
    assert check.day_cut and len(check.day_text) <= 1200 and check.day_refs == [CYCLE]
    check = check_answer(parse_answer(answer_json(day={"text": "x", "refs": ["act:act_gone"]})), RESOLVED, [])
    assert check.day_text is None and check.items_dropped == 1


# --------------------------------------------------------------------------- memories


def test_memories_go_through_the_validator_with_deterministic_ids():
    parsed = parse_answer(answer_json(memories=[
        {"statement": "I read a paper about spiking networks that afternoon.", "refs": [ACT]},
        {"statement": "Too short.", "refs": [ACT]},
        {"statement": "My one sleep of the day linked none of its pairs.", "purpose": "follow_up", "refs": [CYCLE]},
    ]))
    check = check_answer(parsed, RESOLVED, [])
    result = build_memories(check, RESOLVED, DAY)
    assert [r.reason for r in result.rejections] == ["statement_too_short"]
    assert [m.memory_id for m in result.memories] == [dream_memory_id(DAY, 0), dream_memory_id(DAY, 1)]
    assert dream_memory_id(DAY, 0) == dream_memory_id(DAY, 0) != dream_memory_id(DAY + timedelta(days=1), 0)
    m = result.memories[0]
    assert (m.channel, m.voice, m.purpose, m.confirmation_state) == ("dream", "orion_thought", "happened", "auto")
    assert [(e.source_kind, e.source_id, e.verified) for e in m.evidence] == [("dream_ref", ACT, True)]
    # follow_up is not a dream purpose: it becomes "happened"
    assert result.memories[1].purpose == "happened"
    again = build_memories(check_answer(parse_answer(answer_json(memories=[
        {"statement": "I read a paper about spiking networks that afternoon.", "refs": [ACT]}])), RESOLVED, []),
        RESOLVED, DAY)
    assert again.memories[0].memory_id == result.memories[0].memory_id


def test_experience_cites_come_only_from_kept_items():
    check = check_answer(parse_answer(answer_json(plan=[
        {"action": "new", "gist": "g", "reason": "r", "refs": [EXP]}])), RESOLVED, [])
    assert experience_cites(check, [ACT, EXP]) == [EXP]
    assert answer_refs(parse_answer(answer_json()).answer) == [ACT, EXP, CYCLE]


def test_grading():
    assert grade(0) == (False, NO_TRACE_NOTE)
    assert grade(1) == (True, "1 memory written") and grade(3)[0] is True


# --------------------------------------------------------------------------- windows


def test_denver_windows_are_dst_correct():
    assert night_window(date(2026, 11, 18), 1, 3) == (datetime(2026, 11, 18, 8, tzinfo=timezone.utc),
                                                      datetime(2026, 11, 18, 10, tzinfo=timezone.utc))
    assert night_window(date(2026, 7, 1), 1, 3)[0] == datetime(2026, 7, 1, 7, tzinfo=timezone.utc)   # MDT
    lo, hi = day_window(date(2026, 11, 1))     # fall back: a 25-hour day
    assert (hi - lo) == timedelta(hours=25)
    assert dreamt_day(T0) == DAY
    assert dreamt_day(datetime(2026, 11, 18, 6, 59, tzinfo=timezone.utc)) == date(2026, 11, 16)   # 23:59 Denver


# --------------------------------------------------------------------------- the night tick


class TickPool:
    def __init__(self, runs=None, done=None, fail=False):
        self.runs, self.done, self.fail = list(runs or []), done, fail

    @asynccontextmanager
    async def connection(self):
        pool = self

        class Cur:
            def __init__(self, rows):
                self._rows = rows

            async def fetchall(self):
                return self._rows

            async def fetchone(self):
                return self._rows[0] if self._rows else None

        class Conn:
            async def execute(self, sql, params):
                if pool.fail:
                    raise RuntimeError('relation "dream_day" does not exist')
                if "FROM dream_day" in sql:
                    return Cur([{"run_id": pool.done}] if pool.done else [])
                prefix = params[0].rstrip("%")
                if "count(*)" in sql:
                    return Cur([{"n": sum(1 for r in pool.runs if r["terminal"] is None and r["run_id"].startswith(prefix))}])
                return Cur([r for r in pool.runs if r["run_id"].startswith(prefix)])

        yield Conn()


def tick_settings(**kw):
    return SimpleNamespace(**{**dict(dream_day_start_hour_local=1, dream_day_end_hour_local=3,
                                     dream_day_route="memory_distill", dream_day_turn_timeout_sec=900.0), **kw})


def ticker(pool, now=T0):
    sent = []

    async def submit(req):
        sent.append(req)
    return DreamDayTicker(settings=tick_settings(), pool=pool, submit=submit, now=lambda: now), sent


def test_tick_submits_one_run_for_the_closed_day_inside_the_window():
    t, sent = ticker(TickPool())
    result = asyncio.run(t.tick_once())
    assert result.decision == "submitted" and result.run_id == RUN_ID and result.day == "2026-11-17"
    req = sent[0]
    assert req.workflow == DREAM_DAY_WORKFLOW and req.brief.day == DAY and req.brief.llm_route == "memory_distill"
    assert req.admission.resource == "llm.route.memory_distill" and req.admission.preferred_lane == "memory_distill"
    assert req.admission.deadline_at == datetime(2026, 11, 18, 10, tzinfo=timezone.utc)
    assert DurableRunRequestV1.model_validate(req.model_dump(mode="json")).brief == req.brief


def test_tick_outside_window_complete_in_flight_and_missing_table():
    assert asyncio.run(ticker(TickPool(), now=T0 - timedelta(hours=2))[0].tick_once()).decision == "outside_window"
    assert asyncio.run(ticker(TickPool(done=RUN_ID))[0].tick_once()).decision == "complete"
    busy = TickPool(runs=[{"run_id": "dream-day-2026-11-16", "terminal": None, "updated_at": T0}])
    assert asyncio.run(ticker(busy)[0].tick_once()).decision == "waiting_in_flight"
    r = asyncio.run(ticker(TickPool(fail=True))[0].tick_once())
    assert r.decision == "state_unreadable" and "dream_day" in r.error
    d, sent = ticker(TickPool())
    assert asyncio.run(d.tick_once(dry_run=True)).decision == "would_submit" and sent == []


def test_retry_rule_and_hand_trigger():
    failed = [{"run_id": RUN_ID, "terminal": "failed", "updated_at": T0 - timedelta(hours=1)}]
    assert next_run_id(DAY, failed, now=T0, deadline=T0 + timedelta(hours=1), min_left_sec=900) == RUN_ID + "-a2"
    assert next_run_id(DAY, failed, now=T0, deadline=T0 + timedelta(minutes=10), min_left_sec=900) is None
    done = [{"run_id": RUN_ID, "terminal": "completed", "updated_at": T0}]
    assert next_run_id(DAY, done, now=T0, deadline=T0 + timedelta(hours=1), min_left_sec=900) is None
    # The hand trigger: outside the window, any closed day of the last week; never today, never older.
    afternoon = datetime(2026, 11, 18, 21, tzinfo=timezone.utc)
    t, sent = ticker(TickPool(), now=afternoon)
    r = asyncio.run(t.submit_for(DAY))
    assert r.decision == "submitted" and sent[0].brief.operator_forced
    assert sent[0].admission.deadline_at == afternoon + timedelta(hours=3)
    assert asyncio.run(t.submit_for(date(2026, 11, 18))).decision == "day_out_of_range"
    assert asyncio.run(t.submit_for(date(2026, 11, 1))).decision == "day_out_of_range"


# --------------------------------------------------------------------------- contracts


def test_contracts_resolve_and_concerns_are_read_only():
    for name in ("DreamDayBriefV1", "DreamDayAnswerV1", "DreamDayV1"):
        assert resolve(name).__name__ == name
    assert concerns_from_state(None, T0) == ([], "absent")
    assert concerns_from_state("{not json", T0) == ([], "unreadable")
    raw = ('{"schema_version": "prime.state.v1", "concerns": [{"concern_id": "c1", "text": "the letter thread",'
           ' "opened_at": "2026-11-16T08:30:00Z", "last_evidence_at": "2026-11-18T02:30:00Z"}]}')
    concerns, status = concerns_from_state(raw, T0)
    assert status == "ok" and concerns[0].concern_id == "c1" and concerns[0].age_h == pytest.approx(48.0)
    assert concerns[0].last_evidence_h == pytest.approx(6.0)


def test_receipt_must_be_this_runs_read():
    read = DreamDayReadV1(day=DAY, read_at=T0, roundup_lines=10, roundup_refs=3, digs_total=5, digs_chat=2,
                          digs_background=3)
    assert check_receipt(read, T0 - timedelta(minutes=5)) is read
    with pytest.raises(HarnessStepFailed, match="no read receipt"):
        check_receipt(None, T0)
    with pytest.raises(HarnessStepFailed, match="older than this attempt"):
        check_receipt(read, T0 + timedelta(hours=1))   # an earlier attempt's (or a hand run's) receipt


def test_prompt_shows_every_part_with_refs_and_says_unread():
    m = material(now_status="unavailable", concerns_status="unreadable")
    prompt = build_prompt(DreamDayBriefV1(dream_day="dream.day", day=DAY, timeout_sec=900, llm_route="memory_distill"), m)
    assert "This is your day, 2026-11-17. Read it." in prompt
    assert "python -P -m orion.dig dream-day 2026-11-17" in prompt
    assert "UNREAD: the now could not be read" in prompt and "UNREAD: Prime's state is unreadable" in prompt
    ok = build_prompt(DreamDayBriefV1(dream_day="dream.day", day=DAY, timeout_sec=900, llm_route="m"), material())
    assert "concern_id c1" in ok and f"[{EXP}]" in ok


# --------------------------------------------------------------------------- the graph


def material(**kw) -> DreamDayMaterialV1:
    lo, hi = day_window(DAY)
    base = dict(day=DAY, window_start=lo, window_end=hi, gathered_at=T0, now_shown_id="shown-1",
                concerns=[DreamDayConcernV1(concern_id="c1", text="the letter thread", age_h=40, last_evidence_h=5)],
                now_items=[DreamDayNowItemV1(event_id=EXP.removeprefix("experience:"), organ="reverie", label="x")])
    return DreamDayMaterialV1(**{**base, **kw})


class World:
    def __init__(self, replies=None, *, receipt=True, max_attempts=3, persist_fail=0, cites_fail=0, winner=None,
                 judge=None):
        from app.admitted_graph import GRANTED, WAITING
        self.GRANTED, self.WAITING = GRANTED, WAITING
        self.now = T0
        self.granted = False
        self.replies = list(replies or [])
        self.receipt = receipt
        self.turns: list[CuriosityTurnRequestV1] = []
        self.releases: list[str] = []
        self.gathers = 0
        self.stored: dict | None = None
        self.cites: list = []
        self.max_attempts = max_attempts
        self.persist_fail, self.cites_fail, self.winner = persist_fail, cites_fail, winner
        self.receipt_at = None   # fixed receipt time (an old read), else "now"
        self.judge = judge       # async (prompt, held, timeout) -> {"text", ...}; None = gate off
        self.judged: list[tuple[str, dict, float]] = []
        self.resolve_fail = 0    # the next N resolves raise (the gate's read comes first)

    async def turn(self, req):
        self.turns.append(req)
        reply = self.replies.pop(0) if self.replies else answer_json()
        return CuriosityTurnResultV1(run_id=req.run_id, correlation_id=req.correlation_id, text=reply, ok=True,
                                     debug={"harness_step_count": 6})

    async def gather(self, brief, run_id):
        self.gathers += 1
        return material()

    async def read_receipt(self, day):
        if not self.receipt:
            return None
        return DreamDayReadV1(day=day, read_at=self.receipt_at or self.now, roundup_lines=80, roundup_refs=30, digs_total=12,
                              digs_chat=4, digs_background=8)

    async def resolve(self, refs):
        if self.resolve_fail:
            self.resolve_fail -= 1
            raise RuntimeError("db away")
        return {r: RESOLVED[r] for r in refs if r in RESOLVED}

    async def referents(self):
        return []

    async def persist(self, fields, result, route):
        if self.persist_fail:
            self.persist_fail -= 1
            raise RuntimeError("db away")
        if self.winner:
            return {"outcome": "already_written", "run_id": self.winner, "memories_written": 2,
                    "memory_ids": ["m1", "m2"], "produced": True, "note": "2 memories written"}
        if self.stored is None:
            n = len(result.memories)
            produced, note = grade(n)
            self.stored = {"fields": fields, "result": result, "route": route, "outcome": "written",
                           "run_id": fields["run_id"], "memories_written": n,
                           "memory_ids": [m.memory_id for m in result.memories], "produced": produced, "note": note}
        return dict(self.stored)

    async def record_cites(self, shown_id, cites, ref):
        self.cites.append((shown_id, cites, ref))
        if self.cites_fail:
            self.cites_fail -= 1
            raise RuntimeError("use endpoint away")
        return {"uses": [{"event_id": c} for c in cites], "rejected_not_shown": [], "rejected_over_cap": []}

    async def _judge(self, prompt, held, timeout):
        self.judged.append((prompt, held, timeout))
        return await self.judge(prompt)

    async def register(self, state):
        return {"status": "waiting_resource", "hold": {"request_id": f"{RUN_ID}:1", "lease_id": "h"}, "hold_seq": 1}

    async def lease(self, state):
        ref = {"lease_id": "h", "generation": 1, "role": "agent", "holder": f"durable-runs:{RUN_ID}"}
        if self.granted:
            return self.GRANTED, {"status": "admitted", "lease": ref, "hold": state.get("hold")}
        return self.WAITING, {"status": "waiting_resource", "lease": None}

    async def execute(self, state, node):
        if not self.granted:
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
        from app.admitted_graph import AdmissionDeps
        deps = DreamDayDeps(run_turn=self.turn, gather=self.gather, read_receipt=self.read_receipt,
                            resolve_refs=self.resolve, known_referents=self.referents, persist=self.persist,
                            record_cites=self.record_cites, judge=self._judge if self.judge else None)
        return build_dream_day_graph(deps, AdmissionDeps(self.register, self.lease, self.execute, self.release,
                                                         self.event, now=lambda: self.now,
                                                         max_attempts=self.max_attempts, retry_base_seconds=10.0),
                                     saver)


CFG = {"configurable": {"thread_id": RUN_ID}}


def initial():
    return {"run_id": RUN_ID, "correlation_id": "trace-dd", "workflow": DREAM_DAY_WORKFLOW, "attempt": 0,
            "requested_at": (T0 - timedelta(minutes=1)).isoformat(),
            "admission": {"resource": "llm.route.memory_distill"},
            "brief": {"dream_day": "dream.day", "day": DAY.isoformat(), "timeout_sec": 900,
                      "llm_route": "memory_distill"}}


async def start_and_grant(world, saver):
    pytest.importorskip("langgraph")
    from langgraph.types import Command
    await asyncio.wait_for(world.graph(saver).ainvoke(initial(), CFG), 2)
    snap = await world.graph(saver).aget_state(CFG)
    assert snap.next == ("resource_wait",) and world.turns == [] and world.gathers == 1
    world.granted = True
    return await asyncio.wait_for(world.graph(saver).ainvoke(Command(resume={}), CFG), 5)


async def after_backoff(world, saver):
    from langgraph.types import Command
    snap = await world.graph(saver).aget_state(CFG)
    assert snap.next == ("retry_wait",)
    world.now = datetime.fromisoformat(snap.values["retry_at"]) + timedelta(seconds=1)
    world.granted = True
    return await asyncio.wait_for(world.graph(saver).ainvoke(Command(resume=True), CFG), 5)


def test_a_night_reads_the_day_writes_memories_and_records_cites():
    pytest.importorskip("langgraph")
    from langgraph.checkpoint.memory import InMemorySaver

    async def scenario():
        world, saver = World(), InMemorySaver()
        result = await start_and_grant(world, saver)
        assert result["status"] == "completed" and result["produced"] is True
        assert len(world.turns) == 1 and world.turns[0].require_lookup is False
        assert world.turns[0].source_tag == "dream_day" and world.turns[0].gpu_lease is not None
        assert world.releases == ["completed"]        # released once the answer was checkpointed
        assert world.stored["route"] == "memory_distill"
        fields = world.stored["fields"]
        assert fields["day"] == DAY and fields["day_refs"] == [CYCLE] and fields["now_shown_id"] == "shown-1"
        assert world.cites == [("shown-1", [EXP], "dream_day:2026-11-17")]
        detail = finish_detail(result)
        assert detail["line"] == "dream_day" and detail["memories_written"] == 1 and detail["produced"]
        assert detail["read_unread"] == [] and detail["cites"]["used"] == 1
        assert detail["turn_correlation_id"] == result["steps"][DREAM_DAY_STEP_KEY]["turn_correlation_id"]
        # The act closes with the run's own grade.
        fam = run_act_family({"workflow": DREAM_DAY_WORKFLOW})
        assert fam.actor == "dream.cycle" and fam.graded_by_run
    asyncio.run(scenario())


def test_no_receipt_and_no_answer_are_failed_attempts_never_successes():
    pytest.importorskip("langgraph")
    from langgraph.checkpoint.memory import InMemorySaver

    async def scenario():
        world, saver = World(receipt=False), InMemorySaver()
        result = await start_and_grant(world, saver)
        assert result["status"] == "retrying" and "dream_day_not_read" in result["last_error"]
        assert world.stored is None
        world.receipt = True
        world.replies = ["I thought about it but wrote no block."] * 2
        result = await after_backoff(world, saver)
        assert result["status"] == "retrying" and "dream_day_no_answer" in result["last_error"]
        result = await after_backoff(world, saver)
        assert result["status"] == "failed" and world.stored is None    # budget spent: nothing written
    asyncio.run(scenario())


def test_a_night_with_no_memory_grades_no_lasting_trace():
    pytest.importorskip("langgraph")
    from langgraph.checkpoint.memory import InMemorySaver

    async def scenario():
        world, saver = World(replies=[answer_json(memories=[])]), InMemorySaver()
        result = await start_and_grant(world, saver)
        assert result["status"] == "completed" and result["produced"] is False
        assert result["produced_note"] == NO_TRACE_NOTE
        fam = run_act_family({"workflow": DREAM_DAY_WORKFLOW})
        ev = closed_event(fam, {**result, "workflow": DREAM_DAY_WORKFLOW}, status="completed", opened_at=T0,
                          grant_entry_id="g1", closed_at=T0 + timedelta(minutes=10))
        assert ev.act.actor == "dream.cycle" and ev.act.verb == "make"
        assert ev.act.outcome.value == 0.0 and ev.act.outcome.note == NO_TRACE_NOTE
        good = closed_event(fam, {"run_id": RUN_ID, "produced": True, "produced_note": "1 memory written"},
                            status="completed", opened_at=T0, grant_entry_id="g1", closed_at=T0)
        assert good.act.outcome.value == 1.0
        bad = closed_event(fam, {"run_id": RUN_ID, "produced": True, "last_error": "x"}, status="failed",
                           opened_at=T0, grant_entry_id="g1", closed_at=T0)
        assert bad.act.outcome.value == 0.0
    asyncio.run(scenario())


def test_a_retry_needs_its_own_fresh_read():
    pytest.importorskip("langgraph")
    from langgraph.checkpoint.memory import InMemorySaver

    async def scenario():
        world, saver = World(replies=["no block this time"]), InMemorySaver()
        result = await start_and_grant(world, saver)
        assert result["status"] == "retrying" and "dream_day_no_answer" in result["last_error"]
        # An earlier read (attempt 1, or a hand run): attempt 2 answers but never re-runs the dig.
        world.receipt_at = T0 - timedelta(minutes=10)
        result = await after_backoff(world, saver)
        assert result["status"] == "retrying" and "older than this attempt" in result["last_error"]
        assert world.stored is None
    asyncio.run(scenario())


def test_persist_retries_without_a_second_turn_and_cites_give_up_without_failing():
    pytest.importorskip("langgraph")
    from langgraph.checkpoint.memory import InMemorySaver
    from app.dream_day_graph import CITES_MAX_ATTEMPTS

    async def scenario():
        world, saver = World(persist_fail=1, cites_fail=CITES_MAX_ATTEMPTS), InMemorySaver()
        result = await start_and_grant(world, saver)
        assert result["status"] == "retrying" and result["retry_node"] == "persist"
        for _ in range(CITES_MAX_ATTEMPTS):
            from langgraph.types import Command
            snap = await world.graph(saver).aget_state(CFG)
            if snap.next != ("retry_wait",):
                break
            world.now = datetime.fromisoformat(snap.values["retry_at"]) + timedelta(seconds=1)
            result = await asyncio.wait_for(world.graph(saver).ainvoke(Command(resume=True), CFG), 5)
        assert result["status"] == "completed" and result["produced"] is True
        assert len(world.turns) == 1                         # the turn ran once
        assert len(world.cites) == CITES_MAX_ATTEMPTS and "error" in result["cites"]
    asyncio.run(scenario())


def test_a_run_that_lost_the_day_grades_zero():
    pytest.importorskip("langgraph")
    from langgraph.checkpoint.memory import InMemorySaver

    async def scenario():
        world, saver = World(winner="dream-day-2026-11-17-other"), InMemorySaver()
        result = await start_and_grant(world, saver)
        assert result["status"] == "completed" and result["persist_outcome"] == "already_written"
        assert result["produced"] is False and "already_written by" in result["produced_note"]
        assert world.cites == []                               # only the winner records cites
    asyncio.run(scenario())


# --- Patch 2b: what Prime reads (GET /dream-day/result) ----------------------------------------------------


class _ResultPool:
    def __init__(self, row):
        self.row, self.sql = row, []

    @asynccontextmanager
    async def connection(self):
        pool = self

        class _Cur:
            async def fetchone(self):
                return pool.row

        class _Conn:
            async def execute(self, sql, params):
                pool.sql.append((sql, params))
                return _Cur()

        yield _Conn()


def test_read_result_returns_the_stored_plan_for_prime():
    from app.dream_day_store import RESULT_SQL, read_result

    plan = [{"action": "drop", "concern_id": "a", "reason": "r", "refs": []}, "junk"]
    pool = _ResultPool({"day": date(2026, 11, 17), "run_id": "dream-day-2026-11-17", "produced": 1,
                        "plan": __import__("json").dumps(plan)})
    out = asyncio.run(read_result(pool, date(2026, 11, 17)))
    assert out == {"day": "2026-11-17", "found": True, "run_id": "dream-day-2026-11-17", "produced": True,
                   "plan": [plan[0]]}
    assert pool.sql == [(RESULT_SQL, (date(2026, 11, 17),))] and RESULT_SQL.lstrip().upper().startswith("SELECT")


def test_read_result_without_a_row_is_none():
    from app.dream_day_store import read_result

    assert asyncio.run(read_result(_ResultPool(None), date(2026, 11, 17))) is None


# --- the grounding gate (2026-11-18 hardening) --------------------------------------------------------------

from orion.memory.dream_day import (  # noqa: E402
    CheckedMemory, apply_grounding, grounding_prompt, grounding_summary, judge_memories, parse_verdict,
)

OVERCLAIM = ("That night you played Zelda for 59 minutes while the office camera watched the kids packing a box; "
             "we traded only eleven chat turns.")
SUPPORTED = "I read a paper about spiking networks and it pulled at my reverie that afternoon."


def verdict(supported, claims=()):
    import json
    return {"text": json.dumps({"supported": supported, "unsupported_claims": list(claims)}), "finish_reason": "stop"}


def test_parse_verdict_is_strict():
    assert parse_verdict('{"supported": true, "unsupported_claims": []}').supported is True
    v = parse_verdict('```json\n{"supported": false, "unsupported_claims": ["the kids"]}\n```')
    assert v.supported is False and v.unsupported_claims == ["the kids"]
    for bad in ("", "yes", "supported: true", '{"supported": "true", "unsupported_claims": []}',
                '{"supported": true}', '{"supported": true, "unsupported_claims": "none"}',
                '{"supported": true, "unsupported_claims": [1]}', '[true]',
                # says supported but lists a claim: contradicts itself, unjudged
                '{"supported": true, "unsupported_claims": ["eleven chat turns"]}',
                'Sure! {"supported": true, "unsupported_claims": []}'):
        assert parse_verdict(bad) is None, bad


def test_the_judge_prompt_is_its_own_short_factual_prompt():
    prompt = grounding_prompt(OVERCLAIM, [EXP], RESOLVED)
    assert OVERCLAIM in prompt and f"[{EXP}] {RESOLVED[EXP]}" in prompt
    assert '"supported"' in prompt and '"unsupported_claims"' in prompt
    writer = build_prompt(DreamDayBriefV1(dream_day="dream.day", day=DAY, timeout_sec=900, llm_route="memory_distill"),
                          material())
    assert "dig dream-day" not in prompt and writer.splitlines()[0] not in prompt
    assert len(prompt) < 2500


def test_judge_memories_maps_failures_to_unjudged():
    memories = [CheckedMemory(OVERCLAIM, "about_juniper", [EXP]), CheckedMemory(SUPPORTED, "happened", [ACT]),
                CheckedMemory("x happened.", "happened", [ACT]), CheckedMemory("y happened.", "happened", [ACT]),
                CheckedMemory("z happened.", "happened", [ACT])]
    answers = [verdict(False, ["the kids", "eleven chat turns"]), verdict(True), RuntimeError("gateway down"),
               {"text": "I think it is fine", "finish_reason": "stop"}, {"text": '{"supported": tr', "finish_reason": "length"}]

    async def ask(prompt):
        a = answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return a
    out = asyncio.run(judge_memories(memories, RESOLVED, ask))
    assert [v["status"] for v in out] == ["unsupported", "supported", "unjudged", "unjudged", "unjudged"]
    assert out[0]["unsupported_claims"] == ["the kids", "eleven chat turns"]
    assert "call_failed" in out[2]["error"] and out[3]["error"] == "verdict_unparseable"
    assert out[4]["error"] == "verdict_cut_at_max_tokens"
    # No time left: never asked, unjudged.
    out = asyncio.run(judge_memories(memories[:1], RESOLVED, ask, time_left=lambda: 0.0))
    assert out[0]["status"] == "unjudged" and out[0]["error"] == "no_time_left"
    s = grounding_summary([{"status": "supported"}, {"status": "unsupported"}, {"status": "unjudged"}], enabled=True)
    assert (s["judged"], s["supported"], s["unsupported"], s["unjudged"]) == (3, 1, 1, 1)


def test_apply_grounding_writes_only_matched_supported_memories():
    parsed = parse_answer(answer_json(memories=[
        {"statement": SUPPORTED, "refs": [ACT]}, {"statement": OVERCLAIM, "refs": [EXP]},
        {"statement": "A third memory with no verdict at all.", "refs": [CYCLE]}]))
    check = check_answer(parsed, RESOLVED, [])
    verdicts = [{"statement": SUPPORTED, "refs": [ACT], "status": "supported"},
                {"statement": OVERCLAIM, "refs": [EXP], "status": "unsupported", "unsupported_claims": ["kids"]}]
    check, rows = apply_grounding(check, verdicts)
    assert [m.statement for m in check.memories] == [SUPPORTED]
    assert [r["status"] for r in rows] == ["supported", "unsupported", "unjudged"]
    assert rows[2]["error"] == "no_verdict_for_this_memory"
    # A run checkpointed before the gate (no verdicts at all) writes nothing.
    check2, rows2 = apply_grounding(check_answer(parsed, RESOLVED, []), None)
    assert check2.memories == [] and {r["status"] for r in rows2} == {"unjudged"}


def _gate_world(judge_answers):
    answers = list(judge_answers)

    async def judge(prompt):
        a = answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return a
    return World(replies=[answer_json(memories=[
        {"statement": OVERCLAIM, "purpose": "about_juniper", "refs": [EXP]},
        {"statement": SUPPORTED, "purpose": "happened", "refs": [ACT, EXP]}])], judge=judge)


def test_the_gate_refuses_an_over_claim_and_writes_the_supported_one():
    pytest.importorskip("langgraph")
    from langgraph.checkpoint.memory import InMemorySaver

    async def scenario():
        world, saver = _gate_world([verdict(False, ["the kids packing a box", "eleven chat turns"]), verdict(True)]), \
            InMemorySaver()
        result = await start_and_grant(world, saver)
        assert result["status"] == "completed" and result["produced"] is True
        # The judge ran inside the hold, on the run's lease, once per memory, before the hold was released.
        assert len(world.judged) == 2 and all(h["lease"]["lease_id"] == "h" for _, h, _ in world.judged)
        assert world.releases == ["completed"]
        from app.dream_day_graph import GROUNDING_CALL_TIMEOUT_SEC
        assert all(0 < t <= GROUNDING_CALL_TIMEOUT_SEC for _, _, t in world.judged)
        stored = world.stored
        assert [m.statement for m in stored["result"].memories] == [SUPPORTED]
        fields = stored["fields"]
        assert fields["memories_unsupported"] == 1 and fields["memories_unjudged"] == 0
        g = fields["grounding"]
        assert g["enabled"] and (g["supported"], g["unsupported"]) == (1, 1)
        refused = [v for v in g["verdicts"] if v["status"] == "unsupported"][0]
        assert refused["statement"] == OVERCLAIM and "eleven chat turns" in refused["unsupported_claims"]
        detail = finish_detail(result)
        assert detail["memories_unsupported"] == 1 and detail["memories_written"] == 1
        # The turn's limit leaves the gate its reserve.
        from app.dream_day_graph import GROUNDING_RESERVE_SEC
        assert world.turns[0].timeout_sec <= 900 - 30 - GROUNDING_RESERVE_SEC
    asyncio.run(scenario())


def test_a_judge_failure_means_not_written_and_a_night_of_refusals_grades_zero():
    pytest.importorskip("langgraph")
    from langgraph.checkpoint.memory import InMemorySaver

    async def scenario():
        world, saver = _gate_world([RuntimeError("gateway_error: seat busy"),
                                    {"text": "looks fine to me", "finish_reason": "stop"}]), InMemorySaver()
        result = await start_and_grant(world, saver)
        assert result["status"] == "completed"           # a judge failure never fails the run or re-runs the turn
        assert len(world.turns) == 1
        assert world.stored["result"].memories == []     # never written by default
        assert result["produced"] is False and result["produced_note"] == NO_TRACE_NOTE
        fields = world.stored["fields"]
        assert fields["memories_unjudged"] == 2 and fields["memories_unsupported"] == 0
        errors = [v["error"] for v in fields["grounding"]["verdicts"]]
        assert "call_failed" in errors[0] and errors[1] == "verdict_unparseable"
    asyncio.run(scenario())


def test_gate_off_keeps_the_ref_check_alone():
    pytest.importorskip("langgraph")
    from langgraph.checkpoint.memory import InMemorySaver

    async def scenario():
        world, saver = World(), InMemorySaver()
        result = await start_and_grant(world, saver)
        assert result["produced"] is True and world.judged == []
        assert world.stored["fields"]["grounding"]["enabled"] is False
        assert world.turns[0].timeout_sec > 900 - 30 - 1
    asyncio.run(scenario())


def test_the_gate_stops_asking_when_the_attempt_runs_out_of_time():
    pytest.importorskip("langgraph")
    from langgraph.checkpoint.memory import InMemorySaver

    async def scenario():
        world = _gate_world([verdict(True), verdict(True)])
        inner = world.judge

        async def slow(prompt):
            world.now += timedelta(seconds=900)      # the first call eats the rest of the attempt
            return await inner(prompt)
        world.judge = slow
        result = await start_and_grant(world, InMemorySaver())
        g = world.stored["fields"]["grounding"]
        assert len(world.judged) == 1                  # the second memory was never asked
        assert [v["status"] for v in g["verdicts"]] == ["supported", "unjudged"]
        assert g["verdicts"][1]["error"] == "no_time_left"
        assert [m.statement for m in world.stored["result"].memories] == [OVERCLAIM]
        assert result["status"] == "completed"
    asyncio.run(scenario())


def test_a_failed_read_in_the_gate_leaves_every_memory_unjudged():
    pytest.importorskip("langgraph")
    from langgraph.checkpoint.memory import InMemorySaver

    async def scenario():
        world = _gate_world([verdict(True), verdict(True)])
        world.resolve_fail = 1                         # the gate's own read fails; persist's read works
        result = await start_and_grant(world, InMemorySaver())
        assert result["status"] == "completed" and world.judged == [] and len(world.turns) == 1
        assert world.stored["result"].memories == [] and result["produced"] is False
        g = world.stored["fields"]["grounding"]
        assert g["unjudged"] == 2 and g["supported"] == 0
    asyncio.run(scenario())


def test_a_flag_flip_between_dream_and_persist_never_opens_the_gate():
    pytest.importorskip("langgraph")
    from langgraph.checkpoint.memory import InMemorySaver

    async def scenario():
        # Dreamt with the gate off (or before it existed), persisted with it on: no verdicts, nothing written.
        world = _gate_world([verdict(True), verdict(True)])
        judge, world.judge, world.persist_fail = world.judge, None, 1
        saver = InMemorySaver()
        result = await start_and_grant(world, saver)
        assert result["retry_node"] == "persist" and world.stored is None
        world.judge = judge
        result = await after_backoff(world, saver)
        assert result["status"] == "completed" and world.judged == []
        assert world.stored["result"].memories == [] and world.stored["fields"]["grounding"]["unjudged"] == 2

        # Dreamt with the gate on, persisted after it was turned off: the run's own verdicts still gate it.
        world = _gate_world([verdict(False, ["the kids"]), verdict(True)])
        world.persist_fail = 1
        saver = InMemorySaver()
        result = await start_and_grant(world, saver)
        assert result["retry_node"] == "persist" and len(world.judged) == 2
        world.judge = None
        result = await after_backoff(world, saver)
        assert result["status"] == "completed"
        assert [m.statement for m in world.stored["result"].memories] == [SUPPORTED]
        assert world.stored["fields"]["memories_unsupported"] == 1
    asyncio.run(scenario())


def test_a_nul_in_a_verdict_never_reaches_the_jsonb_row():
    # Postgres jsonb refuses a real NUL character (it would fail the night's row on every retry).
    answers = [{"text": '{"supported": false, "unsupported_claims": ["the kids\\u0000"]}', "finish_reason": "stop"},
               {"text": '{"supported": false, "unsupported_claims": ["x"]}\x00', "finish_reason": "stop"}]

    async def ask(prompt):
        return answers.pop(0)
    out = asyncio.run(judge_memories([CheckedMemory(OVERCLAIM, "happened", [EXP]),
                                      CheckedMemory(SUPPORTED, "happened", [ACT])], RESOLVED, ask))
    assert out[0]["unsupported_claims"] == ["the kids"] and out[1]["unsupported_claims"] == ["x"]
    assert all("\x00" not in str(v) for row in out for v in row.values())
