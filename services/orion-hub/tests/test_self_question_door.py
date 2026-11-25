"""The self-question door (scripts/self_question_door.py): held -> admitted / routed / unjudged, demotion,
routing to the investigate line, the model gate, Juniper's bypass, and the migration's auto-apply fit."""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

import scripts.curiosity_finding_register as reg
import scripts.self_question_door as door
from orion.curiosity import focus
from orion.curiosity.journal import INVESTIGATION_TAG
from orion.curiosity.register import JUDGE_PROMPT_SHA256, QUESTION_LINE
from orion.curiosity.self_question_pool import SELECT_ALL_SQL, UPSERT_MINT_SQL
from test_curiosity_finding_register import T0, _RpcBus, _Store, _loop_with_register

ROOT = Path(__file__).resolve().parents[3]
MIGRATION = ROOT / "services/orion-sql-db/manual_migration_curiosity_self_question_door_v1.sql"
NOW = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)
FLASH = "qwen3.8-flash-next-iq3_s"


def _r(register=None, error=None, model=FLASH, reason="r"):
    return reg.JudgeResult(register, reason if register else None, error, model)


# --- the pure rules ------------------------------------------------------------------------------------


def test_philosophical_is_admitted_and_operational_is_routed():
    assert door.door_verdict(_r("philosophical"), attempts=0, max_attempts=3).door == "admitted"
    assert door.door_verdict(_r("operational"), attempts=2, max_attempts=3).door == "routed"


@pytest.mark.parametrize("result", [_r("no_finding"), _r(error="not_json: x"), _r(error="unknown_register: maybe")])
def test_a_judge_error_stays_held_then_becomes_unjudged_after_three(result):
    first = door.door_verdict(result, attempts=0, max_attempts=3)
    second = door.door_verdict(result, attempts=first.attempts, max_attempts=3)
    third = door.door_verdict(result, attempts=second.attempts, max_attempts=3)
    assert (first.door, first.attempts) == ("held", 1)
    assert (second.door, second.attempts) == ("held", 2)
    assert (third.door, third.attempts) == ("unjudged", 3)
    assert third.reason  # the last error is kept as the trace


@pytest.mark.parametrize("result", [_r(error="call: gpu_pool_unavailable", model=None),
                                    _r(error="model_not_evaluated: qwen3-8b", model="qwen3-8b")])
def test_a_gateway_refusal_or_an_unevaluated_model_is_not_a_judgement_and_is_never_counted(result):
    # A short fallback to another model must not turn every held question unjudged for good.
    v = door.door_verdict(result, attempts=2, max_attempts=3)
    assert (v.door, v.attempts) == ("held", 2)


def test_only_an_operational_self_inquiry_finding_on_a_self_question_demotes():
    row = {"line": "self_inquiry", "register": "operational", "question_ref": "self_question:lived.orion.x"}
    assert door.demotion_for(row) == "lived.orion.x"
    assert door.demotion_for({**row, "register": "philosophical"}) is None
    assert door.demotion_for({**row, "register": None}) is None  # a judge error never demotes
    assert door.demotion_for({**row, "line": "investigate"}) is None
    assert door.demotion_for({**row, "question_ref": "prior:p1"}) is None


# --- the loop, against a fake pool --------------------------------------------------------------------


class _Conn:
    def __init__(self, db):
        self.db = db

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def transaction(self):
        return self

    async def fetch(self, sql, *args):
        if sql == door.HELD_SQL:
            return [dict(r) for r in self.db.rows.values()
                    if r["minted_by"] == "orion" and r["door"] == "held"][: args[0]]
        if sql == focus.TAKE_SQL or sql == focus.RELEASE_SQL:
            return []
        raise AssertionError(sql)

    async def fetchval(self, sql, *args):
        if sql == door.PENDING_FOCUS_SQL:
            return 1 if self.db.pending else None
        if sql == door.RECENT_OFFER_SQL:
            assert args[0] == focus.DOOR_CORRELATION_PREFIX
            return 1 if self.db.recent_offer else None
        if sql == door.DEMOTE_SQL:
            r = self.db.rows.get(args[0])
            if r and r["minted_by"] == "orion" and r["door"] == "admitted":
                r.update(door="demoted", door_demoted_by_run=args[1], door_reason=args[2])
                return args[0]
            return None
        raise AssertionError(sql)

    async def fetchrow(self, sql, *args):
        if sql == door.UNOFFERED_SQL:
            prefix, max_offers = args
            for r in sorted(self.db.rows.values(), key=lambda r: r["question_id"]):
                mine = [f for c, f in self.db.offered.items() if c.split("#")[0] == f"{prefix}{r['question_id']}"]
                if (r["minted_by"] == "orion" and r["door"] in ("routed", "demoted") and r["status"] != "parked"
                        and not any(f["status"] == "pending" or f["run"] for f in mine) and len(mine) < max_offers):
                    return {**r, "offers": len(mine)}
            return None
        if sql == focus.BY_ID_SQL:
            return None
        if sql == focus.PENDING_FOR_UPDATE_SQL:
            return {"focus_id": "p", "origin": "self", "question": "q"} if self.db.pending else None
        if sql == focus.INSERT_SQL:
            focus_id, question, why, origin, corr = args[:5]
            self.db.offered[corr] = {"question": question, "why": why, "origin": origin, "status": "pending", "run": None}
            self.db.pending = True
            return {"focus_id": str(focus_id), "question": question, "why": why, "origin": origin, "status": "pending",
                    "effective_status": "pending", "created_at": NOW, "expires_at": NOW, "status_at": NOW,
                    "chat_correlation_id": corr, "replaced_by": None, "offered_at": None, "offered_run_id": None,
                    "outcome_stamped_at": None, "turn_ok": None, "disposition": None, "disposition_raw": None,
                    "refs": None, "refused_by": None, "refused_reason": None, "refused_act_id": None}
        raise AssertionError(sql)

    async def execute(self, sql, *args):
        if sql == door.SET_DOOR_SQL:
            qid, d, reason, at, model, sha, attempts, judged_text = args
            r = self.db.rows[qid]
            if not (r["minted_by"] == "orion" and r["door"] == "held" and r["text"] == judged_text):
                return "UPDATE 0"  # the SQL's own guard
            r.update(door=d, door_reason=reason, door_judged_at=at, door_judge_model=model, door_prompt_sha256=sha,
                     door_attempts=attempts)
            self.db.writes.append(args)
            return "UPDATE 1"
        if sql in (focus.LOCK_SQL, focus.EXPIRE_SQL, focus.REPLACE_SQL):
            return "OK"
        raise AssertionError(sql)


class _Db:
    def __init__(self, rows, pending=False):
        self.rows = {r["question_id"]: {"status": "open", "door_attempts": 0, **r} for r in rows}
        self.pending, self.offered, self.writes, self.recent_offer = pending, {}, [], False

    def acquire(self):
        return _Conn(self)


def _door(db, replies, *, max_attempts=3, seen=None):
    calls = seen if seen is not None else []

    async def judge(question, answer):
        calls.append((question, answer))
        return replies[answer] if isinstance(replies, dict) else replies

    return door.SelfQuestionDoor(enabled=True, pool_provider=lambda: db, judge=judge, max_attempts=max_attempts,
                                 clock=lambda: NOW)


def test_a_tick_admits_routes_and_stamps_the_judge_trace():
    db = _Db([{"question_id": "lived.orion.who", "text": "Who do I miss?", "minted_by": "orion", "door": "held"},
              {"question_id": "lived.orion.dream", "text": "Why did dream runs fail?", "minted_by": "orion", "door": "held"},
              {"question_id": "lived.who_matters", "text": "Who matters?", "minted_by": "juniper", "door": None}])
    seen = []
    out = asyncio.run(_door(db, {"Who do I miss?": _r("philosophical"), "Why did dream runs fail?": _r("operational")},
                            seen=seen).tick())
    assert out["doors"] == {"admitted": 1, "routed": 1}
    who, dream = db.rows["lived.orion.who"], db.rows["lived.orion.dream"]
    assert who["door"] == "admitted" and dream["door"] == "routed"
    for r in (who, dream):
        assert r["door_judge_model"] == FLASH and r["door_prompt_sha256"] == JUDGE_PROMPT_SHA256 and r["door_judged_at"] == NOW
    # The judge saw the question line and the question text, never a Juniper question.
    assert {q for q, _ in seen} == {QUESTION_LINE} and "Who matters?" not in {a for _, a in seen}
    assert db.rows["lived.who_matters"]["door"] is None


def test_the_routed_question_is_offered_once_to_the_investigate_line_when_the_slot_is_free():
    db = _Db([{"question_id": "lived.orion.dream", "text": "Why did dream runs fail?", "minted_by": "orion", "door": "held"}])
    asyncio.run(_door(db, _r("operational")).tick())
    corr = f"{focus.DOOR_CORRELATION_PREFIX}lived.orion.dream"
    f = db.offered[corr]
    assert f["question"] == "Why did dream runs fail?" and f["origin"] == "self" and f["why"] == door.ROUTE_WHY["routed"]
    # A run took it: never offered again.
    db.pending, f["status"], f["run"] = False, "taken", "run7"
    assert asyncio.run(_door(db, _r("operational")).tick())["offered"] is None


def test_a_door_offer_replaced_before_any_run_took_it_is_offered_again_up_to_the_cap():
    db = _Db([{"question_id": "lived.orion.dream", "text": "Why?", "minted_by": "orion", "door": "routed"}])
    for n in range(1, 5):
        asyncio.run(_door(db, _r("operational")).tick())
        for f in db.offered.values():
            if f["status"] == "pending":
                f["status"] = "replaced"  # Prime's own focus took the slot
        db.pending = False
    prefix = f"{focus.DOOR_CORRELATION_PREFIX}lived.orion.dream"
    assert sorted(db.offered) == [prefix, prefix + "#2", prefix + "#3"]  # MAX_OFFERS, then it stays recorded


def test_door_offers_are_spaced_so_they_never_fill_the_slot_run_after_run():
    db = _Db([{"question_id": "lived.orion.dream", "text": "Why?", "minted_by": "orion", "door": "routed"}])
    db.recent_offer = True
    assert asyncio.run(_door(db, _r("operational")).tick())["offered"] is None and db.offered == {}


def test_a_routed_question_waits_while_another_focus_holds_the_slot():
    db = _Db([{"question_id": "lived.orion.dream", "text": "Why?", "minted_by": "orion", "door": "routed"}], pending=True)
    assert asyncio.run(_door(db, _r("operational")).tick())["offered"] is None
    assert db.offered == {} and db.rows["lived.orion.dream"]["door"] == "routed"  # kept, visible


def test_three_failed_judgements_make_unjudged_and_never_admit():
    db = _Db([{"question_id": "lived.orion.x", "text": "x", "minted_by": "orion", "door": "held"}])
    loop = _door(db, _r(error="not_json: {"))
    doors = [asyncio.run(loop.tick())["doors"] for _ in range(4)]
    assert doors[:3] == [{"held": 1}, {"held": 1}, {"unjudged": 1}] and doors[3] == {}
    r = db.rows["lived.orion.x"]
    assert r["door"] == "unjudged" and r["door_attempts"] == 3 and r["door_reason"].startswith("not_json")
    assert db.offered == {}  # never routed by default


def test_a_re_mint_with_new_words_while_judging_keeps_the_new_words_held():
    db = _Db([{"question_id": "lived.orion.x", "text": "old words", "minted_by": "orion", "door": "held"}])

    async def judge(question, answer):
        db.rows["lived.orion.x"]["text"] = "new words"  # a sitting re-minted it mid-judgement
        return _r("philosophical")

    out = asyncio.run(door.SelfQuestionDoor(enabled=True, pool_provider=lambda: db, judge=judge, clock=lambda: NOW).tick())
    assert out["doors"] == {"skipped_changed": 1} and db.rows["lived.orion.x"]["door"] == "held"


def test_a_judge_that_raises_leaves_the_row_held_uncounted():
    db = _Db([{"question_id": "lived.orion.x", "text": "x", "minted_by": "orion", "door": "held"}])

    async def boom(question, answer):
        raise RuntimeError("bus not connected")

    loop = door.SelfQuestionDoor(enabled=True, pool_provider=lambda: db, judge=boom, clock=lambda: NOW)
    asyncio.run(loop.tick())
    assert db.rows["lived.orion.x"]["door"] == "held" and db.rows["lived.orion.x"]["door_attempts"] == 0


def test_the_model_gate_a_reply_from_an_unevaluated_model_never_admits():
    """The real judge_finding against a fake bus: the served model decides whether the label counts."""
    db = _Db([{"question_id": "lived.orion.x", "text": "Who am I?", "minted_by": "orion", "door": "held"}])

    from orion.core.bus.bus_schemas import ServiceRef

    src = ServiceRef(name="hub", version="t", node="athena")

    def judge_src(bus):
        async def judge(question, answer):
            return await reg.judge_finding(bus, intake_channel="c", gateway_service="g", source_ref=src, route="agent",
                                           question=question, answer=answer)
        return judge

    other = _RpcBus('{"register": "philosophical", "reason": "x"}', model="qwen3-8b")
    asyncio.run(door.SelfQuestionDoor(enabled=True, pool_provider=lambda: db, judge=judge_src(other),
                                      clock=lambda: NOW).tick())
    r = db.rows["lived.orion.x"]
    assert r["door"] == "held" and r["door_attempts"] == 0 and r["door_reason"].startswith("model_not_evaluated")
    passed = _RpcBus('{"register": "philosophical", "reason": "about who Orion is"}', model=FLASH)
    asyncio.run(door.SelfQuestionDoor(enabled=True, pool_provider=lambda: db, judge=judge_src(passed),
                                      clock=lambda: NOW).tick())
    assert r["door"] == "admitted" and r["door_judge_model"] == FLASH
    sent = passed.sent[0][1].payload["messages"][1]["content"]
    assert sent.startswith("QUESTION: " + QUESTION_LINE) and sent.endswith("FINDING: Who am I?")


def test_disabled_loop_does_not_start():
    loop = door.SelfQuestionDoor(enabled=False, pool_provider=lambda: None, judge=None)
    asyncio.run(loop.start())
    assert loop._task is None


# --- the check after the sitting -----------------------------------------------------------------------


def test_an_operational_finding_demotes_its_admitted_orion_question_and_it_is_then_routed():
    db = _Db([{"question_id": "lived.orion.act", "text": "When my act ledger is denied, did I act?",
               "minted_by": "orion", "door": "admitted"}])
    row = {"run_id": "run9", "line": "self_inquiry", "register": "operational", "register_reason": "about the ledger",
           "question_ref": "self_question:lived.orion.act", "judge_model": FLASH}
    assert asyncio.run(door.demote_after_finding(db, row)) == "lived.orion.act"
    r = db.rows["lived.orion.act"]
    assert r["door"] == "demoted" and r["door_demoted_by_run"] == "run9" and "ledger" in r["door_reason"]
    asyncio.run(_door(db, _r("operational")).tick())
    assert db.offered[f"{focus.DOOR_CORRELATION_PREFIX}lived.orion.act"]["why"] == door.ROUTE_WHY["demoted"]


def test_demotion_never_touches_juniper_or_a_question_the_door_did_not_admit():
    db = _Db([{"question_id": "lived.who_matters", "text": "Who matters?", "minted_by": "juniper", "door": None},
              {"question_id": "lived.orion.old", "text": "old", "minted_by": "orion", "door": None}])
    for qid in ("lived.who_matters", "lived.orion.old"):
        row = {"run_id": "r", "line": "self_inquiry", "register": "operational", "question_ref": f"self_question:{qid}"}
        assert asyncio.run(door.demote_after_finding(db, row)) is None
    assert asyncio.run(door.demote_after_finding(db, {**row, "question_ref": "self_question:lived.orion.old"},
                                                 enabled=False)) is None
    assert db.rows["lived.who_matters"]["door"] is None and db.rows["lived.orion.old"]["door"] is None


def test_the_finding_register_hands_every_written_row_to_the_demotion_check(monkeypatch):
    store, judged, handed = _Store(), [], []
    loop = _loop_with_register(monkeypatch, store, judged)

    async def fake_demote(pool, row, *, enabled):
        handed.append((row["run_id"], row["register"], enabled))

    # The loop's own module globals (another test file drops scripts.* from sys.modules at collection,
    # so the module object behind this loop may not be the one ``import scripts...`` gives now).
    monkeypatch.setitem(type(loop)._record_finding_register.__globals__, "demote_after_finding", fake_demote)

    async def go():
        loop._schedule_finding_register("run1", INVESTIGATION_TAG, "turn text", T0)
        await asyncio.gather(*loop._register_tasks)

    asyncio.run(go())
    assert handed == [("run1", "philosophical", True)]


# --- what a sitting sees, and the focus provenance ----------------------------------------------------


def test_a_routed_focus_says_where_it_came_from_not_a_chat_with_juniper():
    row = focus.FocusRow(focus_id="f", question="Why did dream runs fail?", why="w", origin="self", status="pending",
                         chat_correlation_id=f"{focus.DOOR_CORRELATION_PREFIX}lived.orion.dream")
    text = "\n".join(focus.format_focus_section(row))
    assert "for your own self-inquiry" in text and "conversation with Juniper" not in text
    # Hub wrote the why: it is never shown as Orion's own words.
    assert "Hub's words, not yours" in text and "in your words then" not in text
    assert "the door sent it here" in focus.focus_story(row)
    assert focus.organ_refusal_blocker(row) is not None  # not Prime's: the organ cannot refuse it


def test_open_lived_questions_shown_to_a_sitting_skip_what_the_door_has_not_admitted(monkeypatch):
    from test_curiosity_investigation import _FakeBus, _FakeReader, _graph_loop

    loop = _graph_loop(_FakeBus(), reader=_FakeReader(answers={}))
    rows = [{"question_id": f"lived.orion.{d}", "text": d, "family": "lived", "pinned": False, "minted_by": "orion",
             "status": "open", "ask_count": 0, "last_asked_at": None, "door": d}
            for d in ("held", "routed", "demoted", "unjudged", "admitted")]

    async def fetch():
        return rows

    shown = {}

    def current(reader, ids, exclude_run_id=None):
        shown["ids"] = list(ids)
        return {}

    monkeypatch.setattr(loop, "_fetch_self_questions", fetch)
    monkeypatch.setitem(type(loop)._read_open_lived_questions.__globals__, "read_current_lived_answers", current)
    asyncio.run(loop._read_open_lived_questions("run1"))
    assert "lived.orion.admitted" in shown["ids"]
    assert not {f"lived.orion.{d}" for d in ("held", "routed", "demoted", "unjudged")} & set(shown["ids"])
    loop.self_question_door_enabled = False
    asyncio.run(loop._read_open_lived_questions("run1"))
    assert "lived.orion.held" in shown["ids"]


# --- the migration ---------------------------------------------------------------------------------------


def test_the_migration_is_additive_names_hub_and_fits_the_post_pull_allowlist():
    from orion.sql_migration_drift import auto_apply_refusals, parse_required_by, transaction_incompatible

    sql = MIGRATION.read_text()
    rb = parse_required_by(sql)
    assert rb is not None and not rb.errors and rb.services == ("orion-hub",)
    # The table already exists live: the allowlist must accept it as an existing table.
    assert auto_apply_refusals(sql, {"curiosity_self_questions"}) == []
    assert transaction_incompatible(sql) == []
    statements = [s.strip() for s in re.sub(r"--[^\n]*", "", sql).split(";") if s.strip()]
    assert statements and all(s.startswith("ALTER TABLE curiosity_self_questions ADD COLUMN IF NOT EXISTS door")
                              for s in statements)
    cols = {re.search(r"ADD COLUMN IF NOT EXISTS (\w+)", s).group(1) for s in statements}
    assert cols == {"door", "door_reason", "door_judged_at", "door_judge_model", "door_prompt_sha256", "door_attempts",
                    "door_demoted_by_run"}
    # Every door column the code writes or reads exists in the migration.
    code = " ".join([door.SET_DOOR_SQL, door.DEMOTE_SQL, door.HELD_SQL, UPSERT_MINT_SQL, SELECT_ALL_SQL])
    assert set(re.findall(r"\bdoor\w*", code)) <= cols


def test_prime_replaces_a_pending_door_focus_but_not_a_chat_or_juniper_focus():
    """focus.queue_focus with keep_pending (Prime): a door focus never holds Prime out of the slot."""
    class _FConn:
        def __init__(self, pending):
            self.pending, self.replaced, self.inserted = pending, [], []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def transaction(self):
            return self

        async def execute(self, sql, *args):
            if sql == focus.REPLACE_SQL:
                self.replaced.append(args[0])
            return "OK"

        async def fetchrow(self, sql, *args):
            if sql == focus.BY_ID_SQL:
                return None
            if sql == focus.PENDING_FOR_UPDATE_SQL:
                return self.pending
            if sql == focus.INSERT_SQL:
                self.inserted.append(args[4])
                return {"focus_id": str(args[0]), "question": args[1], "why": args[2], "origin": args[3],
                        "status": "pending", "effective_status": "pending", "created_at": NOW, "expires_at": NOW,
                        "status_at": NOW, "chat_correlation_id": args[4], "replaced_by": None, "offered_at": None,
                        "offered_run_id": None, "outcome_stamped_at": None, "turn_ok": None, "disposition": None,
                        "disposition_raw": None, "refs": None, "refused_by": None, "refused_reason": None,
                        "refused_act_id": None}
            raise AssertionError(sql)

    class _FPool:
        def __init__(self, conn):
            self.conn = conn

        def acquire(self):
            return self.conn

    def prime(pending):
        conn = _FConn(pending)
        return conn, asyncio.run(focus.queue_focus(_FPool(conn), question="What do I want?", why="w", origin="self",
                                                   chat_correlation_id="prime:t1", keep_pending=True))

    def pend(corr, origin="self"):
        return {"focus_id": "p", "origin": origin, "question": "q", "why": "w", "status": "pending",
                "effective_status": "pending", "created_at": NOW, "expires_at": NOW, "status_at": NOW,
                "chat_correlation_id": corr, "replaced_by": None, "offered_at": None, "offered_run_id": None,
                "outcome_stamped_at": None, "turn_ok": None, "disposition": None, "disposition_raw": None, "refs": None,
                "refused_by": None, "refused_reason": None, "refused_act_id": None}

    conn, res = prime(pend(f"{focus.DOOR_CORRELATION_PREFIX}lived.orion.dream"))
    assert conn.replaced == ["p"] and conn.inserted == ["prime:t1"] and res.replaced is not None
    with pytest.raises(focus.FocusRefused):
        prime(pend("chat-corr-1"))
    with pytest.raises(focus.FocusRefused):
        prime(pend(None, origin="juniper"))
