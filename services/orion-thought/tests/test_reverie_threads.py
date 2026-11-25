"""Reveries that ask, Patch 3: waiting questions, read-back, threads.

Spec: docs/superpowers/specs/2026-11-18-reverie-questions-design.md, "Threads (idea 9) and read-back".
Live problem (2026-11-25): Hub refused 2 of 4 reverie questions because a focus was already pending
(keep_pending), and they were lost. They now wait and are offered again when the slot is free.
"""

from __future__ import annotations

import asyncio
import dataclasses
import importlib
from datetime import datetime, timedelta, timezone

import pytest

NOW = datetime(2026, 11, 25, 18, 0, tzinfo=timezone.utc)
LAPSE = timedelta(days=14)
REF = "self_bet_bin:turn.self.fetches_outside:0.8-0.9"


def rev():
    return importlib.import_module("app.reverie")


def th():
    return importlib.import_module("app.reverie_threads")


def _slot_full_note():
    from orion.curiosity.focus import PENDING_KEPT

    return f"focus refused: {PENDING_KEPT} (pending: 'something else')"


def _waiting_row(qid="w1", *, created=NOW - timedelta(hours=2), question="Which fetches did I bet on and skip?"):
    return {"question_id": qid, "thread_id": f"t-{qid}", "thread_key": "lost_bet:x", "seed_kind": "lost_bet",
            "seed_refs": [REF], "seed_material": "a bet I keep losing", "question": question, "why": "bets miss",
            "refs": [REF], "cites": [], "correlation_id": "c", "model_id": "m", "created_at": created,
            "route_note": _slot_full_note()}


class Fake:
    """Every dep a poll touches, recording what it did."""

    def __init__(self, *, waiting=(), slot_pending=False, routes=0, route_result=("focus:f9", "queued"),
                 reoffer_texts=(), thread_rows=(), stored=None, reconcile=()):
        self.waiting = list(waiting)
        self.slot = slot_pending
        self.routes = routes
        self.route_result = route_result
        self.reoffer_texts = list(reoffer_texts)
        self.thread_rows_ = list(thread_rows)
        self.stored = stored or {}
        self.reconcile = list(reconcile)
        self.asked, self.asked_threads, self.persisted, self.routed, self.updates = 0, [], [], [], []
        self.outcomes, self.thread_writes, self.readers_called = [], [], []

    def deps(self):
        r = rev()

        async def arousal():
            return "idle"

        async def gate_rows():
            return {"last_winner_at": NOW - timedelta(minutes=30), "latest_tick_at": NOW,
                    "rows_since_stretch": len(self.persisted), "rows_last_24h": len(self.persisted)}

        async def state():
            return {"last_used": {}, "spent_refs": set(), "threads": {}}

        def reader(kind):
            async def read(spent):
                self.readers_called.append(kind)
                if kind != "lost_bet":
                    return []
                from orion.schemas.reverie import ReverieSeedV1

                return [ReverieSeedV1(kind="lost_bet", material="a fresh bet", refs=["self_bet_bin:a:0.1-0.2"],
                                      thread_key="lost_bet:a:0.1-0.2")]
            return read

        async def open_texts():
            return []

        async def now_cue(corr, run_ref):
            return r.NowCue()

        async def ask(seed, corr, now_block, thread=None):
            self.asked += 1
            self.asked_threads.append((seed, thread))
            return {"question": "What did the next fetch find instead?", "why": "follow the dig",
                    "refs": list(seed.refs)}, "model-x"

        async def persist(q):
            self.persisted.append(q)
            return True

        async def routes():
            return self.routes

        async def route(q):
            self.routed.append(q.question_id)
            return self.route_result

        async def update(qid, to, note, offer_status=None):
            self.updates.append((qid, to, note, offer_status))
            return True

        async def use(cue, q):
            return None

        async def sweep():
            return {"adopted": 0}

        async def waiting():
            return list(self.waiting)

        async def slot():
            return self.slot

        async def reoffer_texts(qid):
            return list(self.reoffer_texts)

        async def reconcile_rows():
            return list(self.reconcile)

        async def write_outcome(qid, outcome):
            self.outcomes.append((qid, outcome))
            return True

        async def thread_rows():
            return list(self.thread_rows_), dict(self.stored)

        async def write_threads(statuses):
            self.thread_writes.extend(statuses)
            return len(statuses)

        return r.ReverieDeps(
            read_arousal=arousal, gate_rows=gate_rows, seed_state=state,
            seed_readers={k: reader(k) for k in ("absence", "lost_bet", "far_memory")},
            open_texts=open_texts, now_cue=now_cue, ask=ask, persist=persist, focus_routes_24h=routes,
            route=route, update_route=update, record_use=use, now=lambda: NOW,
            sweep_waiting=sweep, waiting=waiting, slot_pending=slot, reoffer_open_texts=reoffer_texts,
            reconcile_rows=reconcile_rows, write_outcome=write_outcome, thread_rows=thread_rows,
            write_threads=write_threads, threads_enabled=True,
        )


@pytest.fixture
def r(monkeypatch):
    mod = rev()
    mod._STRETCH_REFUSALS.clear()
    mod._STRETCH_NO_MATERIAL.clear()
    mod._REOFFER_TRIED.clear()
    monkeypatch.setattr(mod.settings, "reverie_focus_max_per_day", 3)
    monkeypatch.setattr(mod.settings, "reverie_thread_lapse_days", 14.0)
    yield mod


# --- 1. waiting and re-offer -------------------------------------------------------------------------


def test_slot_full_refusal_is_recognised_from_hubs_own_words():
    from orion.curiosity.focus import JUNIPER_PROTECTED

    t = th()
    assert t.is_slot_full_refusal(_slot_full_note())
    assert t.is_slot_full_refusal(f"focus refused: {JUNIPER_PROTECTED} (pending: 'x')")
    assert not t.is_slot_full_refusal("focus refused: the reverie focus cap is reached (3 of 3 in 24 h)")
    assert not t.is_slot_full_refusal("focus: no answer (TimeoutError; answer unknown)")
    assert not t.is_slot_full_refusal(None)


def test_a_question_turned_away_for_a_full_slot_waits(r):
    f = Fake(route_result=("none", _slot_full_note()))
    q = asyncio.run(r.run_reverie_once(f.deps()))
    assert q.accepted and q.routed_to == "none" and "waiting for the focus slot" in q.route_note
    assert f.updates[-1][3] == "waiting"


def test_a_question_refused_for_the_cap_does_not_wait(r):
    f = Fake(routes=3)
    q = asyncio.run(r.run_reverie_once(f.deps()))
    assert q.routed_to == "none" and "cap reached" in q.route_note and f.updates[-1][3] is None


def test_no_reoffer_while_the_slot_is_full_then_reoffer_when_it_frees(r):
    f = Fake(waiting=[_waiting_row()], slot_pending=True)
    deps = f.deps()
    # Slot full: no re-offer; the poll goes on to a normal reverie (which asks the model once).
    asyncio.run(r.run_reverie_once(deps))
    assert "w1" not in f.routed
    f2 = Fake(waiting=[_waiting_row()], slot_pending=False)
    q = asyncio.run(r.run_reverie_once(f2.deps()))
    assert q.question_id == "w1" and q.routed_to == "focus:f9"
    assert f2.routed == ["w1"] and f2.updates[-1][0] == "w1" and f2.updates[-1][3] == "sent"
    assert "re-offered" in f2.updates[-1][2]


def test_reoffer_makes_no_model_call_and_spends_no_stretch(r):
    f = Fake(waiting=[_waiting_row()])
    asyncio.run(r.run_reverie_once(f.deps()))
    assert f.asked == 0 and f.persisted == []
    # Nothing waits any more: the next poll in the same stretch still writes a reverie.
    f.waiting = []
    q = asyncio.run(r.run_reverie_once(f.deps()))
    assert q is not None and f.asked == 1 and len(f.persisted) == 1


def test_reoffer_takes_the_oldest_waiting_question_first(r):
    f = Fake(waiting=[_waiting_row("new", created=NOW - timedelta(hours=1)),
                      _waiting_row("old", created=NOW - timedelta(hours=5))])
    q = asyncio.run(r.run_reverie_once(f.deps()))
    assert q.question_id == "old" and f.routed == ["old"]


def test_reoffer_holds_the_daily_focus_cap(r):
    f = Fake(waiting=[_waiting_row()], routes=3)
    asyncio.run(r.run_reverie_once(f.deps()))
    assert "w1" not in f.routed


def test_a_waiting_question_older_than_the_lapse_window_lapses(r):
    f = Fake(waiting=[_waiting_row(created=NOW - timedelta(days=15))])
    asyncio.run(r.run_reverie_once(f.deps()))
    assert ("w1", "none", "lapsed: waited 14 days for the focus slot", "lapsed") in f.updates
    assert "w1" not in f.routed


def test_reoffer_applies_the_near_duplicate_rule_again(r):
    f = Fake(waiting=[_waiting_row()], reoffer_texts=["Which fetches did I bet on and then skip?"])
    q = asyncio.run(r.run_reverie_once(f.deps()))
    assert q.question_id == "w1" and q.routed_to == "none" and f.routed == []
    assert f.updates[-1][3] == "dropped" and "near_duplicate" in f.updates[-1][2]


def test_a_reoffer_turned_away_again_keeps_waiting(r):
    f = Fake(waiting=[_waiting_row()], route_result=("none", _slot_full_note()))
    asyncio.run(r.run_reverie_once(f.deps()))
    assert f.updates[-1][0] == "w1" and f.updates[-1][3] == "waiting"


def test_pick_reoffer_is_pure_and_ordered():
    t = th()
    rows = [_waiting_row("b", created=NOW), _waiting_row("a", created=NOW - timedelta(hours=1))]
    assert t.pick_reoffer(rows, slot_pending=False, routed_24h=0, cap=3).question_id == "a"
    assert t.pick_reoffer(rows, slot_pending=True, routed_24h=0, cap=3).reason == "slot_full"
    assert t.pick_reoffer(rows, slot_pending=False, routed_24h=3, cap=3).reason == "focus_cap"
    assert t.pick_reoffer(rows, slot_pending=False, routed_24h=0, cap=0).reason == "focus_routing_off"
    assert t.pick_reoffer([], slot_pending=False, routed_24h=0, cap=3).reason == "none_waiting"


def test_patch3_off_is_patch1(r):
    f = Fake(waiting=[_waiting_row()], route_result=("none", _slot_full_note()))
    deps = dataclasses.replace(f.deps(), threads_enabled=False)
    q = asyncio.run(r.run_reverie_once(deps))
    assert q.question_id != "w1" and f.asked == 1 and f.updates[-1][3] is None
    assert q.route_note.startswith("focus refused:")


# --- 2. read-back ------------------------------------------------------------------------------------

# Fixtures shaped like store.RECONCILE_SQL rows, values from run 6eed301f7743 (2026-11-25).
RAN = {"question_id": "q-ran", "routed_to": "focus:58007734", "focus_status": "ran",
       "status_at": NOW - timedelta(hours=4), "offered_run_id": "6eed301f7743", "disposition": "explored",
       "focus_refused_reason": None, "register": "operational",
       "register_reason": "The finding analyzes database records and timestamps.", "novelty": 0.230919,
       "finish_found": True, "continue_line": "true", "finding_text": "I took the queued focus. The arrow reversed.",
       "continue_note": "Explored the focus and the arrow reversed. Still unknown: (1) the hard-edge claim."}


def test_outcome_from_a_finished_dig():
    o = th().build_outcome(RAN, now=NOW)
    assert o["dug"] is True and o["continue_line"] is True and o["disposition"] == "explored"
    assert o["finding"] == "operational" and o["run_id"] == "6eed301f7743" and o["novelty"] == pytest.approx(0.2309, 1e-3)
    assert o["continue_note"].startswith("Explored the focus") and o["finding_excerpt"].startswith("I took")
    from orion.reverie.questions import outcome_words

    assert outcome_words({"outcome": o}).startswith("explored; operational; Explored the focus")


def test_outcome_waits_for_the_finish_row_but_not_forever():
    t = th()
    fresh = {**RAN, "finish_found": False, "continue_line": None, "status_at": NOW - timedelta(minutes=2)}
    assert t.build_outcome(fresh, now=NOW) is None
    stale = {**fresh, "status_at": NOW - timedelta(hours=2)}
    o = t.build_outcome(stale, now=NOW)
    assert o["dug"] is True and o["continue_line"] is None and o["finish_found"] is False


def test_outcome_none_while_the_focus_is_open_and_undug_when_it_expired():
    t = th()
    assert t.build_outcome({**RAN, "focus_status": "pending"}, now=NOW) is None
    assert t.build_outcome({**RAN, "focus_status": "taken"}, now=NOW) is None
    assert t.build_outcome({**RAN, "focus_status": "expired", "offered_run_id": None}, now=NOW) == {
        "dug": False, "focus_status": "expired"}
    assert t.build_outcome({**RAN, "focus_status": None}, now=NOW) == {"dug": False, "focus_status": "missing"}


def test_reconcile_writes_finished_outcomes_only(r):
    f = Fake(reconcile=[RAN, {**RAN, "question_id": "q-open", "focus_status": "pending"}])
    assert asyncio.run(r.reconcile_outcomes(f.deps())) == 1
    assert [q for q, _ in f.outcomes] == ["q-ran"]


# --- 3. threads --------------------------------------------------------------------------------------


def _q(qid, at, *, thread="T", question="Q?", none=None, refused=None, outcome=None):
    return {"question_id": qid, "thread_id": thread, "thread_key": "far_memory:a:b", "seed_kind": "far_memory",
            "seed_refs": ["episode_memory:a", "episode_memory:b"], "seed_material": "an old memory beside a new one",
            "question": question, "none_reason": none, "refused_reason": refused, "outcome": outcome,
            "created_at": at, "routed_at": None, "outcome_at": None}


DUG_CONTINUE = {"dug": True, "continue_line": True, "disposition": "explored", "finding": "operational",
                "finding_excerpt": "the arrow reversed", "continue_note": "is SimEarth aimed at me?"}
DUG_STOP = {**DUG_CONTINUE, "continue_line": False}


def test_concluded_when_the_dig_ends_with_no_continue():
    status, _, wants = th().thread_status([_q("a", NOW - timedelta(days=1), outcome=DUG_STOP)], now=NOW, lapse=LAPSE)
    assert status == "concluded" and not wants


def test_exhausted_after_two_reveries_in_a_row_with_no_question():
    rows = [_q("a", NOW - timedelta(days=2), outcome=DUG_CONTINUE),
            _q("b", NOW - timedelta(days=1), question=None, none="nothing more"),
            _q("c", NOW - timedelta(hours=1), question=None, none="still nothing")]
    status, _, wants = th().thread_status(rows, now=NOW, lapse=LAPSE)
    assert status == "exhausted" and not wants
    # One none_reason is not enough: the thread still wants continuing.
    status, _, wants = th().thread_status(rows[:2], now=NOW, lapse=LAPSE)
    assert status == "open" and wants


def test_lapsed_after_fourteen_days_without_activity():
    status, _, _ = th().thread_status([_q("a", NOW - timedelta(days=15))], now=NOW, lapse=LAPSE)
    assert status == "lapsed"
    status, _, _ = th().thread_status([_q("a", NOW - timedelta(days=13))], now=NOW, lapse=LAPSE)
    assert status == "open"


def test_a_followup_already_asked_carries_the_thread():
    rows = [_q("a", NOW - timedelta(days=2), outcome=DUG_CONTINUE), _q("b", NOW - timedelta(days=1))]
    status, _, wants = th().thread_status(rows, now=NOW, lapse=LAPSE)
    assert status == "open" and not wants


def test_thread_book_writes_only_changes_and_skips_closed_threads():
    t = th()
    rows = [_q("a", NOW - timedelta(days=1), outcome=DUG_CONTINUE)]
    book = t.thread_book(rows, {}, now=NOW, lapse=LAPSE)
    assert [(s.thread_id, s.status) for s in book.statuses] == [("T", "open")]
    assert book.continuation.thread_id == "T" and book.continuation.still_open == "is SimEarth aimed at me?"
    stored = {"T": {"status": "open", "status_reason": "continue", "last_question_id": "a"}}
    assert t.thread_book(rows, stored, now=NOW, lapse=LAPSE).statuses == []
    closed = t.thread_book(rows, {"T": {"status": "concluded"}}, now=NOW, lapse=LAPSE)
    assert closed.statuses == [] and closed.continuation is None


def test_thread_first_before_the_picker_rotates_kinds(r):
    f = Fake(thread_rows=[_q("a", NOW - timedelta(days=1), outcome=DUG_CONTINUE)])
    q = asyncio.run(r.run_reverie_once(f.deps()))
    assert f.readers_called == []  # the picker never ran
    seed, thread = f.asked_threads[0]
    assert thread is not None and thread.thread_id == "T" and thread.earlier_question == "Q?"
    assert seed.refs == ["episode_memory:a", "episode_memory:b"] and seed.thread_id == "T"
    assert q.thread_id == "T" and q.accepted and q.routed_to == "focus:f9"


def test_without_a_continuing_thread_the_picker_runs(r):
    f = Fake(thread_rows=[_q("a", NOW - timedelta(days=1), outcome=DUG_STOP)])
    q = asyncio.run(r.run_reverie_once(f.deps()))
    assert "lost_bet" in f.readers_called and f.asked_threads[0][1] is None and q.seed_kind == "lost_bet"
    assert [(s.status) for s in f.thread_writes] == ["concluded"]


def test_the_prompt_shows_the_thread_and_the_plan_carries_it(r):
    from orion.schemas.reverie import ReverieSeedV1

    t = th()
    book = t.thread_book([_q("a", NOW - timedelta(days=1), outcome=DUG_CONTINUE)], {}, now=NOW, lapse=LAPSE)
    seed = ReverieSeedV1(kind="far_memory", material="m", refs=["episode_memory:a"], thread_key="k")
    req = r.build_question_plan_request(seed, correlation_id="c", thread=book.continuation)
    assert req.context["thread_earlier_question"] == "Q?"
    assert "arrow reversed" in req.context["thread_dig_found"]
    assert req.context["thread_still_open"] == "is SimEarth aimed at me?"
    from pathlib import Path

    from jinja2 import Environment, FileSystemLoader

    repo = Path(__file__).resolve().parents[3]
    env = Environment(loader=FileSystemLoader(str(repo / "orion" / "cognition" / "prompts")))
    text = env.get_template("reverie_question.j2").render(**req.context)
    assert "THIS IS A THREAD YOU ARE CONTINUING" in text and "is SimEarth aimed at me?" in text
    plain = env.get_template("reverie_question.j2").render(
        **r.build_question_plan_request(seed, correlation_id="c").context)
    assert "THREAD YOU ARE CONTINUING" not in plain


def test_inquiry_ledger_is_not_built():
    """#3442 path (b) is not built here: no curiosity_inquiry writer in orion-thought."""
    from pathlib import Path

    app = Path(__file__).resolve().parents[1] / "app"
    src = "\n".join(p.read_text() for p in app.glob("*.py"))
    assert "INSERT INTO curiosity_inquiry" not in src
