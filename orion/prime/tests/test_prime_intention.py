"""Intentions Orion takes on or lets go (choice-as-EFE Step -1): the pure parts.

orion/prime/intention.py (the primitive), orion/prime/intention_sources/turn_promise.py (the promise adapter), the
take_on move in orion/prime/thought.py, and the eval on the live concern d8db9d7b1bba4cf7."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from orion.prime import intention as it
from orion.prime.block import render_prime_block
from orion.prime.evals import run_intention_take_on_replay as replay
from orion.prime.intention_sources import turn_promise
from orion.prime.state import PrimeConcernV1, PrimeStateV1, observe, PrimeSnapshotV1
from orion.prime.thought import FrameInputs, PrimeAnswerV1, TakeOnMoveV1, apply_answer, build_frame, parse_answer
from orion.schemas.intention import INTENTION_SESSION_ID, IntentionSourceV1

T0 = datetime(2026, 11, 12, 7, 8, 57, tzinfo=timezone.utc)
CORR = "a129b72f-e698-425a-a435-3388cd7cd759"
QUOTE = "While you're off in Hyrule I'll take a lazy browse through the last 24 hours"


def turn_concern(cid="d8db9d7b1bba4cf7", opened=T0):
    return PrimeConcernV1(concern_id=cid, text="a thread I opened in a chat with Juniper", event_ref=f"turn:{CORR}",
                          opened_at=opened, last_evidence_at=opened)


def letter_concern(cid="f1e01e2a2cf54886"):
    return PrimeConcernV1(concern_id=cid, text="a thread from my letter", event_ref="orion_day_letter:2026-11-12#thread1",
                          opened_at=T0, last_evidence_at=T0)


def row(**kw):
    base = {"promised_at": datetime(2026, 11, 12, 7, 8, 40), "session_id": "9c91d646-cbd0-4977-8cdd-d2c82045c7f9",
            "promises_made": [{"offer": False, "quote": QUOTE}], "prompt": "have a look at the PRs", "response": "ok"}
    return {**base, **kw}


def checker(st, now, *, enabled=True):
    def check(move, state):
        refusal = it.take_on_refusal(move.concern_id, state, st, now, enabled=enabled)
        if refusal:
            return refusal
        c = next(c for c in state.concerns if c.concern_id == move.concern_id)
        return it.parse_not_before(move.not_before, now, it.horizon_for(c))[1]
    return check


# --- the adapter (S-1.2) and the loop guard ------------------------------------------------------------------------

def test_the_promise_adapter_returns_the_exact_quote_the_thread_and_a_72h_horizon():
    src = turn_promise.adapt(f"turn:{CORR}", row(), why="I said I would")
    assert isinstance(src, IntentionSourceV1)
    assert f'"{QUOTE}"' in src.prompt and "have a look at the PRs" in src.prompt and "I said I would" in src.prompt
    assert src.route == "chat_thread" and src.route_target == "9c91d646-cbd0-4977-8cdd-d2c82045c7f9"
    assert src.horizon_at == datetime(2026, 11, 15, 7, 8, 40, tzinfo=timezone.utc)   # naive Postgres time is UTC
    assert src.quote_ref == f"harness_turn_trace:{CORR}#promises_made[0]" and src.source == "turn_promise"


def test_the_promise_adapter_reads_json_text_skips_offers_and_points_at_the_right_index():
    raw = '[{"offer": true, "quote": "want me to look?"}, {"offer": false, "quote": "I will check the logs"}]'
    src = turn_promise.adapt(f"turn:{CORR}", row(promises_made=raw), why="w")
    assert "want me to look?" not in src.prompt and '"I will check the logs"' in src.prompt
    assert src.quote_ref.endswith("#promises_made[1]")


@pytest.mark.parametrize("r,words", [
    (None, "no turn trace"),
    (row(session_id=None), "no chat history row"),
    (row(session_id=INTENTION_SESSION_ID), "loop guard"),
    (row(session_id="orion_curiosity"), "loop guard"),
    (row(promises_made=[{"offer": True, "quote": "shall I?"}]), "records no promise"),
])
def test_the_adapter_refuses_with_a_plain_reason(r, words):
    out = turn_promise.adapt(f"turn:{CORR}", r, why="w")
    assert isinstance(out, str) and words in out


def test_fetch_reads_the_trace_joined_to_the_chat_row_by_correlation_id():
    import asyncio

    class Conn:
        async def fetchrow(self, sql, corr):
            self.args = (sql, corr)
            return {"session_id": "s"}

    conn = Conn()
    assert asyncio.run(turn_promise.fetch(conn, f"turn:{CORR}")) == {"session_id": "s"}
    assert conn.args[1] == CORR and "harness_turn_trace" in conn.args[0] and "chat_history_log" in conn.args[0]


# --- the move (S-1.1) ----------------------------------------------------------------------------------------------

def test_take_on_parses_and_passes_on_a_promise_concern():
    answer, err = parse_answer({"reading": "r", "move": {"kind": "take_on", "concern_id": "d8db9d7b1bba4cf7",
                                                         "why": "I said I would", "not_before": None}})
    assert err is None and isinstance(answer.move, TakeOnMoveV1)
    state = PrimeStateV1(concerns=[turn_concern()])
    now = T0 + timedelta(hours=2)
    applied = apply_answer(state, answer, now=now, arc_ids=[], focus_pending=False,
                           take_on_check=checker(it.PrimeIntentionsStateV1(), now))
    assert applied.move_refusal is None and applied.move.concern_id == "d8db9d7b1bba4cf7"


@pytest.mark.parametrize("cid,setup,words", [
    ("f1e01e2a2cf54886", None, "only promises"),                       # no adapter for a letter thread (yet)
    ("0000000000000000", None, "I hold no concern"),
    ("d8db9d7b1bba4cf7", "open", "already took that concern on"),
    ("d8db9d7b1bba4cf7", "done", "once"),
])
def test_take_on_is_refused_with_a_reason_prime_will_see(cid, setup, words):
    state = PrimeStateV1(concerns=[turn_concern(), letter_concern()])
    now = T0 + timedelta(hours=2)
    st = it.PrimeIntentionsStateV1()
    if setup:
        act = it.make_act("d8db9d7b1bba4cf7", source="turn_promise", horizon_sec=3600)
        st = it.take_on(st, turn_concern(), why="w", thought_id="t", act=act, horizon_sec=3600, not_before=None, now=now)
        if setup == "done":
            st = it.finish(st, st.open[0], outcome="not_carried", note="x", now=now)
    answer = PrimeAnswerV1.model_validate({"move": {"kind": "take_on", "concern_id": cid, "why": "w"}})
    applied = apply_answer(state, answer, now=now, arc_ids=[], focus_pending=False, take_on_check=checker(st, now))
    assert applied.move is None and words in applied.move_refusal


def test_take_on_off_is_refused_and_the_prompt_offers_nothing():
    state = PrimeStateV1(concerns=[turn_concern()])
    answer = PrimeAnswerV1.model_validate({"move": {"kind": "take_on", "concern_id": "d8db9d7b1bba4cf7", "why": "w"}})
    applied = apply_answer(state, answer, now=T0, arc_ids=[], focus_pending=False)
    assert applied.move is None and "switched off" in applied.move_refusal
    frame = build_frame(state, FrameInputs(now=T0))
    assert "take this on" not in frame and "WHAT YOU TOOK ON" not in frame


def test_the_frame_marks_only_adapter_concerns_and_lists_what_was_taken_on():
    state = PrimeStateV1(concerns=[turn_concern(), letter_concern()])
    frame = build_frame(state, FrameInputs(now=T0, take_on=True, intentions=["(nothing taken on yet)"]))
    marked = [line for line in frame.splitlines() if "you can take this on" in line]
    assert len(marked) == 1 and "d8db9d7b1bba4cf7" in marked[0]
    assert "WHAT YOU TOOK ON" in frame and "(nothing taken on yet)" in frame


@pytest.mark.parametrize("raw,expect_when,refused", [
    (None, None, False),
    ("2026-11-12T06:00:00Z", None, False),                  # already past: at once
    ("2026-11-13T20:00:00Z", datetime(2026, 11, 13, 20, tzinfo=timezone.utc), False),
    ("2026-11-13T20:00:00", datetime(2026, 11, 13, 20, tzinfo=timezone.utc), False),   # naive = UTC
    ("2026-11-15T06:00:00Z", None, True),                   # inside the last 2 h before the horizon
    ("tomorrow evening", None, True),
])
def test_not_before_is_parsed_leniently_and_never_lands_after_the_horizon(raw, expect_when, refused):
    when, why = it.parse_not_before(raw, T0 + timedelta(hours=1), turn_promise.horizon_from(T0))
    assert when == expect_when and (why is not None) == refused


# --- not yet, keep alive, the act ------------------------------------------------------------------------------------

def test_a_concern_never_taken_on_ages_out_at_its_horizon_and_others_stay():
    taken = turn_concern("aaaaaaaaaaaaaaaa")
    state = PrimeStateV1(concerns=[turn_concern(), letter_concern(), taken])
    act = it.make_act(taken.concern_id, source="turn_promise", horizon_sec=3600)
    st = it.take_on(it.PrimeIntentionsStateV1(), taken, why="w", thought_id="t", act=act, horizon_sec=3600,
                    not_before=None, now=T0)
    before = it.age_out(state, st, T0 + timedelta(hours=71))
    assert len(before.concerns) == 3
    after = it.age_out(state, st, T0 + timedelta(hours=72, minutes=1))
    assert {c.concern_id for c in after.concerns} == {"f1e01e2a2cf54886", "aaaaaaaaaaaaaaaa"}
    rec = after.concern_record[-1]
    assert rec.concern_id == "d8db9d7b1bba4cf7" and rec.end == "lapsed" and "never took it on" in rec.note


def test_a_concern_taken_on_does_not_fade_while_its_run_is_open():
    c = turn_concern()
    state = PrimeStateV1(concerns=[c])
    act = it.make_act(c.concern_id, source="turn_promise", horizon_sec=3600)
    st = it.take_on(it.PrimeIntentionsStateV1(), c, why="w", thought_id="t", act=act, horizon_sec=3600,
                    not_before=None, now=T0)
    later = T0 + timedelta(hours=30)
    kept = observe(it.keep_taken_on(state, st, later), PrimeSnapshotV1(), later)
    assert [x.concern_id for x in kept.concerns] == [c.concern_id]
    faded = observe(state, PrimeSnapshotV1(), later)          # without the take-on it fades at 24 h
    assert faded.concerns == []


def test_the_make_act_is_stable_and_names_its_source():
    a = it.make_act("d8db9d7b1bba4cf7", source="turn_promise", horizon_sec=7200)
    b = it.make_act("d8db9d7b1bba4cf7", source="turn_promise", horizon_sec=7200)
    assert a.act_id == b.act_id and a.actor == "prime" and a.verb == "make"
    assert a.object_ref == "intention:d8db9d7b1bba4cf7" and a.test_before == "intention_source:turn_promise"
    assert a.expect.signal_id == "artifact_produced"


@pytest.mark.parametrize("run,status", [
    ({"status": "running"}, None), ({"status": "completed"}, "completed"), ({"status": "failed", "error": "x"}, "failed"),
    (None, None)])
def test_run_terminal(run, status):
    assert it.run_terminal(run)[0] == status


# --- S-1.8, S-1.9 ----------------------------------------------------------------------------------------------------

def test_no_promise_quote_ever_enters_prime_state_or_the_motor_prefix_block():
    c = turn_concern()
    state = PrimeStateV1(concerns=[c], last_thought_at=T0)
    act = it.make_act(c.concern_id, source="turn_promise", horizon_sec=3600)
    st = it.take_on(it.PrimeIntentionsStateV1(), c, why="I said I would", thought_id="t", act=act, horizon_sec=3600,
                    not_before=None, now=T0)
    assert QUOTE not in st.model_dump_json() and "promises_made" not in st.model_dump_json()
    block = render_prime_block(it.keep_taken_on(state, st, T0), ticked_at=T0, now=T0)
    assert QUOTE not in block.text and "Hyrule" not in block.text


def test_a_second_source_plugs_in_with_one_adapter_and_one_map_entry(monkeypatch):
    letter = SimpleNamespace(PREFIX="orion_day_letter:", SOURCE="letter_thread", HORIZON_SEC=144 * 3600.0,
                             horizon_from=lambda opened: opened + timedelta(hours=144))
    monkeypatch.setitem(it.ADAPTERS, letter.PREFIX, letter)
    state = PrimeStateV1(concerns=[letter_concern()])
    assert it.take_on_refusal("f1e01e2a2cf54886", state, it.PrimeIntentionsStateV1(), T0, enabled=True) is None
    frame = build_frame(state, FrameInputs(now=T0, take_on=True))
    assert "you can take this on" in frame


def test_the_live_concern_replay_eval_passes_on_its_fixture():
    out = replay.run(replay.load_fixture())
    assert out["ok"], out["checks"]


def test_the_frame_does_not_mark_a_concern_already_taken_on():
    state = PrimeStateV1(concerns=[turn_concern()])
    frame = build_frame(state, FrameInputs(now=T0, take_on=True, no_take_on={"d8db9d7b1bba4cf7"}))
    assert "you can take this on" not in frame


def test_age_out_names_how_a_taken_on_concern_ended():
    c = turn_concern()
    act = it.make_act(c.concern_id, source="turn_promise", horizon_sec=3600)
    st = it.take_on(it.PrimeIntentionsStateV1(), c, why="w", thought_id="t", act=act, horizon_sec=3600,
                    not_before=None, now=T0)
    st = it.finish(st, st.open[0], outcome="not_carried", note="run failed", now=T0)
    aged = it.age_out(PrimeStateV1(concerns=[c]), st, T0 + timedelta(hours=73))
    assert aged.concerns == [] and "ended not_carried" in aged.concern_record[-1].note
