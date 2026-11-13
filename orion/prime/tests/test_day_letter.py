"""The day letter as Prime's end-of-day turn (living Prime Part 3): the pure parts of orion/prime/day_letter.py."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from orion.prime import day_letter as dl
from orion.prime.state import MAX_CONCERNS, PrimeConcernV1, PrimeStateV1

NOW = datetime(2026, 11, 13, 7, 30, tzinfo=timezone.utc)  # 00:30 in Denver on 11-13
DAY = date(2026, 11, 12)


def test_yesterday_is_the_local_day_before_not_the_utc_one():
    assert dl.yesterday(NOW) == DAY
    assert dl.yesterday(datetime(2026, 11, 13, 6, 59, tzinfo=timezone.utc)) == date(2026, 11, 11)  # 23:59 Denver 11-12


def test_the_make_act_is_prime_s_existing_make_verb_on_the_day():
    act = dl.make_act(DAY, horizon_sec=3600.0)
    assert (act.actor, act.verb, act.object_ref) == ("prime", "make", "orion_day_letter:2026-11-12")
    assert act.act_id == dl.make_act(DAY, horizon_sec=99.0).act_id  # stable: open and close are one act
    assert act.expect.signal_id == "artifact_produced"


def test_ids_are_derived_from_the_day():
    assert dl.letter_run_id(DAY) == "orion-day-2026-11-12-1"
    assert dl.letter_correlation_id(DAY) == dl.letter_correlation_id(DAY) != dl.letter_correlation_id(DAY + timedelta(1))
    assert dl.thread_concern_id(DAY, 0) != dl.thread_concern_id(DAY, 1)


def test_attempt_ids_keep_attempt_1_and_differ_per_attempt():
    assert dl.letter_run_id(DAY, 2) == "orion-day-2026-11-12-2"
    assert dl.letter_correlation_id(DAY, 1) == dl.letter_correlation_id(DAY)  # the pre-retry id is unchanged
    assert len({dl.letter_correlation_id(DAY, n) for n in (1, 2, 3)}) == 3


def _writing(attempt=1, deadline_h=12):
    op = dl.start(dl.PrimeDayLetterStateV1(), DAY, NOW).open
    return op.model_copy(update={"act_id": "a", "attempt": attempt, "run_id": dl.letter_run_id(DAY, attempt),
                                 "correlation_id": dl.letter_correlation_id(DAY, attempt), "order_status": "submitted",
                                 "deadline_at": NOW + timedelta(hours=deadline_h)})


def test_a_retry_is_allowed_while_attempts_and_the_deadline_last():
    assert dl.can_retry(_writing(1), NOW, max_attempts=3)
    assert not dl.can_retry(_writing(3), NOW, max_attempts=3)
    assert not dl.can_retry(_writing(1), NOW + timedelta(hours=12), max_attempts=3)


def test_next_attempt_is_a_fresh_order_for_the_same_act():
    op = _writing(1).model_copy(update={"missed_sent": [DAY - timedelta(1)], "order_detail": "x"})
    nxt = dl.next_attempt(op, "the run ended failed: E2BIG")
    assert (nxt.attempt, nxt.run_id, nxt.act_id, nxt.deadline_at) == (2, "orion-day-2026-11-12-2", "a", op.deadline_at)
    assert nxt.correlation_id == dl.letter_correlation_id(DAY, 2)
    assert nxt.order_status is None and nxt.order_detail is None and nxt.missed_sent == []
    assert nxt.failed_attempts == ["attempt 1: the run ended failed: E2BIG"]


def test_a_record_the_bidding_image_saved_still_loads():
    raw = {"open": {"letter_date": "2026-11-12", "seen_at": NOW.isoformat(), "bid_until": NOW.isoformat(), "bids": 3,
                    "next_bid_at": NOW.isoformat(), "last_refusal": "deferred", "want_id": "w", "act_id": "a",
                    "run_id": "orion-day-2026-11-12-1"},
           "done": [{"letter_date": "2026-11-11", "outcome": "not_written", "closed_at": NOW.isoformat()}]}
    st = dl.PrimeDayLetterStateV1.model_validate(raw)
    assert (st.open.act_id, st.open.attempt, st.done[0].outcome) == ("a", 1, "not_written")
    import pytest
    with pytest.raises(Exception):
        dl.PrimeDayLetterOpenV1.model_validate({"letter_date": "2026-11-12", "seen_at": NOW.isoformat(), "typo": 1})


def test_run_outcome_reads_terminal_states_only():
    assert dl.run_outcome({"status": "running"})[0] is None
    ok = dl.run_outcome({"status": "completed", "orion_day": {"persist_outcome": "written", "threads": ["t"]}})
    assert ok[0] is True and ok[2]["threads"] == ["t"]
    assert dl.run_outcome({"status": "completed", "orion_day": {"persist_outcome": "already_written"}})[0] is False
    failed = dl.run_outcome({"status": "failed", "error": "workflow_deadline"})
    assert failed[0] is False and "workflow_deadline" in failed[1]


def test_missed_days_are_listed_until_told_then_never_again():
    st = dl.PrimeDayLetterStateV1()
    st = dl.finish(st, dl.PrimeDayLetterDoneV1(letter_date=DAY - timedelta(1), outcome="not_written",
                                               reason="never funded", closed_at=NOW))
    st = dl.finish(st, dl.PrimeDayLetterDoneV1(letter_date=DAY - timedelta(2), outcome="written", closed_at=NOW))
    missed = dl.missed_for(st, DAY)
    assert [(m.letter_date, m.reason) for m in missed] == [(DAY - timedelta(1), "never funded")]
    assert dl.missed_for(dl.mark_told(st, [DAY - timedelta(1)]), DAY) == []
    assert dl.missed_for(st, DAY - timedelta(1)) == []  # only days before the letter's own


def test_thread_concerns_respect_the_letter_cap_and_max_concerns_and_replay_opens_nothing_twice():
    state = PrimeStateV1()
    state, opened, no_room = dl.thread_concerns(state, DAY, ["one", "", "two", "three"], NOW)
    assert len(opened) == dl.LETTER_CONCERNS_MAX and no_room == 1
    assert [c.text for c in state.concerns] == ["a thread from my letter for 2026-11-12: one",
                                                "a thread from my letter for 2026-11-12: two"]
    assert state.concerns[0].event_ref == "orion_day_letter:2026-11-12#thread0"
    assert any("named 3 thread(s)" in p for p in state.pending_changes)  # wakes a thought: Prime may ask
    again, opened2, _ = dl.thread_concerns(state, DAY, ["one", "", "two", "three"], NOW)
    assert opened2 == [] or set(opened2).isdisjoint(opened)
    assert len([c for c in again.concerns if c.text.endswith(": one")]) == 1
    full = PrimeStateV1(concerns=[PrimeConcernV1(concern_id=f"{i:016x}", text="x", opened_at=NOW, last_evidence_at=NOW)
                                  for i in range(MAX_CONCERNS)])
    _, opened3, no_room3 = dl.thread_concerns(full, DAY, ["one"], NOW)
    assert opened3 == [] and no_room3 == 1


def test_state_round_trips_and_keeps_only_recent_days():
    st = dl.PrimeDayLetterStateV1()
    for i in range(dl.MAX_DONE + 3):
        st = dl.finish(st, dl.PrimeDayLetterDoneV1(letter_date=DAY - timedelta(i), outcome="written", closed_at=NOW))
    assert len(st.done) == dl.MAX_DONE
    assert dl.PrimeDayLetterStateV1.model_validate_json(st.model_dump_json()) == st
    assert dl.handled(st, DAY) and not dl.handled(st, DAY + timedelta(1))
