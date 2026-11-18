"""Day-dream Patch 2b: the nightly plan applied to Prime's concerns (orion/prime/day_dream.py). Pure; no I/O."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from orion.prime import day_dream as dd
from orion.prime.state import MAX_CONCERNS, PrimeConcernV1, PrimeStateV1, touch_concern
from orion.schemas.dream_day import DreamDayPlanItemV1

NOW = datetime(2026, 11, 18, 8, 30, tzinfo=timezone.utc)
DAY = date(2026, 11, 17)
OLD = NOW - timedelta(hours=10)


def _state(*ids: str) -> PrimeStateV1:
    return PrimeStateV1(concerns=[PrimeConcernV1(concern_id=i, text=f"concern {i}", opened_at=OLD,
                                                 last_evidence_at=OLD) for i in ids])


def _plan(*items: dict) -> list[DreamDayPlanItemV1]:
    return [DreamDayPlanItemV1.model_validate(i) for i in items]


def test_drop_closes_with_its_own_note_not_faded():
    st, done = dd.apply_plan(_state("a", "b"), DAY, _plan(
        {"action": "drop", "concern_id": "a", "reason": "  the  lane came back  "}), NOW, run_id="dream-day-2026-11-17")
    assert [c.concern_id for c in st.concerns] == ["b"]
    rec = st.concern_record[-1]
    assert rec.concern_id == "a" and rec.end == "dropped" and not rec.produced
    assert rec.note == "dropped: dream 2026-11-17: the lane came back"  # #3648 reads 'faded:' notes; this is not one
    assert done.dropped == ["a"] and done.run_id == "dream-day-2026-11-17"


def test_keep_renews_nothing():
    # #3574/#3610 patch 1: only a sent question or a take-on renews. A dream's keep is a model saying so.
    before = _state("a")
    st, done = dd.apply_plan(before, DAY, _plan({"action": "keep", "concern_id": "a", "reason": "still open"}), NOW)
    assert st.concerns[0].last_evidence_at == OLD
    assert done.kept == 1 and not done.dropped and not done.opened
    assert st.concern_record == before.concern_record


def test_new_opens_one_concern_like_a_letter_thread_and_replays_open_nothing_twice():
    plan = _plan({"action": "new", "gist": "check the lane's own table before I call it a blindness"},
                 {"action": "new", "gist": "a second new"})
    st, done = dd.apply_plan(_state("a"), DAY, plan, NOW)
    assert len(done.opened) == 1 and done.skipped == ["new: only one per night"]
    c = next(c for c in st.concerns if c.concern_id == done.opened[0])
    assert c.text.startswith("from my dream on 2026-11-17: check the lane")
    assert c.event_ref == "dream_day:2026-11-17#new0" and c.opened_at == NOW and c.act_opened_at is None
    again, done2 = dd.apply_plan(st, DAY, plan, NOW)
    assert len(again.concerns) == len(st.concerns) and done2.opened == []


def test_new_without_room_is_counted_never_displaces():
    full = _state(*[f"c{i}" for i in range(MAX_CONCERNS)])
    st, done = dd.apply_plan(full, DAY, _plan({"action": "new", "gist": "x"}), NOW)
    assert [c.concern_id for c in st.concerns] == [c.concern_id for c in full.concerns]
    assert done.skipped == ["new: no room"] and done.opened == []


def test_a_drop_frees_room_for_the_new_in_the_same_plan():
    full = _state(*[f"c{i}" for i in range(MAX_CONCERNS)])
    st, done = dd.apply_plan(full, DAY, _plan({"action": "drop", "concern_id": "c0", "reason": "done"},
                                              {"action": "new", "gist": "the next thing"}), NOW)
    assert done.dropped == ["c0"] and len(done.opened) == 1 and len(st.concerns) == MAX_CONCERNS


def test_drop_of_a_concern_no_longer_held_is_skipped():
    st, done = dd.apply_plan(_state("a"), DAY, _plan({"action": "drop", "concern_id": "gone", "reason": "r"}), NOW)
    assert [c.concern_id for c in st.concerns] == ["a"] and done.skipped == ["drop gone: not held"]


def test_unreadable_plan_items_are_left_out():
    assert [i.action for i in dd.parse_plan([{"action": "keep", "concern_id": "a"}, {"action": "explode"}, "x"])] == [
        "keep"]
    assert dd.parse_plan(None) == []


def test_applied_record_is_per_day_and_bounded():
    st = dd.PrimeDreamDayStateV1()
    first = DAY - timedelta(days=dd.MAX_APPLIED + 2)
    for i in range(dd.MAX_APPLIED + 3):  # one night after another, oldest first, as the loop records them
        st = dd.record(st, dd.PrimeDreamDayAppliedV1(day=first + timedelta(days=i), applied_at=NOW))
    assert len(st.applied) == dd.MAX_APPLIED and dd.is_applied(st, DAY) and not dd.is_applied(st, first)
    assert not dd.is_applied(st, DAY + timedelta(1))


def test_the_evidence_clock_is_still_the_only_clock_after_a_plan():
    # A kept concern still fades on its own clock; touch_concern is the one renewal and stays untouched by the plan.
    st, _ = dd.apply_plan(_state("a"), DAY, _plan({"action": "keep", "concern_id": "a"}), NOW)
    assert touch_concern(st, "a", NOW).concerns[0].last_evidence_at == NOW
