"""Prime's slots decided by the concern contest (design 2026-11-17 patch 2, #3574; #3656). Pure."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from orion.prime import day_letter as dl
from orion.prime import slots
from orion.prime.state import (MAX_CONCERNS, PrimeConcernV1, PrimeNominationV1, PrimeStateV1, track_record)
from orion.prime.thought import (FrameInputs, PrimeAnswerV1, apply_answer, build_frame, system_prompt)

NOW = datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc)
DAY = date(2026, 11, 23)


def _c(cid, quiet_h=1.0, *, event_ref=None, sal=None, graded=None):
    at = NOW - timedelta(hours=quiet_h)
    return PrimeConcernV1(concern_id=cid, text=f"concern {cid}", event_ref=event_ref, opened_at=at,
                          last_evidence_at=at, act_opened_at=at, evidence_salience=sal, graded=graded)


def _promise(cid, quiet_h=0.0):
    return _c(cid, quiet_h, event_ref=f"turn:{cid}").model_copy(update={"act_opened_at": None})


def test_the_letter_constants_are_the_day_letters():
    assert slots.LETTER_EVENT_PREFIX == dl.LETTER_EVENT_PREFIX
    assert slots.LETTER_GROUP_CAP == dl.LETTER_CONCERNS_MAX


def test_letter_threads_without_room_wait_instead_of_being_forgotten():
    # Live 11-23 07:33Z: "5 thread(s), 0 now held as concerns, 5 without room", and nothing stored them.
    full = PrimeStateV1(concerns=[_c(f"{i:016x}") for i in range(MAX_CONCERNS)])
    st, opened, waiting = dl.thread_concerns(full, DAY, ["a", "b", "c", "d", "e"], NOW)
    assert opened == [] and waiting == 5
    assert [w.concern_id for w in st.waiting] == [dl.thread_concern_id(DAY, i) for i in range(5)]
    # A replay queues nothing twice.
    again, _, waiting2 = dl.thread_concerns(st, DAY, ["a", "b", "c", "d", "e"], NOW)
    assert waiting2 == 0 and len(again.waiting) == 5


def test_waiting_concerns_take_free_slots_best_first_a_letter_at_most_two():
    st = PrimeStateV1(waiting=[_c(dl.thread_concern_id(DAY, i), 1.0, event_ref=f"orion_day_letter:{DAY}#thread{i}")
                               for i in range(4)])
    st, rep = slots.run_contest(st, NOW)
    assert [c.concern_id for c in st.concerns] == [dl.thread_concern_id(DAY, 0), dl.thread_concern_id(DAY, 1)]
    assert rep.seated == [c.concern_id for c in st.concerns] and len(st.waiting) == 2
    assert all(c.opened_at == NOW and c.act_opened_at for c in st.concerns)


def test_a_quiet_held_concern_is_displaced_with_a_close_note_and_a_young_one_is_not():
    st = PrimeStateV1(concerns=[_c("aaaaaaaaaaaaaaaa", 21), _c("bbbbbbbbbbbbbbbb", 5), _c("cccccccccccccccc", 3)],
                      waiting=[_c("dddddddddddddddd", 0.5)], last_thought_at=NOW - timedelta(minutes=5))
    # Prime has not thought since it went quiet (12 h after its evidence, 9 h ago): it is protected until it does.
    unseen, rep0 = slots.run_contest(st.model_copy(update={"last_thought_at": NOW - timedelta(hours=10)}), NOW)
    assert rep0.displaced == [] and len(unseen.waiting) == 1
    st, rep = slots.run_contest(st, NOW)
    assert rep.displaced == [("aaaaaaaaaaaaaaaa", "dddddddddddddddd")]
    assert {c.concern_id for c in st.concerns} == {"bbbbbbbbbbbbbbbb", "cccccccccccccccc", "dddddddddddddddd"}
    rec = st.concern_record[-1]
    assert rec.end == "displaced" and not rec.produced and not rec.published
    assert rec.note.startswith("displaced: by dddddddddddddddd; score unscored held vs unscored new; quiet 21.0 h")
    assert any(c.startswith("concern displaced") for c in st.pending_changes)
    assert "1 displaced" in track_record(st)
    # Nothing else is quiet for 12 h: a second waiting concern stays waiting.
    st2 = st.model_copy(update={"waiting": [_c("eeeeeeeeeeeeeeee", 0.1)]})
    st2, rep2 = slots.run_contest(st2, NOW)
    assert rep2.displaced == [] and [w.concern_id for w in st2.waiting] == ["eeeeeeeeeeeeeeee"]


def test_contest_off_means_no_displacement_but_free_slots_still_fill():
    st = PrimeStateV1(concerns=[_c("aaaaaaaaaaaaaaaa", 21)], waiting=[_c("dddddddddddddddd", 0.5)],
                      last_thought_at=NOW)
    st_off, rep = slots.run_contest(st, NOW, contest=False)
    assert rep.displaced == [] and len(st_off.concerns) == 2
    full = PrimeStateV1(concerns=[_c(f"{i:016x}", 21) for i in range(3)], waiting=[_c("dddddddddddddddd", 0.5)],
                        last_thought_at=NOW)
    full_off, rep2 = slots.run_contest(full, NOW, contest=False)
    assert rep2.displaced == [] and len(full_off.waiting) == 1


def test_an_open_question_protects_a_quiet_concern():
    st = PrimeStateV1(concerns=[_c(f"{i:016x}", 21) for i in range(3)], waiting=[_c("dddddddddddddddd", 0.5)],
                      last_thought_at=NOW, open_nominations=[PrimeNominationV1(act_id=f"act_{i}", thought_id="t", focus_id="f", question="q",
                                                          why="w", arm="treated", opened_at=NOW, concern_id=f"{i:016x}")
                                        for i in range(3)])
    st, rep = slots.run_contest(st, NOW)
    assert rep.displaced == []


def test_a_taken_on_concern_and_a_promise_are_never_displaced():
    st = PrimeStateV1(concerns=[_c("aaaaaaaaaaaaaaaa", 30), _promise("bbbbbbbbbbbbbbbb", 60), _c("cccccccccccccccc", 1)],
                      waiting=[_c("dddddddddddddddd", 0.5)], last_thought_at=NOW)
    st, rep = slots.run_contest(st, NOW, taken_on=frozenset({"aaaaaaaaaaaaaaaa"}))
    assert rep.displaced == []


def test_a_waiting_promise_opens_at_once_by_pushing_out_the_weakest_non_promise_no_quiet_rule():
    st = PrimeStateV1(concerns=[_c("aaaaaaaaaaaaaaaa", 2), _c("bbbbbbbbbbbbbbbb", 3), _promise("cccccccccccccccc", 1)],
                      waiting=[_promise("pppppppppppppppp")])
    st, rep = slots.run_contest(st, NOW)
    assert rep.displaced == [("bbbbbbbbbbbbbbbb", "pppppppppppppppp")] and "pppppppppppppppp" in rep.seated
    assert st.concern_record[-1].note.startswith("displaced: by pppppppppppppppp; score unscored held vs unscored new;"
                                                 " a new promise always gets a slot")


def test_a_promise_waits_while_the_loop_guard_holds_it_and_when_every_slot_is_a_promise():
    st = PrimeStateV1(concerns=[_c("aaaaaaaaaaaaaaaa", 20)], waiting=[_promise("pppppppppppppppp")])
    held_back, rep = slots.run_contest(st, NOW, promise_bound=lambda s, p: "already holding 1 concern opened by a turn")
    assert rep.seated == [] and len(held_back.waiting) == 1
    full = PrimeStateV1(concerns=[_promise(f"{i:016x}", 30) for i in range(3)], waiting=[_promise("pppppppppppppppp")])
    full, rep = slots.run_contest(full, NOW)
    assert rep.seated == [] and rep.displaced == [] and len(full.waiting) == 1


def test_a_waiting_concern_that_outlives_its_life_leaves_with_a_line_a_promise_wakes_prime():
    st = PrimeStateV1(concerns=[_c(f"{i:016x}", 1) for i in range(3)],
                      waiting=[_c("dddddddddddddddd", 25), _promise("pppppppppppppppp", 73)])
    st, rep = slots.run_contest(st, NOW, promise_bound=lambda s, p: "bound")
    assert set(rep.expired) == {"dddddddddddddddd", "pppppppppppppppp"} and st.waiting == []
    woke = [c for c in st.pending_changes if "waited for a slot" in c]
    assert len(woke) == 1 and "pppppppppppppppp" in woke[0]  # only the promise wakes Prime
    assert any("dddddddddddddddd" in o and "waited for a slot" in o for o in st.recent_outcomes)


def test_the_waiting_list_is_capped_and_a_promise_is_the_last_to_leave():
    st = PrimeStateV1(waiting=[_promise("pppppppppppppppp", 1)])
    for i in range(slots.WAITING_MAX + 2):
        st = slots.queue(st, _c(f"{i:016x}", 0.5))
    assert len(st.waiting) == slots.WAITING_MAX and st.waiting[0].concern_id == "pppppppppppppppp"


def test_a_concern_prime_names_with_every_slot_full_waits_and_carries_its_want_salience():
    st = PrimeStateV1(concerns=[_c(f"{i:016x}", 1) for i in range(3)])
    answer = PrimeAnswerV1.model_validate({"reading": "r", "concerns": {"add": [{"text": "x", "event_id": "ev:1"}]}})
    wants = {"ev:1": {"family": "hub.world_pulse_read", "salience": 0.85}}
    on = apply_answer(st, answer, now=NOW, arc_ids=[], focus_pending=False, event_ids=["ev:1"], event_wants=wants,
                      contest=True)
    assert on.added == [] and len(on.state.waiting) == 1 and on.state.waiting[0].evidence_salience == 0.85
    assert any("waits for a slot" in w for w in on.warnings)
    off = apply_answer(st, answer, now=NOW, arc_ids=[], focus_pending=False, event_ids=["ev:1"], event_wants=wants)
    assert off.state.waiting == [] and any("not added" in w for w in off.warnings)
    # A held event concern records its want's salience; a missing or bad one is unscored, never 0.
    room = apply_answer(PrimeStateV1(), answer, now=NOW, arc_ids=[], focus_pending=False, event_ids=["ev:1"],
                        event_wants={"ev:1": {"salience": True}}, contest=True)
    assert room.state.concerns[0].evidence_salience is None


def test_new_state_fields_are_left_out_while_empty_so_old_readers_still_load_it():
    out = PrimeStateV1(concerns=[_c("aaaaaaaaaaaaaaaa")]).model_dump(mode="json")
    assert "waiting" not in out and "evidence_salience" not in out["concerns"][0]


def test_the_frame_shows_each_standing_and_the_waiting_list():
    st = PrimeStateV1(concerns=[_c("aaaaaaaaaaaaaaaa", 13), _promise("bbbbbbbbbbbbbbbb", 2)],
                      waiting=[_c("dddddddddddddddd", 0.5)], last_thought_at=NOW - timedelta(minutes=5))
    inputs = FrameInputs(now=NOW, standings=slots.standing_lines(st, NOW), waiting=slots.waiting_lines(st, NOW))
    frame = build_frame(st, inputs)
    assert "contest: score unscored, quiet 13.0 h, can be displaced: quiet past half its life (12.0 h)" in frame
    assert "contest: score unscored, quiet 2.0 h, never displaced: a promise" in frame
    assert "WAITING FOR A SLOT" in frame and "- dddddddddddddddd: concern dddddddddddddddd (waiting;" in frame
    assert "WAITING FOR A SLOT" not in build_frame(st, FrameInputs(now=NOW))
    assert "decided by a contest" in system_prompt(None, contest=True)
    assert "decided by a contest" not in system_prompt(None)
