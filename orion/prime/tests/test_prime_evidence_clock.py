"""The concern evidence clock (design docs/superpowers/specs/2026-11-17-prime-concerns-decided-by-attention-design.md,
patch 1, #3574): a bare keep no longer renews a concern; only real evidence (``touch_concern``) does. Pure parts."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.prime import hunger as hg
from orion.prime.day_letter import thread_concern_id, thread_concerns
from orion.prime.state import (CONCERN_TTL, PROMISE_CONCERN_TTL, PrimeConcernV1, PrimeSnapshotV1, PrimeStateV1,
                               observe, touch_concern)
from orion.prime.thought import FrameInputs, PrimeAnswerV1, apply_answer, build_frame, system_prompt

T0 = datetime(2026, 11, 14, 10, 6, tzinfo=timezone.utc)
KEEP = PrimeAnswerV1.model_validate({"reading": "nothing new", "concerns": {"keep": ["c1", "p1"]}})


def _concern(cid="c1", ref=None, at=T0):
    return PrimeConcernV1(concern_id=cid, text=f"concern {cid}", event_ref=ref, opened_at=at, last_evidence_at=at)


def _state():
    return PrimeStateV1(concerns=[_concern(), _concern("p1", ref="turn:abc")])


def test_a_keep_is_recorded_but_does_not_move_the_clock():
    later = T0 + timedelta(hours=20)
    applied = apply_answer(_state(), KEEP, now=later, arc_ids=[], focus_pending=False, evidence_clock=True)
    assert applied.kept == ["c1", "p1"]
    assert [c.last_evidence_at for c in applied.state.concerns] == [T0, T0]


def test_with_the_clock_off_a_keep_moves_it_as_before():
    later = T0 + timedelta(hours=20)
    applied = apply_answer(_state(), KEEP, now=later, arc_ids=[], focus_pending=False)
    assert [c.last_evidence_at for c in applied.state.concerns] == [later, later]


def test_a_concern_kept_on_every_thought_still_fades_24h_after_its_last_evidence():
    """The jam: hourly keeps for a day used to hold it forever (0 of 54 concern acts faded, 10-30..11-17)."""
    state = PrimeStateV1(concerns=[_concern()])
    for h in range(1, 25):
        state = apply_answer(state, KEEP, now=T0 + timedelta(hours=h), arc_ids=[], focus_pending=False,
                             evidence_clock=True).state
        state = observe(state, PrimeSnapshotV1(), T0 + timedelta(hours=h))
    assert [c.concern_id for c in state.concerns] == ["c1"]  # exactly 24 h: not yet
    gone = observe(state, PrimeSnapshotV1(), T0 + CONCERN_TTL + timedelta(seconds=1))
    assert gone.concerns == [] and gone.concern_record[-1].end == "faded"
    assert gone.concern_record[-1].note == "faded: nothing renewed it for a day"


def test_a_promise_still_lasts_72h_from_its_last_evidence():
    state = PrimeStateV1(concerns=[_concern("p1", ref="turn:abc")])
    state = apply_answer(state, KEEP, now=T0 + timedelta(hours=70), arc_ids=[], focus_pending=False,
                         evidence_clock=True).state
    assert observe(state, PrimeSnapshotV1(), T0 + PROMISE_CONCERN_TTL).concerns == state.concerns
    touched = touch_concern(state, "p1", T0 + timedelta(hours=60))
    assert observe(touched, PrimeSnapshotV1(), T0 + PROMISE_CONCERN_TTL + timedelta(hours=1)).concerns
    assert observe(touched, PrimeSnapshotV1(), T0 + timedelta(hours=132, seconds=1)).concerns == []
    assert observe(state, PrimeSnapshotV1(), T0 + PROMISE_CONCERN_TTL + timedelta(seconds=1)).concerns == []


def test_touch_moves_only_the_named_concern_and_never_backwards():
    state = _state()
    later = T0 + timedelta(hours=5)
    out = touch_concern(state, "c1", later)
    assert [c.last_evidence_at for c in out.concerns] == [later, T0]
    assert touch_concern(out, "c1", T0) is out              # never backwards
    assert touch_concern(state, "ghost", later) is state     # unknown id: nothing
    assert touch_concern(state, None, later) is state


def test_a_letter_thread_is_touched_by_its_stable_id_not_its_cut_text():
    """Orion's review of #3610 (3): a thread's text is cut at 240/300 chars; its id is a hash of the stable
    ``orion_day_letter:<day>#thread<i>`` event_ref, so evidence keyed on the id survives the cut."""
    day = datetime(2026, 11, 13).date()
    long = "The walkway render break is unresolved and still live each morning. " * 10
    state, opened, _ = thread_concerns(PrimeStateV1(), day, [long], T0)
    [cid] = opened
    assert cid == thread_concern_id(day, 0)
    assert state.concerns[0].event_ref == "orion_day_letter:2026-11-13#thread0"
    assert state.concerns[0].text != f"a thread from my letter for 2026-11-13: {long}"  # cut
    later = T0 + timedelta(hours=3)
    assert touch_concern(state, cid, later).concerns[0].last_evidence_at == later


def test_the_model_is_told_a_keep_does_not_renew():
    on = system_prompt(None, evidence_clock=True)
    off = system_prompt(None)
    assert "Keeping a concern does not renew it" in on and "A concern you do not keep fades" not in on
    assert "A concern you do not keep fades after a day" in off and "Keeping a concern does not renew" not in off
    # The event-mode swap still applies on top.
    assert "Keeping a concern does not renew it" in system_prompt(None, events=True, evidence_clock=True)


def test_the_frame_line_says_last_touched():
    now = T0 + timedelta(hours=3)
    on = build_frame(_state(), FrameInputs(now=now, evidence_clock=True))
    off = build_frame(_state(), FrameInputs(now=now))
    assert "concern_id c1:" in on and ", last touched " in on and ", last kept " not in on
    assert ", last kept " in off


def test_a_horizon_need_says_a_keep_will_not_save_it():
    now = T0 + timedelta(hours=20)
    [(name, _, content, ref, _)] = hg.horizon_needs(PrimeStateV1(concerns=[_concern()]), now, evidence_clock=True)
    assert name == "horizon" and ref == "prime_concern:c1" and "keeping it does not renew it" in content
    [(_, _, old, _, _)] = hg.horizon_needs(PrimeStateV1(concerns=[_concern()]), now)
    assert "fades unless I keep it" in old


def test_a_keep_only_thought_halves_the_horizon_need_once_keeps_answer_nothing():
    """Guard 1 as the runtime calls it with the clock on: kept ids are left out of ``answered_refs``."""
    now = T0 + timedelta(hours=20)
    state = PrimeStateV1(concerns=[_concern()])
    needs = hg.compute_needs(state, now, None, evidence_clock=True)
    assert [n.name for n in needs] == ["horizon"]
    touched = hg.answered_refs(kept=(), dropped=[], cited=[])
    damped = hg.after_answer(state, needs, now, acted=False, touched=touched)
    assert damped.needs.damp["horizon"].factor < 1.0
    # With the clock off the keep answers it, as before.
    kept = hg.after_answer(state, needs, now, acted=False, touched=hg.answered_refs(kept=["c1"]))
    assert "horizon" not in kept.needs.damp


def test_the_replay_eval_passes_on_the_live_window():
    """Acceptance check 1 (11-14..11-17): clock off reproduces the live jam; clock on fades the 11-13 threads and
    the next letters get room."""
    from orion.prime.evals import run_concern_evidence_clock_replay as ev

    assert ev.main() == 0


def test_an_untouched_promise_frees_the_turn_slot_after_24h_while_still_held():
    from orion.prime import handback as hb

    promise = _concern("p1", ref="turn:abc")
    state = PrimeStateV1(concerns=[promise])
    assert hb.holds_turn_slot(promise, T0 + timedelta(hours=23))
    later = T0 + timedelta(hours=25)
    assert not hb.holds_turn_slot(promise, later)
    assert observe(state, PrimeSnapshotV1(), later).concerns == [promise]  # still held (72 h)
