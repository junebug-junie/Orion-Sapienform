"""Orion Prime 2A pure logic: the tick, the thought's answer, the acts and how a question settles."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion.grammar.acts import act_closed_event, act_opened_event, stable_act_id
from orion.prime import acts as pa
from orion.prime.state import (
    CONCERN_TTL,
    PrimeConcernV1,
    PrimeNominationV1,
    PrimeSnapshotV1,
    PrimeStateV1,
    observe,
    wants_to_think,
)
from orion.prime.thought import FrameInputs, PrimeAnswerV1, apply_answer, build_frame, parse_answer, summary_note

T0 = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)
REFRACTORY = timedelta(seconds=120)


def _snap(**kw) -> PrimeSnapshotV1:
    base = dict(active_arc_id="arc1", active_arc_label="Bus synaptic prediction error", binding=[],
                temporal_self_ok=True, body_ok=True)
    return PrimeSnapshotV1(**{**base, **kw})


# --- the tick --------------------------------------------------------------------------------------


def test_first_tick_wakes_and_a_quiet_tick_after_a_thought_does_not():
    state = observe(PrimeStateV1(), _snap(), T0)
    assert any("first wake" in c for c in state.pending_changes) and wants_to_think(state, T0, refractory=REFRACTORY)
    thought = state.model_copy(update={"pending_changes": [], "last_thought_at": T0})
    again = observe(thought, _snap(), T0 + timedelta(seconds=15))
    assert again.pending_changes == [] and not wants_to_think(again, T0 + timedelta(minutes=5), refractory=REFRACTORY)


def test_attention_moving_is_a_change_and_is_evidence_for_its_concern():
    concern = PrimeConcernV1(concern_id="c1", text="the bus keeps surprising me", arc_id="arc2",
                             opened_at=T0, last_evidence_at=T0)
    state = PrimeStateV1(concerns=[concern], seen_active_arc_id="arc1", seen_binding=[],
                         last_thought_at=T0)
    later = T0 + timedelta(hours=3)
    state = observe(state, _snap(active_arc_id="arc2", active_arc_label="heat"), later)
    assert state.pending_changes == ["attention moved to 'heat' (arc arc2)"]
    assert state.concerns[0].last_evidence_at == later


def test_an_unknown_read_is_never_a_change():
    state = PrimeStateV1(seen_active_arc_id="arc1", seen_binding=[], last_thought_at=T0)
    state = observe(state, PrimeSnapshotV1(), T0 + timedelta(seconds=15))
    assert state.pending_changes == []


def test_a_concern_with_no_evidence_for_a_day_fades_and_says_so():
    concern = PrimeConcernV1(concern_id="c1", text="old worry", opened_at=T0, last_evidence_at=T0)
    state = PrimeStateV1(concerns=[concern], seen_active_arc_id="arc1", seen_binding=[],
                         last_thought_at=T0)
    state = observe(state, _snap(), T0 + CONCERN_TTL + timedelta(minutes=1))
    assert state.concerns == [] and "old worry" in state.pending_changes[-1]


def test_the_refractory_interval_holds_even_with_changes():
    state = PrimeStateV1(pending_changes=["x"], last_thought_at=T0)
    assert not wants_to_think(state, T0 + timedelta(seconds=60), refractory=REFRACTORY)
    assert wants_to_think(state, T0 + timedelta(seconds=121), refractory=REFRACTORY)


# --- the thought -----------------------------------------------------------------------------------


def test_the_frame_marks_every_failed_read_as_a_gap_never_calm():
    frame = build_frame(PrimeStateV1(pending_changes=["first wake"]),
                        FrameInputs(now=T0, body_error="URLError: refused", temporal_error="timeout",
                                    curiosity_error="HTTPError 500"))
    assert "(unavailable: URLError: refused)" in frame and "(unavailable: timeout)" in frame
    assert "(unavailable: HTTPError 500)" in frame and "- first wake" in frame


def test_a_held_back_question_is_shown_as_held_back():
    nom = PrimeNominationV1(act_id="a", thought_id="t", focus_id="f", question="why is the bus loud?", why="w",
                            arm="control", opened_at=T0)
    frame = build_frame(PrimeStateV1(open_nominations=[nom]), FrameInputs(now=T0 + timedelta(minutes=5)))
    assert "held back as a control, not sent" in frame


@pytest.mark.parametrize("parsed", [None, [], {"move": {"kind": "shout"}}, {"concerns": {"add": [{"text": ""}]}}])
def test_an_answer_that_does_not_parse_is_a_failed_thought(parsed):
    answer, err = parse_answer(parsed)
    assert answer is None and err


def test_apply_edits_concerns_and_never_trusts_an_id_the_frame_did_not_show():
    c1 = PrimeConcernV1(concern_id="c1", text="keep", opened_at=T0, last_evidence_at=T0)
    c2 = PrimeConcernV1(concern_id="c2", text="drop me", opened_at=T0, last_evidence_at=T0)
    state = PrimeStateV1(concerns=[c1, c2], pending_changes=["x"])
    answer = PrimeAnswerV1.model_validate({
        "reading": "quiet afternoon",
        "concerns": {"add": [{"text": "new one", "arc_id": "invented"}, {"text": "fits", "arc_id": "arc1"},
                             {"text": "one too many"}],
                     "drop": [{"concern_id": "c2", "why": "resolved"}, {"concern_id": "ghost"}]},
        "move": None,
    })
    applied = apply_answer(state, answer, now=T0, arc_ids=["arc1"], focus_pending=False)
    texts = [c.text for c in applied.state.concerns]
    assert texts == ["keep", "new one", "fits"]
    assert applied.state.concerns[1].arc_id is None and applied.state.concerns[2].arc_id == "arc1"
    assert applied.state.pending_changes == [] and applied.state.last_reading == "quiet afternoon"
    assert len(applied.warnings) == 3   # invented arc, unknown drop, a fourth concern
    note = summary_note(applied)
    assert "concerns +2 -1" in note and "c2 (resolved)" in note and "no move" in note


@pytest.mark.parametrize("open_nom, focus_pending, refusal", [
    (True, False, "still open"),
    (False, True, "already has a focus waiting"),
    (False, None, "could not read"),
])
def test_a_move_is_refused_by_the_runtime_not_the_model(open_nom, focus_pending, refusal):
    noms = [PrimeNominationV1(act_id="a", thought_id="t", focus_id="f", question="q", why="w", arm="treated",
                              opened_at=T0)] if open_nom else []
    answer = PrimeAnswerV1.model_validate({"move": {"kind": "curiosity_focus", "question": "q2", "why": "w2"}})
    applied = apply_answer(PrimeStateV1(open_nominations=noms), answer, now=T0, arc_ids=[], focus_pending=focus_pending)
    assert applied.move is None and refusal in applied.move_refusal


def test_a_move_passes_when_the_slot_is_empty():
    answer = PrimeAnswerV1.model_validate({"move": {"kind": "curiosity_focus", "question": "q2", "why": "w2",
                                                    "concern_id": "ghost"}})
    applied = apply_answer(PrimeStateV1(), answer, now=T0, arc_ids=[], focus_pending=False)
    assert applied.move is not None and applied.move.concern_id is None


# --- acts ------------------------------------------------------------------------------------------


def test_acts_are_valid_ledger_events_from_orion_mind():
    th = pa.thought_act("t1", llm_timeout_sec=60)
    assert th.actor == "prime" and th.verb == "infer" and th.expect.horizon_sec == 180
    assert th.act_id == stable_act_id("prime", "infer", "t1")
    fid = pa.prime_focus_id(question="q", why="w", thought_id="t1")
    nom = pa.nomination_act(focus_id=fid, arm="control", thought_act_id=th.act_id, concern_id="c1")
    assert nom.verb == "ask" and nom.test_before == "holdback:control" and nom.parent_act_id == th.act_id
    assert nom.object_ref == f"curiosity_focus:{fid}" and nom.goal_ref == "prime_concern:c1"
    assert nom.expect.horizon_sec == pa.NOMINATION_HORIZON_SEC
    opened = act_opened_event(nom, source_service="orion-mind", source_component=pa.NOMINATION_COMPONENT)
    closed = act_closed_event(nom, produced=True, source_service="orion-mind", source_component="x")
    assert opened.provenance.source_service == "orion-mind" and closed.act.outcome.value == 1.0


def test_the_holdback_draw_is_deterministic_and_near_its_fraction():
    ids = [pa.prime_focus_id(question=f"q{i}", why="w", thought_id="t") for i in range(2000)]
    arms = [pa.draw_arm(i, 0.2) for i in ids]
    assert arms == [pa.draw_arm(i, 0.2) for i in ids]
    assert 0.16 < arms.count("control") / len(arms) < 0.24
    assert pa.draw_arm(ids[0], 0.0) == "treated" and pa.draw_arm(ids[0], 1.0) == "control"


# --- settlement ------------------------------------------------------------------------------------


def _run(rid, started, kind, *, focus_id=None, line="investigate", disposition=None):
    run = {"run_id": rid, "line": line, "started_at_iso": started.isoformat(), "outcome_kind": kind}
    if focus_id:
        run["focus"] = {"focus_id": focus_id, "disposition": disposition}
    return run


def test_treated_settles_on_the_run_that_took_it_with_its_disposition():
    runs = [_run("r1", T0 + timedelta(hours=1), "finished"),
            _run("r2", T0 + timedelta(hours=2), "wrote_nothing", focus_id="F", disposition="set_aside:not now")]
    s = pa.settle_nomination(arm="treated", focus_id="F", opened_at=T0, queued=True, queue_error=None, runs=runs,
                             now=T0 + timedelta(hours=9))
    assert s.produced is False and s.run_id == "r2" and "set_aside:not now" in s.note


def test_control_settles_on_the_first_run_after_it_opened():
    runs = [_run("old", T0 - timedelta(hours=1), "wrote_nothing"), _run("r2", T0 + timedelta(hours=2), "wrote_nothing"),
            _run("r1", T0 + timedelta(hours=1), "reached_out_sent"),
            _run("self", T0 + timedelta(minutes=5), "finished", line="self_inquiry")]
    s = pa.settle_nomination(arm="control", focus_id="F", opened_at=T0, queued=None, queue_error=None, runs=runs,
                             now=T0 + timedelta(hours=9))
    assert s.produced is True and s.run_id == "r1"


def test_a_running_run_waits_and_a_refused_queue_closes_at_once():
    running = [_run("r1", T0 + timedelta(hours=1), "running", focus_id="F")]
    assert pa.settle_nomination(arm="treated", focus_id="F", opened_at=T0, queued=True, queue_error=None,
                                runs=running, now=T0 + timedelta(hours=2)) is None
    s = pa.settle_nomination(arm="treated", focus_id="F", opened_at=T0, queued=False, queue_error="slot taken",
                             runs=[], now=T0)
    assert s.produced is False and "slot taken" in s.note


def test_a_treated_focus_no_run_took_closes_after_the_focus_ttl():
    kw = dict(arm="treated", focus_id="F", opened_at=T0, queued=True, queue_error=None, runs=[])
    assert pa.settle_nomination(now=T0 + timedelta(hours=40), **kw) is None
    s = pa.settle_nomination(now=T0 + timedelta(hours=50), **kw)
    assert s.produced is False and "expired" in s.note


def test_only_a_change_in_what_binds_is_body_news():
    state = PrimeStateV1(seen_active_arc_id="arc1", seen_binding=[], last_thought_at=T0)
    assert observe(state, _snap(binding=[]), T0).pending_changes == []
    hot = observe(state, _snap(binding=["heat"]), T0)
    assert hot.pending_changes == ["body: binding constraints now heat"] and hot.seen_binding == ["heat"]
    assert observe(hot, _snap(binding=None, body_ok=False), T0).pending_changes == hot.pending_changes


def test_a_move_can_cite_the_concern_added_in_the_same_answer():
    answer = PrimeAnswerV1.model_validate({"concerns": {"add": [{"text": "energy over the envelope"}]},
                                           "move": {"kind": "curiosity_focus", "question": "q", "why": "w",
                                                    "concern_id": "new"}})
    applied = apply_answer(PrimeStateV1(), answer, now=T0, arc_ids=[], focus_pending=False)
    assert applied.move.concern_id == applied.state.concerns[0].concern_id == applied.added[0]
    lone = PrimeAnswerV1.model_validate({"move": {"kind": "curiosity_focus", "question": "q", "why": "w",
                                                  "concern_id": "new"}})
    applied = apply_answer(PrimeStateV1(), lone, now=T0, arc_ids=[], focus_pending=False)
    assert applied.move.concern_id is None and any('"new"' in w for w in applied.warnings)
