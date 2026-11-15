"""Living Prime Part 2, the turn hand-back (orion/prime/handback.py): pure fold, no bus, no model."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from orion.prime import handback as hb
from orion.prime.state import (CONCERN_TTL, MAX_CONCERNS, PROMISE_CONCERN_TTL, PrimeConcernV1, PrimeSnapshotV1,
                               PrimeStateV1, concern_ttl, is_promise_concern, observe)
from orion.prime.thought import FrameInputs, build_frame
from orion.schemas.cockpit_sighting import CockpitHopV1

T0 = datetime(2026, 11, 12, 12, 0, tzinfo=timezone.utc)
FIXTURE = Path(__file__).parent / "fixtures" / "turn_hops_shaped_like_live.json"


def _ingress(corr, session, prompt="hello", *, mode=None, at=T0):
    raw = {"session_id": session, "user_message": prompt, "attachment_count": 0, "observation_published": True}
    if mode is not None:
        raw["mode"] = mode
    return CockpitHopV1(correlation_id=corr, seq=0, ts=at, stage="ingress", visor_line="ingress", status="ok",
                        summary={"session_id": session}, raw=raw, producer="orion-hub").model_dump(mode="json")


def _finalize(corr, final="Done.", *, verdict="completed", promises=(), kept=(), at=T0 + timedelta(minutes=2)):
    raw = {"correlation_id": corr, "final_text": final, "compliance_verdict": verdict, "finalize_ran": True,
           "step_count": 3, "grounding_status": "ok",
           "reflection": {"promises_made": [dict(p) for p in promises], "promises_kept": list(kept)}}
    return CockpitHopV1(correlation_id=corr, seq=9, ts=at, stage="finalize", visor_line="finalize · completed",
                        status="ok", summary={"compliance_verdict": verdict}, raw=raw,
                        producer="orion-hub").model_dump(mode="json")


def _signals(*hops):
    return [s for s in (hb.signal_from_hop(h) for h in hops) if s is not None]


def _fold(state, turns, *hops, now=T0 + timedelta(minutes=3)):
    return hb.fold_turn_signals(state, turns, _signals(*hops), now)


def _concern(cid, *, text="watching the cabinet heat", event_ref=None):
    return PrimeConcernV1(concern_id=cid, text=text, event_ref=event_ref, opened_at=T0 - timedelta(hours=1),
                          last_evidence_at=T0 - timedelta(hours=1), act_opened_at=T0 - timedelta(hours=1))


# --- who ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("session,mode,who", [
    ("9c91d646-cbd0-4977-8cdd-d2c82045c7f9", "orion", "chat_juniper"),
    ("9c91d646-cbd0-4977-8cdd-d2c82045c7f9", None, None),  # Orion speaking first in her session: not her turn
    ("orion_curiosity", None, "curiosity"),
    ("consult-9f138a849520", None, "consult"),
    ("orion_day_letter", None, "letter"),
    ("orion_journal", "orion", None),
    ("orion_world_pulse_read", None, None),
    ("self-sense-eval", None, None),
    ("pr-field-notes", None, None),
    ("", "orion", None),
])
def test_who_is_read_from_the_ingress_session(session, mode, who):
    raw = {"mode": mode} if mode is not None else {}
    assert hb.classify_who(session, raw) == who


def test_consult_prefix_matches_the_consult_seam():
    from orion.consult.core import ORION_SESSION_PREFIX

    assert hb.CONSULT_SESSION_PREFIX == ORION_SESSION_PREFIX


def test_a_system_session_ingress_never_reaches_the_inbox():
    assert hb.signal_from_hop(_ingress("c1", "orion_world_pulse_read")) is None
    assert hb.signal_from_hop({"stage": "motor_hop", "correlation_id": "c1", "ts": T0.isoformat()}) is None


# --- recorded turns fold deterministically -------------------------------------------------------------


def test_recorded_turn_hops_fold_deterministically():
    hops = json.loads(FIXTURE.read_text())
    now = T0 + timedelta(hours=1)
    a = hb.fold_turn_signals(PrimeStateV1(), hb.PrimeTurnsStateV1(), _signals(*hops), now)
    b = hb.fold_turn_signals(PrimeStateV1(), hb.PrimeTurnsStateV1(), _signals(*hops), now)
    assert a == b
    assert [t.who for t in a[2]] == ["chat_juniper", "curiosity", "consult"]
    chat, cur, consult = a[2]
    assert chat.gist.startswith('Juniper: "can you look at why the walkway camera')
    assert cur.gist.startswith("The walkway gaps are the vision gate skipping")  # first string field of the JSON
    assert consult.promise_refs == []
    assert a[1].turns_total == 3 and a[1].pending == []
    # The world-pulse read in the recording was not handed back; nothing woke Prime (no thread, no answer).
    assert all(t.correlation_id != "wp-1" for t in a[1].recent)
    assert a[0].pending_changes == []


# --- thread opened -> concern citing the turn ------------------------------------------------------------


def test_a_promise_opens_a_concern_citing_the_turn_correlation_id():
    promise = {"quote": "I'll check the walkway logs tonight and tell you what I find", "offer": False}
    state, turns, folded = _fold(PrimeStateV1(), hb.PrimeTurnsStateV1(),
                                 _ingress("corr-7", "sess-1", "walkway?", mode="orion"),
                                 _finalize("corr-7", "I'll check the walkway logs tonight and tell you what I find.",
                                           promises=[promise]))
    (concern,) = state.concerns
    assert concern.event_ref == "turn:corr-7"
    assert "harness_turn_trace:corr-7#promises_made[0]" in concern.text
    assert folded[0].opened_concern_id == concern.concern_id
    assert folded[0].promise_refs == ["harness_turn_trace:corr-7#promises_made[0]"]
    assert any("made a promise, now concern" in c and "corr-7" in c for c in state.pending_changes)
    frame = build_frame(state, FrameInputs(now=T0, turns=hb.turn_lines(turns, T0 + timedelta(minutes=5))))
    assert "from turn corr-7" in frame and "TURNS THAT HANDED BACK TO YOU" in frame


def test_an_offer_is_not_a_thread():
    state, _, folded = _fold(PrimeStateV1(), hb.PrimeTurnsStateV1(), _ingress("c", "s", mode="orion"),
                             _finalize("c", "If you want, I'll dig into it.",
                                       promises=[{"quote": "If you want, I'll dig into it.", "offer": True}]))
    assert state.concerns == [] and folded[0].promise_refs == [] and state.pending_changes == []


def test_a_handed_back_turn_without_thread_or_answer_does_not_wake_prime():
    state, turns, folded = _fold(PrimeStateV1(), hb.PrimeTurnsStateV1(), _ingress("c", "orion_curiosity"),
                                 _finalize("c", "Found the gap."))
    assert state.pending_changes == [] and len(turns.recent) == 1 and folded[0].gist == "Found the gap."


# --- answered concern settles --------------------------------------------------------------------------


def test_a_turn_whose_prompt_cites_an_open_concern_and_replies_settles_it():
    cid = "a1b2c3d4e5f60718"
    state = PrimeStateV1(concerns=[_concern(cid)])
    state, _, folded = _fold(state, hb.PrimeTurnsStateV1(),
                             _ingress("c9", "consult-abc", f"Prime holds prime_concern:{cid}; what is the heat doing?"),
                             _finalize("c9", "The cabinet is at 27 C and falling."))
    assert state.concerns[0].graded == "settled"
    (rec,) = state.concern_record
    assert rec.end == "settled" and rec.produced and "c9" in rec.note
    assert folded[0].settled_concern_ids == [cid]
    assert any("answered my concern" in c for c in state.pending_changes)


def test_a_reply_that_only_mentions_a_concern_settles_nothing():
    cid = "a1b2c3d4e5f60718"
    state = PrimeStateV1(concerns=[_concern(cid)])
    state, _, _ = _fold(state, hb.PrimeTurnsStateV1(), _ingress("c9", "s", "what are you holding?", mode="orion"),
                        _finalize("c9", f"I am holding prime_concern:{cid} about the heat."))
    assert state.concerns[0].graded is None and state.concern_record == []


@pytest.mark.parametrize("verdict,final", [("failed", "partial text"), ("partial", "cut short"), ("completed", "   ")])
def test_a_turn_that_did_not_reply_settles_nothing(verdict, final):
    cid = "a1b2c3d4e5f60718"
    state = PrimeStateV1(concerns=[_concern(cid)])
    state, _, _ = _fold(state, hb.PrimeTurnsStateV1(), _ingress("c9", "consult-x", f"prime_concern:{cid}"),
                        _finalize("c9", final, verdict=verdict))
    assert state.concerns[0].graded is None


# --- loop guard ---------------------------------------------------------------------------------------------


def _promise_turn(corr):
    return (_ingress(corr, "s", "hi", mode="orion"),
            _finalize(corr, "I'll come back to this tomorrow morning.",
                      promises=[{"quote": "I'll come back to this tomorrow morning.", "offer": False}]))


def test_only_one_turn_concern_is_held_at_once_and_the_refusal_is_recorded():
    state, turns, _ = _fold(PrimeStateV1(), hb.PrimeTurnsStateV1(), *_promise_turn("t1"))
    state, turns, folded = _fold(state, turns, *_promise_turn("t2"))
    assert len(state.concerns) == 1 and folded[0].opened_concern_id is None
    assert "already holding 1 concern opened by a turn" in (folded[0].thread_refused or "")
    assert "no concern" in "\n".join(hb.turn_lines(turns, T0))


def test_turn_concerns_respect_max_concerns():
    state = PrimeStateV1(concerns=[_concern(f"{i:016x}") for i in range(MAX_CONCERNS)])
    state, _, folded = _fold(state, hb.PrimeTurnsStateV1(), *_promise_turn("t1"))
    assert len(state.concerns) == MAX_CONCERNS and "already holding 3 concerns" in folded[0].thread_refused


def test_at_most_four_turn_concerns_a_day():
    state, turns = PrimeStateV1(), hb.PrimeTurnsStateV1()
    opened = 0
    for i in range(6):
        state, turns, folded = _fold(state, turns, *_promise_turn(f"t{i}"), now=T0 + timedelta(minutes=10 * i + 3))
        opened += folded[0].opened_concern_id is not None
        # Prime drops the turn concern before the next turn (so only the per-day bound can refuse).
        state = state.model_copy(update={"concerns": []})
    assert opened == hb.TURN_CONCERNS_PER_DAY
    # A day later the window has moved on.
    state, turns, folded = _fold(state, turns, *_promise_turn("t9"), now=T0 + timedelta(hours=25))
    assert folded[0].opened_concern_id is not None


def test_a_turn_concern_is_a_promise_and_fades_after_72h_not_24h():
    state, _, _ = _fold(PrimeStateV1(), hb.PrimeTurnsStateV1(), *_promise_turn("t1"))
    assert is_promise_concern(state.concerns[0]) and concern_ttl(state.concerns[0]) == PROMISE_CONCERN_TTL
    snap = PrimeSnapshotV1(binding=[], body_ok=True)
    assert observe(state, snap, T0 + CONCERN_TTL + timedelta(minutes=10)).concerns == state.concerns
    assert observe(state, snap, T0 + timedelta(hours=48)).concerns == state.concerns
    later = observe(state, snap, T0 + PROMISE_CONCERN_TTL + timedelta(minutes=10))
    assert later.concerns == [] and later.concern_record[-1].end == "faded"
    assert later.concern_record[-1].note == "faded: not kept for 3 days"


# --- dedupe / replay safety -------------------------------------------------------------------------------


def test_replaying_the_same_hops_changes_nothing():
    hops = _promise_turn("t1")
    state, turns, _ = _fold(PrimeStateV1(), hb.PrimeTurnsStateV1(), *hops)
    again, turns2, folded = _fold(state, turns, *hops)
    assert folded == [] and again == state and turns2.recent == turns.recent and turns2.pending == []


def test_a_duplicate_finalize_in_one_batch_folds_once():
    ing, fin = _promise_turn("t1")
    state, turns, folded = _fold(PrimeStateV1(), hb.PrimeTurnsStateV1(), ing, fin, fin)
    assert len(folded) == 1 and len(state.concerns) == 1


def test_a_finalize_without_ingress_is_ignored_and_stale_pending_expires():
    state, turns, folded = _fold(PrimeStateV1(), hb.PrimeTurnsStateV1(), _finalize("orphan"))
    assert folded == [] and turns.recent == []
    _, turns, _ = _fold(PrimeStateV1(), hb.PrimeTurnsStateV1(), _ingress("slow", "orion_curiosity"))
    assert [p.correlation_id for p in turns.pending] == ["slow"]
    _, turns, _ = hb.fold_turn_signals(PrimeStateV1(), turns, [], T0 + hb.PENDING_TTL + timedelta(minutes=1))
    assert turns.pending == []


def test_link_started_turn_marks_the_act():
    _, turns, _ = _fold(PrimeStateV1(), hb.PrimeTurnsStateV1(), _ingress("c", "orion_curiosity"), _finalize("c"))
    linked = hb.link_started_turn(turns, "c", "act_123")
    assert linked.recent[0].started_by_act == "act_123"
    assert hb.link_started_turn(turns, None, "act_123") == turns


def test_turn_state_round_trips_through_json():
    _, turns, _ = _fold(PrimeStateV1(), hb.PrimeTurnsStateV1(), *_promise_turn("t1"))
    assert hb.PrimeTurnsStateV1.model_validate_json(turns.model_dump_json()) == turns


def test_frame_without_handback_is_unchanged():
    frame = build_frame(PrimeStateV1(), FrameInputs(now=T0))
    assert "TURNS THAT HANDED BACK" not in frame


def test_gist_drops_markdown_bold_and_reads_json_replies():
    assert hb.turn_gist("curiosity", "", "**Answer recorded** for run 09b7") == "Answer recorded for run 09b7"
    assert hb.turn_gist("curiosity", "", '```json\n{"what_i_learned": "gaps are the gate"}\n```') == "gaps are the gate"
    assert hb.turn_gist("consult", "", "") == "(no reply text)"


def test_replay_eval_checks_pass_on_the_recorded_hops():
    from orion.prime.evals import run_turn_handback_replay as ev

    hops = json.loads(FIXTURE.read_text())
    result, report = ev.checks(hops)
    assert all(result.values()), result
    assert report["turns_by_who"] == {"chat_juniper": 1, "curiosity": 1, "consult": 1} and report["missing"] == []


def test_a_promise_after_an_offer_links_its_own_index():
    offer = {"quote": "If you want, I'll pull the logs.", "offer": True}
    promise = {"quote": "I'll check the gate thresholds tonight.", "offer": False}
    _, _, folded = _fold(PrimeStateV1(), hb.PrimeTurnsStateV1(), _ingress("c", "s", mode="orion"),
                         _finalize("c", "x", promises=[offer, promise]))
    assert folded[0].promise_refs == ["harness_turn_trace:c#promises_made[1]"]


def test_a_turn_concern_id_is_derived_from_the_turn_and_marked_no_bet():
    state, turns, _ = _fold(PrimeStateV1(), hb.PrimeTurnsStateV1(), *_promise_turn("t1"))
    assert state.concerns[0].concern_id == hb.turn_concern_id("t1")
    assert turns.no_bet_concern_ids == [hb.turn_concern_id("t1")]


def test_a_run_that_settles_before_its_turn_hands_back_is_linked_when_it_does():
    turns = hb.link_started_turn(hb.PrimeTurnsStateV1(), "late", "act_9")
    assert turns.pending_links == {"late": "act_9"}
    _, turns, folded = _fold(PrimeStateV1(), turns, _ingress("late", "orion_curiosity"), _finalize("late"))
    assert folded[0].started_by_act == "act_9" and turns.pending_links == {}


# --- intentions Prime takes on (choice-as-EFE Step -1) --------------------------------------------------------------


def test_the_carry_session_is_handed_back_as_intention_and_pinned_to_the_contract():
    from orion.schemas.intention import INTENTION_SESSION_ID

    assert hb.INTENTION_SESSION_ID == INTENTION_SESSION_ID
    assert hb.classify_who(INTENTION_SESSION_ID, {}) == "intention"


def test_a_carry_turn_settles_its_concern_but_its_own_promises_open_no_concern():
    # S-1.5: the carry turn's prompt cites prime_concern:<id> -> settled on completion.
    # S-1.7 (loop guard): a promise made inside the carry turn opens no new concern.
    state = PrimeStateV1(concerns=[_concern("d8db9d7b1bba4cf7", event_ref="turn:a129b72f")])
    corr = "intention-carry-turn-1"
    state, turns, [turn] = _fold(
        state, hb.PrimeTurnsStateV1(),
        _ingress(corr, "orion_intention", "prime_concern:d8db9d7b1bba4cf7\n\nI am carrying out something I told Juniper"),
        _finalize(corr, "I browsed the last day's PRs. Next I'll write up the two that matter.",
                  promises=[{"offer": False, "quote": "Next I'll write up the two that matter"}]))
    assert turn.who == "intention" and turn.settled_concern_ids == ["d8db9d7b1bba4cf7"]
    assert turn.opened_concern_id is None and "loop guard" in turn.thread_refused
    assert [c.concern_id for c in state.concerns] == ["d8db9d7b1bba4cf7"] and state.concerns[0].graded == "settled"
    assert turns.turn_concerns_opened_at == [] and turns.no_bet_concern_ids == []
