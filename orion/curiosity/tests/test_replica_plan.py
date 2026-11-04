"""The curiosity replica's pure parts (design 2026-11-04-curiosity-replica-design.md, sections 2-6):
the candidate list, the plan check, the plan row, the frame, the answer parse and how a plan is graded."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion.curiosity.replica.candidates import (
    NATS_N,
    UNCERTAIN_N,
    build_candidates,
    last_moves,
    last_tested_age_h,
)
from orion.curiosity.replica.frame import ReplicaFrameInputs, build_frame, parse_answer, system_prompt
from orion.curiosity.replica.plan import PlanRowV1, invalid_status, validate_plan
from orion.curiosity.replica.state import PLAN_ACT_HORIZON_SEC, plan_act
from orion.curiosity.replica.track_record import record_lines, settle_plan
from orion.curiosity.value import PriorTestRecord
from orion.curiosity.worldview import Prior, WorldviewSnapshot, fresh_priors, select_priors, with_head
from orion.prime.state import PrimeConcernsViewV1, PrimeConcernViewV1
from orion.schemas.curiosity_replica import (
    CLAIM_MAX,
    MAX_CANDIDATES,
    CuriosityOfferCandidateV1,
    CuriosityOfferPickV1,
    CuriosityOfferPlanRequestV1,
    CuriositySignalStanceV1,
    CuriosityPendingFocusV1,
)

NOW = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)


def _prior(i: int, conf: float = 0.55, tested: int = 0, last: str = "") -> Prior:
    return Prior(f"p{i:03d}", f"claim {i}", conf, "open", tested, last_tested_at=last)


def _pool(n: int) -> list[Prior]:
    # Confidence spread so the uncertainty and value sorts differ.
    return [_prior(i, conf=0.05 + 0.9 * (i % 19) / 18, tested=i % 3) for i in range(n)]


def _cands(n: int = 8) -> list[CuriosityOfferCandidateV1]:
    return [CuriosityOfferCandidateV1(prior_id=f"p{i:03d}", claim=f"claim {i}", confidence=0.5) for i in range(n)]


def _picks(*ids: str, why: str = "it is close to even") -> list[CuriosityOfferPickV1]:
    return [CuriosityOfferPickV1(prior_id=i, why=why) for i in ids]


# --- candidates -----------------------------------------------------------------------------


def test_candidates_are_the_four_buckets_cut_to_sixty_in_a_neutral_order():
    pool = _pool(200)
    nats = {p.prior_id: (i * 7919) % 200 / 1000.0 for i, p in enumerate(pool)}
    got = build_candidates(pool, now=NOW, rotate_seed="run1", expected_nats_for=lambda p: nats[p.prior_id], moves={})
    ids = [c.prior_id for c in got]
    assert len(ids) == len(set(ids)) == MAX_CANDIDATES
    by_unc = select_priors([_row(p) for p in pool], sample=UNCERTAIN_N, stale_after=0, rotate_seed="run1")[0]
    assert {p.prior_id for p in by_unc} <= set(ids), "the 30 most uncertain are always candidates"
    top_nats = sorted(pool, key=lambda p: -nats[p.prior_id])[:NATS_N]
    assert {p.prior_id for p in top_nats} <= set(ids)
    # Neutral order: not the uncertainty order the rule arm would show.
    assert ids[: len(by_unc)] != [p.prior_id for p in by_unc]
    # Reproducible per run, different across runs.
    again = build_candidates(pool, now=NOW, rotate_seed="run1", expected_nats_for=lambda p: nats[p.prior_id], moves={})
    other = build_candidates(pool, now=NOW, rotate_seed="run2", expected_nats_for=lambda p: nats[p.prior_id], moves={})
    assert [c.prior_id for c in again] == ids and [c.prior_id for c in other] != ids


def test_every_prior_tested_in_the_last_week_is_a_candidate():
    pool = [_prior(i, conf=0.97, tested=1, last="") for i in range(120)]
    recent = _prior(500, conf=0.99, tested=2, last=(NOW - timedelta(days=3)).isoformat())
    old = _prior(501, conf=0.99, tested=2, last=(NOW - timedelta(days=30)).isoformat())
    got = {c.prior_id for c in build_candidates([*pool, recent, old], now=NOW, rotate_seed="r",
                                                expected_nats_for=lambda p: 0.0, moves={})}
    assert recent.prior_id in got


def test_a_candidate_line_carries_its_test_history_and_a_clipped_claim():
    long = Prior("x", "word " * 80, 0.62, "open", 2, last_tested_at=(NOW - timedelta(hours=30)).isoformat())
    never = Prior("y", "short", 1.7, "open", 0)
    hist = [PriorTestRecord("x", 0.5, 0.55), PriorTestRecord("x", 0.55, 0.62)]
    c = {c.prior_id: c for c in build_candidates([long, never], now=NOW, rotate_seed="r",
                                                 expected_nats_for=lambda p: 0.0, moves=last_moves(hist))}
    assert len(c["x"].claim) <= CLAIM_MAX and c["x"].claim.endswith("…")
    assert c["x"].times_tested == 2 and abs(c["x"].last_tested_age_h - 30) < 1e-6
    assert c["x"].last_move == pytest.approx(0.07)
    # A confidence of 1.7 is a protocol error: a gap, never a number.
    assert c["y"].confidence is None and c["y"].last_tested_age_h is None and c["y"].last_move is None


@pytest.mark.parametrize("stamp,expect", [
    ("2026-11-03T12:00:00Z", 24.0),
    (str(int((NOW - timedelta(hours=2)).timestamp() * 1000)), 2.0),
    (str(int((NOW - timedelta(hours=5)).timestamp())), 5.0),  # epoch seconds
    ("2026-11-01T19:4xZ", None),  # a partly typed stamp from the live graph
    ("2027-01-01T00:00:00Z", None),  # the future is unreadable, not "0 h ago"
])
def test_test_ages_read_the_stamp_shapes_found_in_the_live_graph(stamp, expect):
    age = last_tested_age_h(Prior("a", "c", 0.5, "open", 1, last_tested_at=stamp), NOW)
    assert age == (None if expect is None else pytest.approx(expect, abs=1e-6))


def _row(p: Prior) -> dict:
    return {"prior_id": p.prior_id, "claim": p.claim, "confidence": p.confidence, "status": "open",
            "times_tested": p.times_tested, "formed_from": "", "last_tested_at": p.last_tested_at}


# --- the plan check ---------------------------------------------------------------------------


IDS = [f"p{i:03d}" for i in range(8)]


@pytest.mark.parametrize("picks,reason", [
    (_picks("p000", "p001"), "too_few_picks:2"),
    (_picks(*IDS[:6]), "too_many_picks:6"),
    (_picks("p000", "p001", "p000"), "duplicate_pick:3"),
    (_picks("p000", "p001", "zzz"), "not_a_candidate:3"),
    (_picks("p000", "p001", "p007"), "not_fresh:3"),
    (_picks("p000", "p001", "p002", why="   "), "why_length:1"),
])
def test_any_bad_pick_rejects_the_whole_plan(picks, reason):
    assert validate_plan(picks, candidate_ids=IDS, fresh_ids=IDS[:7]) == reason


def test_a_good_plan_passes_and_a_signal_stance_must_name_the_live_signal():
    good = _picks("p003", "p000", "p005")
    assert validate_plan(good, candidate_ids=IDS, fresh_ids=IDS) is None
    stance = CuriositySignalStanceV1(signal_id="sig1", stance="challenge", reason="the run is not stuck")
    assert validate_plan(good, candidate_ids=IDS, fresh_ids=IDS, signal_stance=stance, live_signal_id="sig1") is None
    assert validate_plan(good, candidate_ids=IDS, fresh_ids=IDS, signal_stance=stance) == "unknown_signal"
    assert validate_plan(good, candidate_ids=IDS, fresh_ids=IDS, signal_stance=stance,
                         live_signal_id="other") == "unknown_signal"


def test_a_plan_row_has_a_fallback_reason_exactly_when_it_was_not_used():
    ok = PlanRowV1(run_id="r", status="ok", requested_at=NOW, picks=[{"prior_id": "a", "why": "x"}])
    assert ok.used and ok.head_ids == ["a"]
    bad = PlanRowV1(run_id="r", status=invalid_status("too_few_picks:2"), fallback_reason="too_few_picks:2",
                    requested_at=NOW, picks=[{"prior_id": "a", "why": "x"}])
    assert not bad.used and bad.head_ids == []
    with pytest.raises(ValueError):
        PlanRowV1(run_id="r", status="ok", fallback_reason="x", requested_at=NOW)
    with pytest.raises(ValueError):
        PlanRowV1(run_id="r", status="timeout", requested_at=NOW)
    with pytest.raises(ValueError):
        PlanRowV1(run_id="r", status="late", fallback_reason="late", requested_at=NOW)


# --- the offer a good plan makes --------------------------------------------------------------


def test_a_plan_fills_the_top_and_every_arm_keeps_the_same_layout():
    pool = _pool(30) + [_prior(900, conf=0.5, tested=3), _prior(901, conf=0.5, tested=4)]
    rows = [_row(p) for p in pool]
    rule, stale, _ = select_priors(rows, sample=8, stale_after=3, rotate_seed="run", coverage_slots=2)
    view = WorldviewSnapshot(live_priors=rule, stale_priors=stale, prior_rows=tuple(rows))
    assert {p.prior_id for p in fresh_priors(view, stale_after=3)} == {p.prior_id for p in pool[:30]}
    picks = ["p017", "p004", "p011"]
    planned = with_head(view, picks, sample=8, stale_after=3, rotate_seed="run", coverage_slots=2)
    got = [p.prior_id for p in planned.live_priors]
    assert got[:3] == picks and len(got) == len(rule) == 8
    # Slots 4..6 continue the uncertainty order with the picks skipped.
    rest = [p.prior_id for p in rule if not p.coverage and p.prior_id not in picks]
    assert got[3:6] == rest[:3]
    assert [p.coverage for p in planned.live_priors] == [p.coverage for p in rule]
    assert planned.stale_priors == view.stale_priors
    assert with_head(view, [], sample=8, stale_after=3, rotate_seed="run", coverage_slots=2) is view


# --- the frame ---------------------------------------------------------------------------------


def _request(**over) -> CuriosityOfferPlanRequestV1:
    base = dict(run_id="abc123def456", correlation_id="c1", deadline_at=NOW + timedelta(seconds=90),
                candidates=_cands(), stale_ids=["s1", "s2"])
    base.update(over)
    return CuriosityOfferPlanRequestV1(**base)


def test_every_unread_section_is_a_named_gap_never_calm():
    frame = build_frame(ReplicaFrameInputs(now=NOW, request=_request(), runs_error="URLError: refused",
                                           outcomes_error="pg down", concerns_error="Prime has published no view",
                                           signal_error="ConnectionError: x", digest_error="pg down",
                                           record_error="pg down"))
    assert frame.count("(unavailable:") >= 5
    assert "None" not in frame
    assert "PRIME'S SIGNAL ON CURIOSITY: (unavailable: ConnectionError: x)" in frame
    assert "Hub waits 90 s more" in frame
    assert "STALE PRIORS" in frame and ": 2: s1, s2" in frame


def test_the_frame_shows_prime_juniper_and_the_record_with_their_ages_and_hides_the_signal_arm():
    concerns = PrimeConcernsViewV1(as_of=NOW - timedelta(minutes=2), concerns=[
        PrimeConcernViewV1(concern_id="c1", text="the walkway names drift", age_h=5.0, arc_id="arc9")])
    focus = CuriosityPendingFocusV1(focus_id="f1", origin="juniper", question="what did I see at 11:39?")
    rows = [
        {"run_id": "r2", "requested_at": NOW - timedelta(hours=3), "status": "ok",
         "picks": [{"prior_id": "a", "why": "w"}, {"prior_id": "b", "why": "w"}], "turn_ok": True,
         "realized_nats": 0.02, "per_prior": [{"kind": "tested", "prior_id": "a", "nats": 0.02}]},
        {"run_id": "r1", "requested_at": NOW - timedelta(hours=9), "status": "timeout", "fallback_reason": "timeout",
         "picks": []},
    ]
    frame = build_frame(ReplicaFrameInputs(
        now=NOW, request=_request(pending_focus=focus), runs=[], outcomes={}, concerns=concerns,
        signal={"signal_id": "sig1", "direction": "slow", "why": "too many runs", "expires_at": NOW + timedelta(hours=1)},
        digest={"summary": "Juniper and Orion talked about walkway names.", "title": "Walkway names",
                "window_end": "2026-11-03T06:00:00+00:00", "created_at": NOW - timedelta(hours=23), "index": "day:2026-11-02"},
        record_rows=rows, fallback_counts={"ok": 1, "timeout": 1}, plans_total=2))
    assert "the walkway names drift (held 5 h, about arc arc9)" in frame and "2 min ago" in frame
    assert "slow (signal_id sig1, until 13:00 UTC)" in frame
    assert "treated" not in frame and "control" not in frame
    assert "Juniper asked for this" in frame and "what did I see at 11:39?" in frame
    assert "title: Walkway names" in frame and "written 23 h ago" in frame
    assert "Orion tested 1 of 2 picks (a moved 0.020 nats)" in frame
    assert "fallback (timeout)" in frame and "fallbacks, last 30 days: 1 of 2 plans (timeout 1)" in frame
    assert "None" not in frame


def test_an_old_concerns_view_reads_as_prime_maybe_down():
    concerns = PrimeConcernsViewV1(as_of=NOW - timedelta(hours=2), concerns=[])
    frame = build_frame(ReplicaFrameInputs(now=NOW, request=_request(), concerns=concerns))
    assert "STALE, Prime may not be running" in frame


def test_the_system_prompt_states_the_score_and_keeps_stances_record_only():
    text = system_prompt()
    assert "3 to 5 picks" in text and "nothing you write is shown to Orion" in text
    assert "recorded only; nothing acts on them yet" in text


def test_the_answer_parse_is_lenient_on_extras_and_strict_on_shape():
    ans, err = parse_answer({"reading": "r", "picks": [{"prior_id": "a", "why": "w", "extra": 1}], "mood": "x"})
    assert err is None and ans.picks[0].prior_id == "a"
    assert parse_answer(None) == (None, "no_json_object")
    _, err = parse_answer({"picks": [{"prior_id": "a", "why": "w"}], "abstain": True})
    assert err and err.startswith("invalid_answer")


# --- grading a plan on its run -----------------------------------------------------------------


def _scored(**over) -> dict:
    row = {"run_id": "r", "status": "ok", "picks": [{"prior_id": "a", "why": "w"}, {"prior_id": "b", "why": "w"}],
           "turn_ok": True, "realized_nats": 0.03, "unknown_reason": None,
           "per_prior": [{"kind": "tested", "prior_id": "b", "nats": 0.03}, {"kind": "formed", "prior_id": "n"}]}
    row.update(over)
    return row


def test_a_plan_is_graded_on_the_same_number_the_arms_are_scored_on():
    s = settle_plan(_scored())
    assert (s.kind, s.produced) == ("graded", True) and "Orion tested 1 of my 2 picks (b: 0.030 nats)" in s.note
    assert settle_plan(_scored(realized_nats=0.0, per_prior=[])).produced is False
    assert (settle_plan(_scored(turn_ok=False)).kind, settle_plan(_scored(turn_ok=False)).produced) == ("graded", False)
    assert settle_plan(_scored(turn_ok=None, realized_nats=None)).kind == "pending"
    assert settle_plan(_scored(realized_nats=None, unknown_reason="no_snapshot")).kind == "ungradable"
    fb = settle_plan({"run_id": "r", "status": "invalid:not_fresh:2", "fallback_reason": "not_fresh:2"})
    assert (fb.kind, fb.produced) == ("graded", False) and "plan not used" in fb.note


def test_json_text_from_the_driver_reads_the_same_as_objects():
    import json

    row = _scored(picks=json.dumps(_scored()["picks"]), per_prior=json.dumps(_scored()["per_prior"]))
    assert settle_plan(row) == settle_plan(_scored())
    assert record_lines([]) == ["(no plans yet)"]


def test_the_plan_act_is_the_same_act_when_rebuilt_for_its_close():
    a, b = plan_act("t1", "run1"), plan_act("t1", "run1")
    assert a == b and a.actor == "curiosity.replica" and a.verb == "infer"
    assert a.goal_ref == "curiosity_run:run1" and a.expect.horizon_sec == PLAN_ACT_HORIZON_SEC
    assert a.expect.signal_id == "artifact_produced"
