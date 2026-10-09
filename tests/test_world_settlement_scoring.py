"""Settle-time scoring of world-action episodes (acceptance checks 5, 6, 7 and 10's ledger half)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion.autonomy.prediction import EffectPosterior
from orion.feedback.outcome_resolution import resolve_action_outcomes
from orion.feedback.world_settlement import score_world_episode
from orion.hardware_watch.rules import TempPoint
from orion.schemas.action_prediction import ExpectedEffectV1
from orion.schemas.execution_dispatch_frame import ExecutionDispatchCandidateV1, ExecutionDispatchFrameV1
from orion.schemas.field_state import FieldStateV1

T0 = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
AFTER = T0 + timedelta(minutes=20)


def _points(before_c=30.0, after_c=29.6, hot=False):
    pts = [TempPoint(T0 + timedelta(seconds=30 * i), before_c + (after_c - before_c) * i / 40) for i in range(41)]
    if hot:
        pts.append(TempPoint(T0 + timedelta(minutes=10, seconds=1), 32.2))
        pts.sort(key=lambda p: p.ts)
    return pts


def _episode(arm="treated", terminal="expired", started=True, **kw):
    ep = {"episode_id": "dispatch:1", "arm": arm, "decided_at": T0, "dispatch_kind": "self_regulate",
          "target_id": "pool:background_gpu", "dispatch_frame_id": "frame:1", "open_loop_id": "open-loop-cab",
          "expected_effect": {"signal_id": "cabinet_heat_pressure", "direction": "decrease", "predicted_delta": 0.0},
          "settlement_state": terminal if arm == "treated" else None,
          "settlement": {"ttl_sec": 900, "manipulation_check": {"started_at": T0.isoformat() if started else None,
                                                                 "drain": "drained"}}}
    ep.update(kw)
    return ep


def _score(ep, **kw):
    args = {"episode": ep, "points": _points(), "incidents": [], "prior": None, "control_prior": None,
            "now": AFTER + timedelta(seconds=30)}
    args.update(kw)
    return score_world_episode(**args)


def test_check5_treated_expired_scores_on_the_20_minute_clock_and_updates_the_posterior():
    s = _score(_episode())
    r = s.record
    assert r.signal_id == "cabinet_heat_pressure" and r.arm == "dispatched" and r.observed_at == AFTER
    # minute means (+-30 s) of a falling ramp 30.0 -> 29.6 C
    assert r.baseline == pytest.approx(0.5, abs=0.01) and r.observed_after == pytest.approx(0.4, abs=0.01)
    assert r.observed_delta == pytest.approx(-0.1, abs=0.01) and r.surprise_nats > 0 and r.claim_upheld is True
    assert r.posterior_n == 1 and s.outcome["posterior_updated"] is True
    assert s.loop_outcome == {**s.loop_outcome, "verdict": "acted", "actor": "orion", "loop_id": "open-loop-cab"}


def test_not_ready_before_the_window_closes_or_before_settlement():
    assert _score(_episode(), now=AFTER - timedelta(seconds=1)) is None
    assert _score(_episode(terminal="active")) is None
    late = _score(_episode(terminal="active"), now=AFTER + timedelta(minutes=11))
    assert late.record is None and late.outcome["excluded_reason"] == "terminal:unsettled"


def test_check7_control_arm_is_a_ledger_row_with_no_posterior_update_and_no_acted_verdict():
    s = _score(_episode(arm="control"))
    assert s.record.arm == "randomized_holdback" and s.record.surprise_nats == 0.0
    assert s.record.posterior_n == 0 and s.outcome["posterior_updated"] is False
    key, cell = s.control_cell
    assert key[:2] == ("cabinet_heat_pressure", "randomized_holdback") and key[2] == s.record.baseline_bin
    assert cell.posterior.n == 1
    assert s.loop_outcome is None


@pytest.mark.parametrize("terminal", ["cancelled", "preempted_by_reflex", "settlement_timeout"])
def test_only_expired_reaches_the_posterior(terminal):
    s = _score(_episode(terminal=terminal))
    assert s.record is None and s.outcome["excluded_reason"] == f"terminal:{terminal}"
    assert s.loop_outcome["verdict"] == "acted"          # Orion did act; the verdict stays non-final


def test_refused_shed_never_started_writes_no_verdict():
    s = _score(_episode(terminal="refused:daily_cap", started=False))
    assert s.record is None and s.loop_outcome is None


def test_check10_reflex_overlap_excluded_in_both_arms():
    inc = [{"rule": "cooling", "opened_at": T0 + timedelta(minutes=5)}]
    for arm in ("treated", "control"):
        s = _score(_episode(arm=arm), incidents=inc)
        assert s.record is None and s.control_cell is None
        assert s.outcome["excluded_reason"] == "overlap:reflex" and "overlap:reflex" in s.outcome["overlap"]


def test_heat_incident_and_render_gate_are_kept_as_covariates():
    s = _score(_episode(), incidents=[{"rule": "gpu_heat", "opened_at": T0 + timedelta(minutes=2)}],
               points=_points(hot=True))
    assert s.record is not None and s.outcome["overlap"] == ["overlap:heat_incident", "overlap:render_gate"]


def test_missing_reading_is_not_a_zero():
    s = _score(_episode(), points=_points()[:5])
    assert s.record is None and s.outcome["excluded_reason"] == "missing_reading:after"


def test_warm_prior_is_scored_against_its_own_belief():
    prior = EffectPosterior(mean=-0.1, variance=0.01, n=3)
    s = _score(_episode(), prior=prior)
    assert s.record.posterior_n == 4 and s.record.surprise_nats < _score(_episode()).record.surprise_nats


def test_field_window_never_scores_the_world_signal():
    cand = ExecutionDispatchCandidateV1(
        dispatch_id="d", source_decision_id="p", source_proposal_id="q", dispatch_status="dispatched",
        dispatch_mode="dispatch_read_only", dispatch_kind="self_regulate", target_id="pool:background_gpu",
        target_kind="system", risk_score=0.1, confidence_score=1.0, dispatched_at=T0, result_ref="result:d",
        expected_effect=ExpectedEffectV1(signal_id="cabinet_heat_pressure", direction="decrease", predicted_delta=0.0,
                                         predictor_variance=0.25, predictor_n=0, cold_start=True))
    frame = ExecutionDispatchFrameV1(frame_id="f", generated_at=T0, source_policy_frame_id="p", source_proposal_frame_id="q",
                                     source_field_tick_id="t", dispatch_mode="dispatch_read_only", dispatched_candidates=[cand])
    field = FieldStateV1(generated_at=T0, tick_id="t", node_vectors={"node:athena": {"execution_pressure": 0.4}})
    res = resolve_action_outcomes(dispatch_frame=frame, feedback_frame_id="fb", field_before=field, field_after=field)
    assert res.records == [] and res.skipped == {"d": "settle_time_signal:cabinet_heat_pressure"}
