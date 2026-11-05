"""The cooling world act's pure rules: the cabinet signal, self_shed eligibility, the event want, its
salience, the binding rule, admission, the draw, the episode row and the settle rule.

Until 2026-11-05 this file drove shed_background_gpu through the proposal -> policy -> dispatch builders. The
act left that ladder in retirement PR 5 (docs/superpowers/specs/2026-11-04-retire-proposal-policy-dispatch-
rungs.md): the market decides it now (orion/autonomy/shed_act.py, services/orion-durable-runs/app/
cooling_act.py). The signal, eligibility and winner-binding tests are kept (self_shed still owns them; the act
ledger still imports bind_workspace_winner)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from orion.autonomy import shed_act
from orion.autonomy.cabinet_heat import (
    CABINET_NODE_ID, cabinet_heat_pressure, cabinet_warming_error, minute_mean, read_cabinet_heat,
)
from orion.autonomy.prediction import EffectPosterior
from orion.autonomy.self_shed import HardwareWatchView, bind_workspace_winner, evaluate_shed_eligibility
from orion.autonomy.shed_settlement import SHED_PENDING, settle_shed_result
from orion.autonomy.world_settlement import score_world_episode
from orion.hardware_watch.rules import TempPoint
from orion.proposals.policy import ProposalTemplateV1, load_proposal_policy
from orion.schemas.attention_salience import AttentionLoopOutcomeV1
from orion.substrate.attention.verdicts import TERMINAL_VERDICTS

REPO = Path(__file__).resolve().parents[1]
POLICY = load_proposal_policy(REPO / "config/proposals/proposal_policy.v1.yaml")
NOW = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


def _points(start_c: float, end_c: float, *, minutes: int = 20, now: datetime = NOW) -> list[TempPoint]:
    n = minutes * 2
    return [TempPoint(now - timedelta(seconds=30 * (n - i)), start_c + (end_c - start_c) * i / n) for i in range(n + 1)]


def _projection(*, node=CABINET_NODE_ID, age_sec=30, dwell=3, action="watch") -> dict:
    return {"generated_at": (NOW - timedelta(seconds=age_sec)).isoformat(), "selected_action_type": action,
            "selected_open_loop_id": "open-loop-cab", "dwell_ticks": dwell, "attended_node_ids": [node]}


def _healthy_hw() -> HardwareWatchView:
    return HardwareWatchView.from_health({"enabled": True, "last_tick_ok": True,
                                          "last_tick_at": (NOW - timedelta(seconds=20)).isoformat(),
                                          "open_incidents": []}, now=NOW)


def _eligibility(**kw) -> dict:
    args = {"cabinet": read_cabinet_heat(_points(29.6, 30.4), NOW), "hardware_watch": _healthy_hw(),
            "background_granted": 1, "background_queued": 0, "in_flight_episode_ids": [],
            "holdback_fraction": 0.5, "now": NOW}
    args.update(kw)
    return evaluate_shed_eligibility(**args)


# --- the signal --------------------------------------------------------------------------------

def test_check1_pressure_scale_on_the_thermal_gate_constants():
    assert cabinet_heat_pressure(27.9) == 0.0 and cabinet_heat_pressure(28.0) == 0.0
    assert cabinet_heat_pressure(29.5) == 0.375 and cabinet_heat_pressure(32.0) == 1.0
    assert cabinet_heat_pressure(None) is None


def test_warming_error_needs_elevated_and_rising_and_is_not_the_level():
    assert cabinet_warming_error("elevated", 0.6, 0.5) == pytest.approx(0.6)
    assert cabinet_warming_error("elevated", 0.4, 0.5) == 0.0     # elevated but not rising: zero
    assert cabinet_warming_error("hot", 1.5, 0.5) == 0.0          # hot belongs to the render gate/swap guard
    assert cabinet_warming_error("normal", 1.0, 0.5) == 0.0
    flat = read_cabinet_heat(_points(30.5, 30.5), NOW)
    assert flat.thermal_state == "elevated" and flat.warming_error == 0.0 and flat.pressure > 0.5


def test_hysteresis_keeps_elevated_below_the_trip_and_stale_reads_unknown():
    cooling = _points(30.0, 29.0, minutes=60)
    assert read_cabinet_heat(cooling, NOW).thermal_state == "elevated"     # re-arms only below 28.0
    stale = read_cabinet_heat(_points(29.6, 30.4, now=NOW - timedelta(minutes=10)), NOW)
    assert stale.thermal_state == "unknown" and stale.warming_error == 0.0


def test_minute_mean_absence_is_none_not_zero():
    assert minute_mean([], NOW) is None
    assert minute_mean(_points(29.0, 29.0), NOW) == pytest.approx(29.0)


# --- binding + eligibility ------------------------------------------------------------------------

@pytest.mark.parametrize("proj,reason", [
    (_projection(age_sec=120), "stale"), (_projection(dwell=1), "dwell_1"), (_projection(action="none"), "no_selected_action"),
    (_projection(node="node:substrate.chat"), "node_not_bound"), (None, "no_projection"),
])
def test_winner_unbindable_reasons(proj, reason):
    winner, why = bind_workspace_winner(proj, broadcast_log_id="b", binds_to_nodes=[CABINET_NODE_ID], now=NOW)
    assert winner is None and reason in why


def test_binding_follows_the_selected_loops_node_not_any_coalition_member():
    proj = _projection()
    proj["attended_node_ids"] = ["node:substrate.execution", CABINET_NODE_ID]
    winner, why = bind_workspace_winner(proj, broadcast_log_id="b", binds_to_nodes=[CABINET_NODE_ID], now=NOW)
    assert winner is None and "node_not_bound" in why


def test_winner_binds_within_90s_with_dwell_2():
    winner, why = bind_workspace_winner(_projection(age_sec=89, dwell=2), broadcast_log_id="broadcast-1",
                                        binds_to_nodes=[CABINET_NODE_ID], now=NOW)
    assert why is None and winner.open_loop_id == "open-loop-cab" and winner.broadcast_log_id == "broadcast-1"


@pytest.mark.parametrize("override,refusal", [
    ({"hardware_watch": HardwareWatchView.from_health(None, now=NOW)}, "hardware_watch_unknown:unreachable"),
    ({"hardware_watch": HardwareWatchView.from_health({"enabled": True, "last_tick_ok": True, "last_tick_at": NOW.isoformat(),
                                                        "reflex_shed": {"active": True, "reason": "cabinet_unknown"}}, now=NOW)},
     "reflex_active:cabinet_unknown"),
    ({"hardware_watch": HardwareWatchView.from_health({"enabled": True, "last_tick_ok": True, "last_tick_at": NOW.isoformat(),
                                                        "open_incidents": [{"incident_id": "x", "rule": "cooling",
                                                                            "shed_requested": True, "shed_reason": "cabinet_elevated"}]}, now=NOW)},
     "reflex_active:cooling_incident:cabinet_elevated"),
    ({"hardware_watch": HardwareWatchView.from_health({"enabled": True, "last_tick_ok": True,
                                                        "last_tick_at": (NOW - timedelta(minutes=10)).isoformat()}, now=NOW)},
     "hardware_watch_unknown:stale"),
    ({"background_granted": 0, "background_queued": 0}, "no_background_work"),
    ({"background_granted": None, "background_queued": None}, "pool_occupancy_unknown"),
    ({"in_flight_episode_ids": ["d1"]}, "winner_loop_in_flight"),
    ({"cabinet": read_cabinet_heat(_points(26.0, 26.5), NOW)}, "thermal_not_elevated:normal"),
    ({"cabinet": read_cabinet_heat(_points(33.5, 34.2), NOW)}, "thermal_critical:reflex_covers"),
])
def test_eligibility_refusals(override, refusal):
    snap = _eligibility(**override)
    assert not snap["eligible"] and refusal in snap["refusals"]


def test_template_validation_refuses_a_winner_template_with_no_nodes():
    with pytest.raises(ValueError):
        ProposalTemplateV1(kind="self_regulate", target_kind="system", target_id="x", proposed_effect="reduce_pressure",
                           required_policy_gate="self_reversible", target_binding="workspace.winner")


# --- one trigger, never two -------------------------------------------------------------------------

def test_the_shed_template_is_gone_from_the_proposal_policy():
    """PR 5: the act's only trigger is the want market. A template left here would be a second one."""
    assert "shed_background_gpu" not in POLICY.proposal_templates
    assert not [k for k, t in POLICY.proposal_templates.items() if t.target_binding == "workspace.winner"]


def test_check6_acted_is_a_valid_verdict_and_never_terminal():
    assert "acted" not in TERMINAL_VERDICTS
    AttentionLoopOutcomeV1(outcome_id="o", loop_id="l", theme_key="l", verdict="acted", actor="orion")


# --- D8 (thermal controller v2, APPROVED 2026-10-06): the learned shed owns 29.5-34 C ----------

@pytest.mark.parametrize("points", [
    _points(30.5, 30.5),                                  # elevated, flat: no rise needed any more
    _points(31.5, 32.4),                                  # hot, below the 34 C critical line
    _points(29.6, 30.4, now=NOW - timedelta(minutes=10)), # unknown (stale)
])
def test_d8_eligible_at_elevated_hot_and_unknown(points):
    snap = _eligibility(cabinet=read_cabinet_heat(points, NOW))
    assert snap["eligible"], snap["refusals"]


def test_d8_not_eligible_at_normal_or_critical():
    assert "thermal_not_elevated:normal" in _eligibility(cabinet=read_cabinet_heat(_points(25.0, 25.0), NOW))["refusals"]
    assert "thermal_critical:reflex_covers" in _eligibility(cabinet=read_cabinet_heat(_points(34.0, 34.0), NOW))["refusals"]


def test_d8_a_gpu_heat_incident_no_longer_blocks_it():
    hw = HardwareWatchView.from_health({"enabled": True, "last_tick_ok": True, "last_tick_at": NOW.isoformat(),
                                        "open_incidents": [{"incident_id": "g", "rule": "gpu_heat", "subject": "circe/gpu3"}],
                                        "reflex_shed": {"active": False, "reason": None}}, now=NOW)
    snap = _eligibility(hardware_watch=hw)
    assert snap["eligible"], snap["refusals"]
    assert snap["hardware_watch"]["open_incident_rules"] == ["gpu_heat"]


def test_d8_an_open_cooling_incident_that_does_not_shed_does_not_block():
    """v2 cooling incidents are alert-only (D5); only the reflex's own signal blocks (C7)."""
    hw = HardwareWatchView.from_health({"enabled": True, "last_tick_ok": True, "last_tick_at": NOW.isoformat(),
                                        "open_incidents": [{"incident_id": "c", "rule": "cooling", "shed_requested": False}],
                                        "reflex_shed": {"active": False}}, now=NOW)
    assert _eligibility(hardware_watch=hw)["eligible"]


# --- the event, its salience and the want (ladder retirement PR 5) --------------------------------

def test_the_event_is_a_warming_moment_in_orions_band():
    assert shed_act.is_warming(read_cabinet_heat(_points(29.6, 30.4), NOW))        # elevated, +0.8 C / 15 min
    assert shed_act.is_warming(read_cabinet_heat(_points(32.1, 32.9), NOW))        # hot, still Orion's band
    assert not shed_act.is_warming(read_cabinet_heat(_points(30.5, 30.5), NOW))    # warm but flat: no event
    assert not shed_act.is_warming(read_cabinet_heat(_points(26.0, 27.0), NOW))    # rising but normal
    stale = read_cabinet_heat(_points(29.6, 30.4, now=NOW - timedelta(minutes=10)), NOW)
    assert not shed_act.is_warming(stale)                                           # unknown: the reflex's


def test_warming_minutes_are_the_sources_own_history_on_raw_readings():
    flat = _points(30.5, 30.5, minutes=60)
    assert shed_act.warming_minutes(flat) == []                                     # warm, never rising
    rising = _points(29.0, 31.0, minutes=60)                                        # +0.5 C per 15 min
    mins = shed_act.warming_minutes(rising)
    assert mins and all(r >= 0.5 for _, r in mins)
    assert all(m.second == 0 for m, _ in mins) and len({m for m, _ in mins}) == len(mins)   # one per minute
    first_warm = next(p.ts for p in rising if p.value >= 29.5)
    assert min(m for m, _ in mins) >= first_warm.replace(second=0, microsecond=0)  # not before the band


def test_salience_is_the_strict_below_percentile_with_rest_point_zero():
    hist = [(NOW - timedelta(hours=h), 0.5 + 0.01 * h) for h in range(1, 41)]      # 40 past rises 0.51..0.90
    assert shed_act.cabinet_salience(0.5, hist, NOW) == 0.0                         # the smallest usual rise
    assert shed_act.cabinet_salience(0.95, hist, NOW) == 1.0
    assert shed_act.cabinet_salience(0.705, hist, NOW) == pytest.approx(20 / 40)
    assert shed_act.cabinet_salience(None, hist, NOW) is None                      # no rise, no event
    assert shed_act.cabinet_salience(0.7, hist[:10], NOW) is None                  # too little history
    old = [(NOW - timedelta(days=31), 0.1)] * 40
    assert shed_act.cabinet_salience(0.7, old, NOW) is None                        # only the last 30 days count


def test_the_cooling_want_is_an_event_want_with_a_deterministic_id():
    at = datetime(2026, 11, 5, 12, 0, 7, 123456, tzinfo=timezone.utc)
    w = shed_act.cooling_want(reading_at=at, salience=0.4, opened_at=NOW)
    again = shed_act.cooling_want(reading_at=at, salience=0.9, opened_at=NOW)
    assert w.want_id == again.want_id                                               # a restart re-finds it
    assert w.family == "self_regulate.shed" and w.tier == 1 and w.producer == shed_act.PRODUCER
    assert w.event_ref == "cabinet_reading:athena:2026-11-05T12:00:07Z" and w.salience == 0.4
    assert w.window.latest == NOW + timedelta(seconds=shed_act.EVENT_WINDOW_SEC)
    from orion.physiology.binding import BINDING_PRODUCERS
    from orion.physiology.families import load_families

    assert shed_act.PRODUCER in BINDING_PRODUCERS
    fams, err = load_families(REPO / "config/physiology/families.yaml")
    cfg = fams.get(shed_act.FAMILY)
    assert err is None and cfg.tier == shed_act.TIER and cfg.lanes == () and cfg.deferrable_window_s == 900


# --- the binding rule: only funded sheds ----------------------------------------------------------

def _decision(outcome, reason, *, net=0.0, retry=None):
    return SimpleNamespace(outcome=outcome, reason=reason, net=net, retry_after_sec=retry, decision_id="d1",
                           binding_constraint=None)


def test_only_a_funded_decision_proceeds():
    end = NOW + timedelta(minutes=10)
    assert shed_act.shed_verdict(_decision("funded", "net_positive", net=0.5), NOW, end).proceed
    for d in (None, _decision("deferred", "physiology_unavailable"), _decision("deferred", "unknown_family")):
        v = shed_act.shed_verdict(d, NOW, end)
        assert not v.proceed and not v.close and v.why.startswith("no_answer")     # no answer: never sheds
        assert v.retry_at == NOW + timedelta(seconds=shed_act.NO_ANSWER_RETRY_SEC)
    v = shed_act.shed_verdict(_decision("deferred", "sated", retry=300), NOW, end)
    assert not v.proceed and not v.close and v.why == "deferred:sated" and v.retry_at == NOW + timedelta(seconds=300)
    # a capacity deferral worth its price is "queued at the pool" for curiosity -- never "funded" here
    v = shed_act.shed_verdict(_decision("deferred", "capacity_reserved", net=0.4), NOW, end)
    assert not v.proceed and not v.close


def test_a_held_want_never_waits_past_its_event_and_never_sheds_late():
    end = NOW + timedelta(seconds=60)
    v = shed_act.shed_verdict(_decision("deferred", "price", retry=3600), NOW, end)
    assert not v.proceed and v.retry_at == end
    late = shed_act.shed_verdict(_decision("deferred", "price"), end, end)
    assert late.close and not late.proceed and late.why == "event_expired"
    assert shed_act.shed_verdict(None, end + timedelta(seconds=1), end).close     # no answer at the end: closed


# --- admission, the draw, the episode row (moved from execution dispatch) -------------------------

@pytest.mark.parametrize("in_flight,treated,refusal", [
    (["e:old"], [], "world_action_in_flight"),
    ([], [{"decided_at": NOW - timedelta(minutes=20), "settlement_state": "expired", "settlement": {}}], "pool_gap"),
    ([], [{"decided_at": NOW - timedelta(hours=h), "settlement_state": "expired", "settlement": {}} for h in (2, 4, 6, 8)],
     "pool_daily_cap"),
    ([], [{"decided_at": NOW - timedelta(minutes=40), "settlement_state": "refused:disabled", "settlement": {}}],
     "pool_refusing:disabled"),
    ([], [{"decided_at": NOW - timedelta(hours=3), "settlement_state": "expired", "settlement": {}}], None),
])
def test_admission_before_the_draw(in_flight, treated, refusal):
    assert shed_act.admission_refusal(in_flight=in_flight, treated=treated, now=NOW, ttl_sec=900.0,
                                      min_gap_sec=900.0, max_sec_per_day=3600.0) == refusal


def test_the_draw_is_stable_per_decision_and_roughly_half():
    draws = [shed_act.stable_draw(f"cooling_act:{i}") for i in range(2000)]
    assert shed_act.stable_draw("cooling_act:7") == draws[7]
    assert 0.45 < sum(d < 0.5 for d in draws) / len(draws) < 0.55
    # the same hash rule dispatch used (sha256, first 13 hex digits): an id draws what it always drew
    import hashlib

    assert shed_act.stable_draw("x") == int(hashlib.sha256(b"x").hexdigest()[:13], 16) / float(16 ** 13)
    assert shed_act.arm_for("x", 0.0) == "treated"


def test_expected_effect_is_the_pooled_treated_posterior_or_a_cold_claim():
    cold = shed_act.expected_effect(None)
    assert cold.cold_start and cold.signal_id == "cabinet_heat_pressure" and cold.direction == "decrease"
    post = {("self_regulate", "pool:background_gpu", "cabinet_heat_pressure", 1): EffectPosterior(mean=-0.02, variance=0.005, n=7)}
    warm = shed_act.expected_effect(post)
    assert not warm.cold_start and warm.predictor_n == 7 and warm.predicted_delta == pytest.approx(-0.02)


def _steady(temp_c: float, *, start: datetime, minutes: int) -> list[TempPoint]:
    return [TempPoint(start + timedelta(seconds=30 * i), temp_c) for i in range(minutes * 2 + 1)]


@pytest.mark.parametrize("arm", ["treated", "control"])
def test_the_sql_writer_scorer_scores_the_new_episode_row_unchanged(arm):
    """The scorer (orion/autonomy/world_settlement.py) must keep finding and scoring episodes."""
    snap = _eligibility()
    snap["want"] = {"want_id": "w1", "decision_id": "d1", "salience": 0.4}
    settlement = {"state": SHED_PENDING, "decided_at": NOW.isoformat(), "ttl_sec": 900.0} if arm == "treated" else None
    row = shed_act.episode_row(episode_id="cooling_act:w1", arm=arm, now=NOW, eligibility=snap,
                               effect=shed_act.expected_effect(None), decision_id="d1", ttl_sec=900.0,
                               settlement=settlement)
    assert row["template"] == "shed_background_gpu" and row["dispatch_kind"] == "self_regulate"
    assert row["target_id"] == "pool:background_gpu" and row["open_loop_id"] is None
    assert row["scoring_due_at"] == NOW + timedelta(seconds=1200)
    if arm == "treated":
        settled = settle_shed_result(result_json={"settlement": row["settlement"]},
                                     pool_row={"state": "expired", "started_at": NOW.isoformat(),
                                               "ended_at": (NOW + timedelta(seconds=900)).isoformat()},
                                     now=NOW + timedelta(seconds=910))
        row = {**row, "settlement_state": settled.terminal, "settlement": settled.result_json["settlement"]}
    pts = _steady(30.5, start=NOW - timedelta(minutes=2), minutes=26)
    score = score_world_episode(episode=row, points=pts, incidents=[], prior=None, control_prior=None,
                                now=NOW + timedelta(seconds=1300))
    assert score is not None and score.outcome["excluded_reason"] is None
    assert score.record.arm == ("dispatched" if arm == "treated" else "randomized_holdback")
    assert score.loop_outcome is None                       # no attention loop any more: no loop verdict


def test_settle_rules_terminal_states_and_orphan():
    decided = NOW - timedelta(minutes=30)
    pending = {"settlement": {"state": SHED_PENDING, "decided_at": decided.isoformat(), "ttl_sec": 900}}
    assert settle_shed_result(result_json=pending, pool_row={"state": "active"}, now=decided + timedelta(minutes=5)) is None
    for state, status in (("expired", "success"), ("cancelled", "empty"), ("preempted_by_reflex", "empty")):
        s = settle_shed_result(result_json=pending, pool_row={"state": state}, now=NOW)
        assert s.terminal == state and s.status == status
    s = settle_shed_result(result_json=pending, pool_row={"state": "refused", "refusal": "daily_cap"}, now=NOW)
    assert s.terminal == "refused:daily_cap" and not s.success
    orphan = settle_shed_result(result_json=pending, pool_row=None, now=decided + timedelta(seconds=1201))
    assert orphan.terminal == "settlement_timeout"
    assert settle_shed_result(result_json=pending, pool_row=None, now=decided + timedelta(seconds=1100)) is None


def test_a_funded_bid_at_or_after_the_events_end_never_sheds():
    """Review blocker 2026-11-05: the market still funds a passed window; the end must win."""
    end = NOW + timedelta(seconds=60)
    for at in (end, end + timedelta(seconds=40)):
        v = shed_act.shed_verdict(_decision("funded", "net_positive", net=0.5), at, end)
        assert v.close and not v.proceed and v.why == "event_expired"
