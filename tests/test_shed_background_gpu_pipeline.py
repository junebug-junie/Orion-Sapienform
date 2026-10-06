"""shed_background_gpu from workspace winner to dispatch frame, through the REAL builders and gates.

Acceptance checks 2 (proposal bound to a <= 90 s winner with an eligibility snapshot), 3 (admitted by
the unchanged allocator floor, not through the visual-baseline lane) and 6 (``acted`` is never
terminal), plus the world-action master switch."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from orion.autonomy.allocator import allocate, candidate_from_dispatch
from orion.autonomy.cabinet_heat import (
    CABINET_NODE_ID, cabinet_heat_pressure, cabinet_warming_error, minute_mean, read_cabinet_heat,
)
from orion.autonomy.prediction import DEFAULT_PRIOR_VARIANCE
from orion.autonomy.self_shed import HardwareWatchView, bind_workspace_winner, evaluate_shed_eligibility
from orion.execution_dispatch.builder import build_execution_dispatch_frame
from orion.execution_dispatch.policy import load_execution_dispatch_policy
from orion.hardware_watch.rules import TempPoint
from orion.policy.builder import build_policy_decision_frame
from orion.policy.policy import load_substrate_policy
from orion.proposals.builder import WorkspaceWinnerContext, build_proposal_frame
from orion.proposals.policy import ProposalTemplateV1, load_proposal_policy
from orion.schemas.attention_salience import AttentionLoopOutcomeV1
from orion.substrate.attention.verdicts import TERMINAL_VERDICTS

from tests.test_proposal_frame_builder import _loaded_attention, _loaded_field

REPO = Path(__file__).resolve().parents[1]
POLICY = load_proposal_policy(REPO / "config/proposals/proposal_policy.v1.yaml")
SUBSTRATE = load_substrate_policy(REPO / "config/policy/substrate_policy.v1.yaml")
DISPATCH = load_execution_dispatch_policy(REPO / "config/execution_dispatch/execution_dispatch_policy.v1.yaml")
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


def test_world_candidates_never_push_field_candidates_out_of_the_frame():
    without = build_proposal_frame(field=_loaded_field(), attention=_loaded_attention(), policy=POLICY, now=NOW)
    with_world = _proposal_frame()
    field_ids = lambda f: [c.proposal_id for c in f.candidates if c.attention_winner is None]
    assert field_ids(with_world) == field_ids(without)


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


# --- the real pipeline ----------------------------------------------------------------------------

def _proposal_frame(eligibility=None, projection=None):
    ws = WorkspaceWinnerContext(projection=projection or _projection(), broadcast_log_id="broadcast-1",
                                eligibility={"shed_background_gpu": eligibility or _eligibility()})
    return build_proposal_frame(field=_loaded_field(), attention=_loaded_attention(), policy=POLICY, now=NOW, workspace=ws)


def test_check2_candidate_bound_to_the_winner_with_eligibility_snapshot():
    frame = _proposal_frame()
    shed = [c for c in frame.candidates if c.execution_intent.get("template") == "shed_background_gpu"]
    assert len(shed) == 1
    c = shed[0]
    assert c.attention_winner.open_loop_id == "open-loop-cab" and c.attention_winner.age_sec <= 90
    assert c.world_eligibility["eligible"] and c.world_eligibility["cabinet"]["thermal_state"] == "elevated"
    assert c.world_eligibility["cabinet"]["rise_c"] >= 0.5 and c.world_eligibility["holdback_fraction"] == 0.5
    assert c.world_eligibility["hardware_watch"]["open_incident_ids"] == []
    assert c.expected_signal == "cabinet_heat_pressure" and c.expected_direction == "decrease"
    assert "open_loop:open-loop-cab" in c.evidence_refs


def test_ineligible_world_is_recorded_on_the_frame_never_silently_dropped():
    frame = _proposal_frame(eligibility=_eligibility(background_granted=0, background_queued=0))
    assert not [c for c in frame.candidates if c.execution_intent.get("template") == "shed_background_gpu"]
    assert any(w.startswith("world_action_ineligible:shed_background_gpu:no_background_work") for w in frame.warnings)
    frame = _proposal_frame(projection=_projection(age_sec=200))
    assert any(w.startswith("winner_unbindable:stale") for w in frame.warnings)


def _dispatch(world_actions_allowed):
    proposal = _proposal_frame()
    policy_frame = build_policy_decision_frame(proposal_frame=proposal, policy=SUBSTRATE, now=NOW)
    decision = next(d for d in policy_frame.decisions if "shed_background_gpu" in d.proposal_id)
    assert decision.decision == "approved_self_reversible"
    return build_execution_dispatch_frame(policy_frame=policy_frame, proposal_frame=proposal, field_tick_id="tick_live",
                                          policy=DISPATCH, now=NOW, override_dispatch_mode="dispatch_read_only",
                                          world_actions_allowed=world_actions_allowed)


def test_master_switch_off_blocks_visibly():
    frame = _dispatch(None)
    blocked = [c for c in frame.blocked_candidates if c.dispatch_kind == "self_regulate"]
    assert blocked and blocked[0].blocked_by == ["world_actions_disabled"]
    frame = _dispatch(frozenset({"render_scene"}))
    assert [c.blocked_by for c in frame.blocked_candidates if c.dispatch_kind == "self_regulate"] == [["world_action_not_allowed"]]


def test_check3_allowed_world_action_is_prepared_and_clears_the_unchanged_floor():
    frame = _dispatch(frozenset({"shed_background_gpu"}))
    shed = next(c for c in frame.candidates if c.dispatch_kind == "self_regulate")
    assert shed.dispatch_status == "prepared_for_dispatch" and shed.visual_baseline is None
    assert shed.world_action["attention_winner"]["open_loop_id"] == "open-loop-cab"
    assert shed.expected_effect.signal_id == "cabinet_heat_pressure" and shed.expected_effect.cold_start
    cand = candidate_from_dispatch(dispatch_id=shed.dispatch_id, dispatch_kind=shed.dispatch_kind, target_id=shed.target_id,
                                   signal_id=shed.expected_effect.signal_id, claimed_direction="decrease",
                                   cell_variances_by_volume=[], cost_sec=5.0, cold_variance=DEFAULT_PRIOR_VARIANCE)
    alloc = allocate([cand], allowance_sec=3600.0, min_nats_per_sec=0.02)
    assert [a.dispatch_id for a in alloc.admitted] == [shed.dispatch_id]
    assert cand.cold_start or cand.nats_per_sec >= 0.02


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
