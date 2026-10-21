"""Thinking rung 3: rung-2 acts compete in the motor allocator on their own belief and cost.

Pins, each against real numbers from the live 48 h window ending 2026-10-21:
- a route that produces an act (render_scene -> thought.painting `make`) is scored on that act's
  EffectPosterior and the binary act learner's tau^2 -- before rung 3 it was refused `unmeasurable`
  on every tick (10,649 refusals in 48 h);
- the floor is untouched: the measured painting act (var 0.0083, n=29) at its measured 57.1 s is
  refused below_information_floor; a never-graded act is admitted once (existing cold-start rule);
- no typed cost fallback for an act (no measured cost -> no_cost_estimate), one in flight per act,
  and the kill switch restores the old scoring exactly;
- the per-tick randomized holdback only withholds ticks that can yield a control observation.
"""
from __future__ import annotations

import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

REPO = Path(__file__).resolve().parents[1]
SVC = REPO / "services" / "orion-execution-dispatch-runtime"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SVC))

import app.worker as worker_mod  # noqa: E402
from app.worker import ExecutionDispatchRuntimeWorker, tick_can_yield_control  # noqa: E402
from orion.autonomy.act_candidates import ActRef, act_allocator_candidate, act_expectation_reason  # noqa: E402
from orion.autonomy.allocator import Candidate, allocate, expected_information_gain_nats  # noqa: E402
from orion.autonomy.contrast import HOLDBACK_BLOCK_REASON  # noqa: E402
from orion.autonomy.prediction import (  # noqa: E402
    ACT_BINARY_OBSERVATION_VARIANCE,
    DEFAULT_OBSERVATION_VARIANCE,
    EffectPosterior,
)
from orion.execution_dispatch.policy import (  # noqa: E402
    ExecutionDispatchPolicyV1,
    load_execution_dispatch_policy,
)
from orion.schemas.action_prediction import ExpectedEffectV1  # noqa: E402
from orion.schemas.execution_dispatch_frame import (  # noqa: E402
    ExecutionDispatchCandidateV1,
    ExecutionDispatchFrameV1,
)

NOW = datetime(2026, 10, 21, 3, 0, tzinfo=timezone.utc)
FLOOR = 0.02  # ORION_DISPATCH_MIN_NATS_PER_SEC, unchanged
PAINT = ActRef(verb="make", actor="thought.painting", signal_id="artifact_produced")
PAINT_CELL = ("act:make", "thought.painting", "artifact_produced", 0)
# Live 2026-10-21: substrate_action_effect_posterior for the painting act, and the render_scene
# route's p50 settled latency over 30 days (331 success rows).
LIVE_PAINT = EffectPosterior(mean=0.017, variance=0.0083, n=29)
LIVE_COST = 57.1
RENDER_VERB = "skills.imagination.render_scene.v1"


# --- allocator arithmetic ------------------------------------------------------------------------

def test_existing_candidates_keep_their_tau2() -> None:
    c = Candidate(dispatch_id="d", dispatch_kind="inspect", target_id="t", posterior_variance=0.01, cost_sec=5.0)
    assert c.observation_variance == DEFAULT_OBSERVATION_VARIANCE
    assert c.expected_nats == pytest.approx(0.5 * math.log(1 + 0.01 / 0.04))


def test_act_candidate_is_scored_on_the_act_learner() -> None:
    c = act_allocator_candidate(dispatch_id="d", dispatch_kind="express", target_id="host:circe_gpu",
                                ref=PAINT, posteriors={PAINT_CELL: LIVE_PAINT}, cost_sec=LIVE_COST)
    assert c.observation_variance == ACT_BINARY_OBSERVATION_VARIANCE
    assert c.measurable and not c.cold_start
    assert c.expected_nats == pytest.approx(expected_information_gain_nats(0.0083, 0.25))


def test_measured_painting_is_refused_at_the_unchanged_floor() -> None:
    c = act_allocator_candidate(dispatch_id="d", dispatch_kind="express", target_id="host:circe_gpu",
                                ref=PAINT, posteriors={PAINT_CELL: LIVE_PAINT}, cost_sec=LIVE_COST)
    out = allocate([c], allowance_sec=1e6, min_nats_per_sec=FLOOR)
    assert out.admitted == () and out.refusals_by_reason() == {"below_information_floor": 1}


def test_never_graded_act_is_admitted_once_by_the_existing_cold_start_rule() -> None:
    c = act_allocator_candidate(dispatch_id="d", dispatch_kind="express", target_id="host:circe_gpu",
                                ref=PAINT, posteriors={}, cost_sec=LIVE_COST)
    assert c.cold_start
    assert [a.dispatch_id for a in allocate([c], allowance_sec=1e6, min_nats_per_sec=FLOOR).admitted] == ["d"]


def test_no_measured_cost_is_refused_not_priced() -> None:
    c = act_allocator_candidate(dispatch_id="d", dispatch_kind="express", target_id="host:circe_gpu",
                                ref=PAINT, posteriors={PAINT_CELL: LIVE_PAINT}, cost_sec=None)
    assert allocate([c], allowance_sec=1e6, min_nats_per_sec=FLOOR).refusals_by_reason() == {"no_cost_estimate": 1}


def test_expectation_reason_carries_the_scored_numbers() -> None:
    line = act_expectation_reason(PAINT, {PAINT_CELL: LIVE_PAINT}, LIVE_COST)
    assert line.startswith("act_expectation:make:thought.painting:artifact_produced:")
    assert "n=29" in line and "cold=false" in line and "cost_sec=57.1" in line


# --- policy --------------------------------------------------------------------------------------

def test_render_scene_route_declares_the_painting_act() -> None:
    policy = load_execution_dispatch_policy(REPO / "config/execution_dispatch/execution_dispatch_policy.v1.yaml")
    ref = policy.act_refs_by_cortex_verb()[RENDER_VERB]
    assert (ref.verb, ref.actor, ref.signal_id) == ("make", "thought.painting", "artifact_produced")


def test_one_dispatch_one_act() -> None:
    route = {"dispatch_kind": "express", "cortex_verb": RENDER_VERB, "allowed_scope": "express_bounded"}
    with pytest.raises(ValueError, match="one dispatch, one act"):
        ExecutionDispatchPolicyV1.model_validate({
            "proposal_kind_to_cortex": {"express": {**route, "act": {"verb": "make", "actor": "a", "signal_id": "artifact_produced"}}},
            "template_to_cortex": {"render_scene": {**route, "act": {"verb": "make", "actor": "b", "signal_id": "artifact_produced"}}},
        })


def test_an_act_signal_outside_the_grammar_is_refused_at_load() -> None:
    with pytest.raises(ValueError):
        ExecutionDispatchPolicyV1.model_validate({"template_to_cortex": {"x": {
            "dispatch_kind": "express", "cortex_verb": "v", "allowed_scope": "express_bounded",
            "act": {"verb": "make", "actor": "a", "signal_id": "cabinet_heat_pressure"}}}})


# --- worker: the preview the enforcing allocator acts on ----------------------------------------

def _worker(monkeypatch, *, enabled: bool = True, in_flight: bool = False, cost: float | None = LIVE_COST,
            posteriors=None) -> ExecutionDispatchRuntimeWorker:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    import app.settings as settings_mod

    settings_mod._settings = None
    worker = ExecutionDispatchRuntimeWorker()
    monkeypatch.setattr(worker._settings, "orion_dispatch_act_candidates_enabled", enabled, raising=False)
    store = MagicMock()
    # The 24 h template window has nothing (no render in 48 h); only the act window has a cost.
    store.load_action_cost_estimates.side_effect = lambda lookback_hours=24: (
        {("express", "host:circe_gpu"): cost} if lookback_hours > 24 and cost is not None else {})
    store.load_effect_posteriors.return_value = {PAINT_CELL: LIVE_PAINT} if posteriors is None else posteriors
    store.act_in_flight.return_value = in_flight
    worker._store = store
    return worker


def _render(dispatch_id: str = "dispatch:proposal:render_scene:t1") -> ExecutionDispatchCandidateV1:
    return ExecutionDispatchCandidateV1(
        dispatch_id=dispatch_id, source_decision_id=f"pd:{dispatch_id}", source_proposal_id=f"p:{dispatch_id}",
        dispatch_status="prepared_for_dispatch", dispatch_mode="dispatch_read_only", dispatch_kind="express",
        target_id="host:circe_gpu", target_kind="system", cortex_verb=RENDER_VERB, cortex_mode="brain",
        risk_score=0.05, confidence_score=0.9)


def _frame(*cands) -> ExecutionDispatchFrameV1:
    return ExecutionDispatchFrameV1(
        frame_id="execution.dispatch.frame:test", source_policy_frame_id="policy.frame:test",
        source_proposal_frame_id="proposal.frame:test", generated_at=NOW,
        dispatch_mode="dispatch_read_only", candidates=list(cands))


MOTOR = SimpleNamespace(remaining_sec=1e6)


def test_render_scene_is_scored_on_the_act_not_refused_unmeasurable(monkeypatch) -> None:
    """Fails on pre-rung-3 code: render_scene declares no field signal, so it was `unmeasurable`."""
    worker = _worker(monkeypatch)
    allocation = worker._log_allocator_preview(_frame(_render()), MOTOR)
    assert allocation.refusals_by_reason() == {"below_information_floor": 1}
    assert worker._last_act_reasons["dispatch:proposal:render_scene:t1"].startswith("act_expectation:make:")


def test_kill_switch_restores_the_old_scoring(monkeypatch) -> None:
    worker = _worker(monkeypatch, enabled=False)
    assert worker._log_allocator_preview(_frame(_render()), MOTOR).refusals_by_reason() == {"unmeasurable": 1}
    worker._store.act_in_flight.assert_not_called()


def test_act_with_no_measured_cost_never_borrows_the_typed_fallback(monkeypatch) -> None:
    worker = _worker(monkeypatch, cost=None)
    assert worker._log_allocator_preview(_frame(_render()), MOTOR).refusals_by_reason() == {"no_cost_estimate": 1}


def test_cold_act_is_admitted_but_only_one_in_flight(monkeypatch) -> None:
    assert [c.dispatch_id for c in _worker(monkeypatch, posteriors={})._log_allocator_preview(
        _frame(_render()), MOTOR).admitted] == ["dispatch:proposal:render_scene:t1"]
    blocked = _worker(monkeypatch, posteriors={}, in_flight=True)._log_allocator_preview(_frame(_render()), MOTOR)
    assert blocked.admitted == () and blocked.refusals_by_reason() == {"act_in_flight": 1}


def test_unreadable_in_flight_ledger_fails_closed(monkeypatch) -> None:
    worker = _worker(monkeypatch, posteriors={})
    worker._store.act_in_flight.side_effect = RuntimeError("db down")
    assert worker._log_allocator_preview(_frame(_render()), MOTOR).refusals_by_reason() == {"act_in_flight": 1}


# --- randomized holdback: only ticks that can yield a control ------------------------------------

def _field_candidate(dispatch_id: str = "dispatch:proposal:prune_build_cache:t1") -> ExecutionDispatchCandidateV1:
    return ExecutionDispatchCandidateV1(
        dispatch_id=dispatch_id, source_decision_id=f"pd:{dispatch_id}", source_proposal_id=f"p:{dispatch_id}",
        dispatch_status="prepared_for_dispatch", dispatch_mode="dispatch_read_only", dispatch_kind="maintain",
        target_id="host:docker_images", target_kind="host", cortex_verb="skills.runtime.builder_prune.v1",
        cortex_mode="brain", risk_score=0.05, confidence_score=0.9,
        expected_effect=ExpectedEffectV1(signal_id="resource_pressure", direction="decrease", predicted_delta=0.0,
                                         predictor_variance=0.25, predictor_n=0, cold_start=True))


def test_tick_can_yield_control_only_with_a_field_signal() -> None:
    assert tick_can_yield_control([_field_candidate()])
    assert not tick_can_yield_control([_render()])
    assert not tick_can_yield_control([_field_candidate().model_copy(update={"world_action": {"template": "shed"}})])


def _send_ready_worker(monkeypatch, fraction: float):
    worker = _worker(monkeypatch)
    monkeypatch.setattr(worker._settings, "orion_dispatch_holdback_fraction", fraction, raising=False)
    worker._derive_motor_budget = MagicMock(return_value=None)        # no allocation in the way
    worker._derive_daily_risk_cap = MagicMock(return_value=(1e9, {}))
    worker._store.sum_risk_dispatched_today.return_value = 0.0
    fake_bus = MagicMock(connect=AsyncMock(), close=AsyncMock(), publish=AsyncMock())
    monkeypatch.setattr(worker_mod, "OrionBusAsync", MagicMock(return_value=fake_bus))
    monkeypatch.setattr(worker_mod.RPC_HEALTH_SINK, "absorb_bus", MagicMock())

    async def _sent(client, bus, frame, c):
        return c.model_copy(update={"dispatch_status": "dispatched", "dispatched_at": NOW})

    worker._send_one = _sent
    return worker


@pytest.mark.asyncio
async def test_field_signal_tick_records_both_arms(monkeypatch) -> None:
    """Fraction 0.2 (rung 3, design D3): withheld ticks are recorded as randomized_holdback blocks
    (the control the feedback runtime turns into control cells), sent ticks as dispatched."""
    worker = _send_ready_worker(monkeypatch, 0.2)
    draws = iter([0.1, 0.9])   # first tick drawn into control, second treated
    monkeypatch.setattr(worker_mod.random, "random", lambda: next(draws))
    control = await worker._send_prepared_candidates(_frame(_field_candidate("dispatch:a")))
    treated = await worker._send_prepared_candidates(_frame(_field_candidate("dispatch:b")))
    assert [c.blocked_by for c in control.blocked_candidates] == [[HOLDBACK_BLOCK_REASON]]
    assert control.dispatched_candidates == []
    assert [c.dispatch_id for c in treated.dispatched_candidates] == ["dispatch:b"]


@pytest.mark.asyncio
async def test_a_tick_with_nothing_to_learn_is_never_held_back(monkeypatch) -> None:
    """A held-back render is re-proposed seconds later: a delay, not a control. Fails on old code."""
    worker = _send_ready_worker(monkeypatch, 0.5)
    monkeypatch.setattr(worker_mod.random, "random", lambda: 0.0)
    worker._log_allocator_preview = MagicMock(return_value=None)
    out = await worker._send_prepared_candidates(_frame(_render()))
    assert [c.dispatch_id for c in out.dispatched_candidates] == ["dispatch:proposal:render_scene:t1"]
    assert not any(HOLDBACK_BLOCK_REASON in c.blocked_by for c in out.blocked_candidates)


def test_visual_baseline_lane_is_unchanged_even_with_no_act_cost(monkeypatch) -> None:
    """A scheduled baseline painting keeps its old cost path: a quiet month with no measured render
    cost must never refuse it no_cost_estimate. Its act numbers are still recorded."""
    from orion.schemas.reverie_visual import VisualBaselineEligibilityV1

    worker = _worker(monkeypatch, cost=None)
    baseline = VisualBaselineEligibilityV1(need_id="visual_baseline", observed_at=NOW, due_at=NOW,
                                           policy_id="visual_baseline.v1")
    item = ExecutionDispatchCandidateV1.model_validate({**_render().model_dump(), "visual_baseline": baseline.model_dump()})
    monkeypatch.setattr("orion.reverie.baseline.validate_eligibility", lambda *a, **k: None)
    allocation = worker._log_allocator_preview(_frame(item), MOTOR)
    assert [c.dispatch_id for c in allocation.admitted] == [item.dispatch_id]
    assert worker._last_act_reasons[item.dispatch_id].startswith("act_expectation:make:")
    worker._store.act_in_flight.assert_not_called()
