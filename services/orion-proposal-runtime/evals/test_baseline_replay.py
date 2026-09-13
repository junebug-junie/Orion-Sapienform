"""Integrated pure-policy replay, with synthetic physical readings/receipts.

Exercises production scheduler/builders/allocator/thermal/feedback APIs. This is
not an HTTP, database, or diffusion smoke; thought's test_visual_activity.py
separately covers durable endpoint claims, thermal refusal, production and replay.
"""
import json
from datetime import datetime, timedelta, timezone

from orion.autonomy.allocator import candidate_from_dispatch, allocate
from orion.autonomy.thermal_gate import thermal_state
from orion.execution_dispatch.builder import build_execution_dispatch_frame
from orion.execution_dispatch.policy import load_execution_dispatch_policy
from orion.feedback.builder import build_feedback_frame
from orion.feedback.extractors import normalize_cortex_result_evidence
from orion.feedback.outcome_resolution import resolve_action_outcomes
from orion.feedback.policy import load_feedback_policy
from orion.policy.builder import build_policy_decision_frame
from orion.policy.policy import load_substrate_policy
from orion.proposals.builder import build_proposal_frame
from orion.proposals.policy import load_proposal_policy
from orion.proposals.scoring import PRESSURE_DIMENSIONS
from orion.reverie import baseline
from orion.schemas.field_state import FieldStateV1
from orion.schemas.reverie_visual import VisualActivityV1


def test_calm_capacity_thermal_feedback_production_and_restart(monkeypatch):
    now = datetime(2026, 9, 13, tzinfo=timezone.utc)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now

    monkeypatch.setattr(baseline, "datetime", Clock)
    scheduling = baseline.VisualBaselinePolicy(enabled=True)
    monkeypatch.setattr(baseline, "load_baseline_policy", lambda: scheduling)
    proposal_policy = load_proposal_policy("config/proposals/proposal_policy.v1.yaml")
    proposal_policy.limits.max_candidates = 1
    policy = load_substrate_policy("config/policy/substrate_policy.v1.yaml")
    dispatch_policy = load_execution_dispatch_policy("config/execution_dispatch/execution_dispatch_policy.v1.yaml")
    dispatch_policy.limits.max_dispatch_candidates = 1
    feedback_policy = load_feedback_policy("config/feedback/feedback_policy.v1.yaml")
    checkpoint = {}
    identities, outcomes, allocated = [], [], []
    previous_thermal = "normal"

    for allowance, temperature in ((0, 34), (60, 34), (60, 27)):
        field = FieldStateV1(tick_id=now.isoformat(), generated_at=now,
            node_vectors={"node:host": {"pressure": 0.1}},
            dimension_precision_ewma_n={d: 128 for d in PRESSURE_DIMENSIONS},
            dimension_precision_zscore={d: 0.0 for d in PRESSURE_DIMENSIONS},
            dimension_precision_ewma_var={d: 1.0 for d in PRESSURE_DIMENSIONS})
        activity = VisualActivityV1(observed_at=now, history_status="ok")
        need, checkpoint, reason = baseline.schedule(activity, checkpoint, now=now, policy=scheduling)
        assert reason == "visual_baseline_due"
        identities.append(need.need_id)
        ordinary = build_proposal_frame(field=field, attention=None, policy=proposal_policy, now=now)
        proposal = build_proposal_frame(field=field, attention=None, policy=proposal_policy,
            now=now, baseline_eligibility=need)
        assert proposal.action_warrant == ordinary.action_warrant
        assert proposal.action_warrant_gate == ordinary.action_warrant_gate == "below_threshold"
        assert not ordinary.candidates and len(proposal.candidates) == 1

        # A voluntary candidate coalesces with the due run and keeps provenance.
        voluntary = proposal.candidates[0].model_copy(update={"visual_baseline": None,
            "source": "reverie_thought", "thought_id": "selected-thought"})
        proposal = build_proposal_frame(field=field, attention=None, policy=proposal_policy,
            now=now, baseline_eligibility=need, external_candidates=[voluntary])
        assert len(proposal.candidates) == 1
        assert proposal.candidates[0].thought_id == "selected-thought"
        decision = build_policy_decision_frame(proposal_frame=proposal, policy=policy, now=now)
        assert decision.decisions[0].decision == "approved_express"
        dispatch = build_execution_dispatch_frame(policy_frame=decision, proposal_frame=proposal,
            field_tick_id=field.tick_id, policy=dispatch_policy, now=now)
        assert len(dispatch.candidates) == 1
        candidate = dispatch.candidates[0]
        assert candidate.expected_effect is None
        scored = candidate_from_dispatch(dispatch_id=candidate.dispatch_id,
            dispatch_kind=candidate.dispatch_kind, target_id=candidate.target_id,
            signal_id=None, claimed_direction=None, cell_variances_by_volume=[],
            cost_sec=45, cold_variance=0.25, visual_baseline=candidate.visual_baseline)
        allocation = allocate([scored], allowance_sec=allowance, min_nats_per_sec=1)
        allocated.append(allocation.spent_sec)
        result_rows = []
        if not allocation.admitted:
            assert allocation.refused[0][1] == "allowance_exhausted"
            candidate = candidate.model_copy(update={"dispatch_status": "blocked",
                "blocked_by": ["allocator:allowance_exhausted"]})
            dispatch = dispatch.model_copy(update={"candidates": [], "blocked_candidates": [candidate]})
            outcomes.append("allowance_exhausted")
        else:
            assert allocation.spent_sec == 45 and allocation.spent_sec <= allowance
            thermal = thermal_state(temp_c=temperature, age_sec=1, previous_state=previous_thermal)
            previous_thermal = thermal.state
            outcome = "produced" if thermal.allows_gpu_work else "deferred_thermal"
            outcomes.append(outcome)
            candidate = candidate.model_copy(update={"dispatch_status": "dispatched",
                "dispatched_at": now, "result_ref": "receipt:" + candidate.dispatch_id,
                "visual_outcome": outcome})
            dispatch = dispatch.model_copy(update={"candidates": [], "dispatched_candidates": [candidate],
                "dispatch_count": 1, "dispatch_attempted": True, "dispatch_mode": "dispatch_read_only"})
            result_rows = [normalize_cortex_result_evidence({"result_id": candidate.result_ref,
                "dispatch_id": candidate.dispatch_id, "status": "success", "visual_outcome": outcome})]
            assert result_rows[0]["visual_outcome"] == outcome
        feedback = build_feedback_frame(dispatch_frame=dispatch, policy_frame=decision,
            proposal_frame=proposal, field_before=field, field_after=field,
            cortex_results=result_rows, policy=feedback_policy, now=now)
        resolution = resolve_action_outcomes(dispatch_frame=dispatch, feedback_frame_id=feedback.frame_id,
            field_before=field, field_after=field, now=now,
            cortex_results=result_rows)
        assert not resolution.records and not resolution.posteriors
        if outcomes[-1] == "deferred_thermal":
            cortex = [o for o in feedback.observations if o.source_kind == "cortex_result"]
            assert cortex and all(o.outcome_kind == "deferred" for o in cortex)
        checkpoint = json.loads(json.dumps(checkpoint))  # persisted restart
        no_attempt, checkpoint, reason = baseline.schedule(activity, checkpoint,
            now=now + timedelta(seconds=1), policy=scheduling)
        assert no_attempt is None and reason == "visual_baseline_cooldown"
        if outcomes[-1] != "produced":
            now += timedelta(seconds=scheduling.retry_sec)

    assert len(set(identities)) == 1
    assert outcomes == ["allowance_exhausted", "deferred_thermal", "produced"]
    assert allocated == [0, 45, 45]
    production = VisualActivityV1(observed_at=now, history_status="ok", last_success_at=now,
        last_success_chain_id="acknowledged-production", last_success_sha256="b" * 64)
    no_attempt, checkpoint, reason = baseline.schedule(production, checkpoint, now=now, policy=scheduling)
    assert no_attempt is None and reason == "visual_baseline_not_due"
    assert "need_id" not in checkpoint

    # Many missed intervals produce one new pending need, never a catch-up list.
    now += timedelta(days=5)
    production.observed_at = now
    new_need, checkpoint, reason = baseline.schedule(production, checkpoint, now=now, policy=scheduling)
    assert reason == "visual_baseline_due" and new_need.need_id != identities[0]
    assert checkpoint["need_id"] == new_need.need_id
    # A voluntary production before the pending baseline runs invalidates it.
    production.last_success_at = now
    production.last_success_chain_id = "voluntary-production"
    production.last_success_sha256 = "c" * 64
    no_attempt, checkpoint, reason = baseline.schedule(production, checkpoint, now=now, policy=scheduling)
    assert no_attempt is None and reason == "visual_baseline_not_due"
    assert "need_id" not in checkpoint

