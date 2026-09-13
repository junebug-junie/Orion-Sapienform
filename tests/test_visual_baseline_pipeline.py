from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest
from orion.reverie import baseline as b
from orion.schemas.reverie_visual import VisualActivityV1
from orion.schemas.field_state import FieldStateV1
from orion.proposals.builder import build_proposal_frame
from orion.proposals.policy import load_proposal_policy
from orion.autonomy.allocator import Candidate, allocate

NOW = datetime.now(timezone.utc)

@pytest.fixture
def enabled(monkeypatch):
    policy = b.VisualBaselinePolicy(enabled=True)
    monkeypatch.setattr(b, "load_baseline_policy", lambda: policy)
    return policy

def due(policy, now=NOW):
    activity = VisualActivityV1(observed_at=now, history_status="ok")
    return b.schedule(activity, {}, now=now, policy=policy)[0]

@pytest.mark.parametrize("live", [False, True])
def test_baseline_survives_calm_and_missing_warrant(enabled, live):
    state = FieldStateV1(tick_id="quiet", generated_at=NOW)
    if live:
        from orion.proposals.scoring import PRESSURE_DIMENSIONS
        state = state.model_copy(update={
            "dimension_precision_ewma_n": {d: 128 for d in PRESSURE_DIMENSIONS},
            "dimension_precision_zscore": {d: 0.0 for d in PRESSURE_DIMENSIONS},
            "dimension_precision_ewma_var": {d: 1.0 for d in PRESSURE_DIMENSIONS},
        })
    policy = load_proposal_policy(Path("config/proposals/proposal_policy.v1.yaml"))
    policy.limits.max_candidates = 1
    original = build_proposal_frame(field=state, attention=None, policy=policy, now=NOW)
    frame = build_proposal_frame(field=state, attention=None, policy=policy, now=NOW, baseline_eligibility=due(enabled))
    assert frame.action_warrant == original.action_warrant
    assert frame.action_warrant_gate == original.action_warrant_gate
    assert len(frame.candidates) == 1
    candidate = frame.candidates[0]
    assert candidate.visual_baseline and candidate.expected_signal is None
    forged = build_proposal_frame(field=state, attention=None, policy=policy, now=NOW, external_candidates=[candidate])
    assert not forged.candidates
    assert any("untrusted" in warning for warning in forged.warnings)

@pytest.mark.parametrize("cost,allowance,reason", [(None,100,"no_cost_estimate"),(float("nan"),100,"no_cost_estimate"),(-2,100,"no_cost_estimate"),(20,10,"allowance_exhausted"),(20,100,None)])
def test_baseline_cost_gate(enabled, cost, allowance, reason):
    candidate = Candidate("v", "express", "host:circe_gpu", None, cost, visual_baseline=due(enabled))
    result = allocate([candidate], allowance_sec=allowance, min_nats_per_sec=100)
    if reason:
        assert result.refused[0][1] == reason
        assert result.spent_sec == 0
    else:
        assert result.admitted == (candidate,)
        assert result.spent_sec == cost

def test_baseline_spends_before_information_and_only_once(enabled):
    visual = Candidate("v", "express", "host:circe_gpu", None, 20, visual_baseline=due(enabled))
    ordinary = Candidate("o", "inspect", "host:x", 1, 10)
    result = allocate([ordinary, visual, visual], allowance_sec=25, min_nats_per_sec=0)
    assert result.admitted == (visual,)
    assert result.spent_sec == 20

def test_checkpoint_restart_cooldown_voluntary_success_and_missed_intervals(enabled):
    activity = VisualActivityV1(observed_at=NOW, history_status="ok")
    first, checkpoint, _ = b.schedule(activity, {}, now=NOW, policy=enabled)
    import json
    checkpoint = json.loads(json.dumps(checkpoint))
    again, checkpoint, reason = b.schedule(activity, checkpoint, now=NOW+timedelta(seconds=1), policy=enabled)
    assert again is None and reason == "visual_baseline_cooldown"
    later = NOW+timedelta(days=20)
    activity.observed_at = later
    next_need, checkpoint, _ = b.schedule(activity, checkpoint, now=later, policy=enabled)
    assert next_need.need_id == first.need_id
    activity.last_success_at = later
    activity.last_success_chain_id = "voluntary"
    activity.last_success_sha256 = "a"*64
    complete, checkpoint, reason = b.schedule(activity, checkpoint, now=later, policy=enabled)
    assert complete is None and reason == "visual_baseline_not_due"
    assert "need_id" not in checkpoint

def test_disabled_stale_and_invalid_route_fail_closed(enabled):
    need = due(enabled)
    assert b.validate_eligibility(need, policy=b.VisualBaselinePolicy()) == "visual_baseline_disabled"
    assert b.validate_eligibility(need, policy=enabled, now=NOW+timedelta(seconds=61)) == "visual_baseline_stale"
    assert b.validate_eligibility(need, policy=enabled, target_id="host:other") == "visual_baseline_route_mismatch"

def test_policy_dispatch_identity_and_scope_gates(enabled):
    from orion.policy.builder import build_policy_decision_frame
    from orion.policy.policy import load_substrate_policy
    from orion.execution_dispatch.builder import build_execution_dispatch_frame
    from orion.execution_dispatch.policy import load_execution_dispatch_policy
    state = FieldStateV1(tick_id="quiet", generated_at=NOW)
    proposal = build_proposal_frame(field=state, attention=None,
        policy=load_proposal_policy("config/proposals/proposal_policy.v1.yaml"), now=NOW,
        baseline_eligibility=due(enabled))
    proposal.candidates[0].confidence_score = 0.9
    decisions = build_policy_decision_frame(proposal_frame=proposal,
        policy=load_substrate_policy("config/policy/substrate_policy.v1.yaml"), now=NOW)
    decision = decisions.decisions[0]
    assert decision.visual_baseline == proposal.candidates[0].visual_baseline
    assert decision.decision == "approved_express"
    dispatch_policy = load_execution_dispatch_policy("config/execution_dispatch/execution_dispatch_policy.v1.yaml")
    dispatch_policy.limits.max_dispatch_candidates = 1
    dispatch = build_execution_dispatch_frame(policy_frame=decisions, proposal_frame=proposal,
        field_tick_id=state.tick_id, policy=dispatch_policy, now=NOW)
    candidate = dispatch.candidates[0]
    assert candidate.expected_effect is None
    args = candidate.request_envelope["context"]["skill_args"]
    assert args["dispatch_id"] == candidate.dispatch_id
    assert args["visual_baseline"]["need_id"] == due(enabled).need_id
    decision.allowed_scope = "none"
    denied = build_execution_dispatch_frame(policy_frame=decisions, proposal_frame=proposal,
        field_tick_id=state.tick_id, policy=dispatch_policy, now=NOW)
    assert denied.blocked_candidates[0].visual_baseline == candidate.visual_baseline
    assert not denied.candidates

def test_voluntary_coalesces_without_losing_source(enabled):
    state = FieldStateV1(tick_id="quiet", generated_at=NOW)
    policy = load_proposal_policy("config/proposals/proposal_policy.v1.yaml")
    first = build_proposal_frame(field=state, attention=None, policy=policy, now=NOW, baseline_eligibility=due(enabled))
    voluntary = first.candidates[0].model_copy(update={"visual_baseline": None,
        "proposal_id": "proposal:render_scene:voluntary", "source": "reverie_thought", "thought_id": "source-thought"})
    result = build_proposal_frame(field=state, attention=None, policy=policy, now=NOW,
        baseline_eligibility=due(enabled), external_candidates=[voluntary])
    assert len(result.candidates) == 1
    assert result.candidates[0].thought_id == "source-thought"
    assert result.candidates[0].source == "reverie_thought"
    assert result.candidates[0].visual_baseline is not None


def test_invalid_baseline_policy_decision_is_recorded(enabled):
    from orion.policy.builder import build_policy_decision_frame
    from orion.policy.policy import load_substrate_policy
    state = FieldStateV1(tick_id="quiet", generated_at=NOW)
    proposal = build_proposal_frame(field=state, attention=None,
        policy=load_proposal_policy("config/proposals/proposal_policy.v1.yaml"), now=NOW,
        baseline_eligibility=due(enabled))
    proposal.candidates[0].visual_baseline.policy_id = "forged"
    decisions = build_policy_decision_frame(proposal_frame=proposal,
        policy=load_substrate_policy("config/policy/substrate_policy.v1.yaml"), now=NOW)
    assert decisions.decisions[0].decision == "rejected"
    assert "visual_baseline_policy_mismatch" in decisions.decisions[0].blocked_by
