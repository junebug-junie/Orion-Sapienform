"""The visual-baseline schedule (orion/reverie/baseline.py) and the allocator's baseline lane.

2026-11-05 (ladder retirement PR 2): the schedule runs in orion-thought's painting trigger
(services/orion-thought/app/painting_self_start.py, tests there), not in the proposal arena: the
proposal builder emits no baseline candidate and refuses any producer that claims one. The
allocator's baseline lane below is unreachable now (nothing sends a baseline candidate to
dispatch) and retires with execution-dispatch in PR 6.
"""
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


def test_the_proposal_arena_never_carries_a_baseline_painting(enabled):
    """Fails on pre-PR-2 code, where build_proposal_frame took baseline_eligibility and turned the
    render_scene candidate into a scheduled painting."""
    import inspect

    assert "baseline_eligibility" not in inspect.signature(build_proposal_frame).parameters
    state = FieldStateV1(tick_id="quiet", generated_at=NOW)
    policy = load_proposal_policy(Path("config/proposals/proposal_policy.v1.yaml"))
    frame = build_proposal_frame(field=state, attention=None, policy=policy, now=NOW)
    assert all(c.visual_baseline is None for c in frame.candidates)
    assert all(c.execution_intent.get("template") != "render_scene" for c in frame.candidates)


def test_a_forged_baseline_from_any_producer_is_refused(enabled):
    state = FieldStateV1(tick_id="quiet", generated_at=NOW)
    policy = load_proposal_policy(Path("config/proposals/proposal_policy.v1.yaml"))
    from orion.schemas.proposal_frame import ProposalCandidateV1

    forged = ProposalCandidateV1(
        proposal_id="proposal:forged", proposal_kind="express", title="paint", description="forged",
        target_id="host:circe_gpu", target_kind="system", priority_score=0.9, urgency_score=0.9,
        confidence_score=0.9, risk_score=0.05, reversibility_score=1.0, proposed_effect="increase_observability",
        source="reverie_thought", visual_baseline=due(enabled))
    frame = build_proposal_frame(field=state, attention=None, policy=policy, now=NOW, external_candidates=[forged])
    assert all(c.proposal_id != "proposal:forged" for c in frame.candidates)
    assert any("untrusted" in w for w in frame.warnings)
