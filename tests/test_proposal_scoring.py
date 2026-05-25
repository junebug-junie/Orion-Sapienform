from datetime import datetime, timezone
from pathlib import Path

from orion.proposals.policy import load_proposal_policy
from orion.proposals.scoring import (
    clamp01,
    proposal_priority,
    proposal_risk,
    template_match_score,
)
from orion.schemas.self_state import SelfStateDimensionV1, SelfStateV1

REPO = Path(__file__).resolve().parents[1]
POLICY = load_proposal_policy(REPO / "config" / "proposals" / "proposal_policy.v1.yaml")
NOW = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


def _self_state(**dim_scores: float) -> SelfStateV1:
    dimensions = {
        dim_id: SelfStateDimensionV1(
            dimension_id=dim_id,
            score=score,
            confidence=0.9,
        )
        for dim_id, score in dim_scores.items()
    }
    return SelfStateV1(
        self_state_id="self.state:test:frame:policy",
        generated_at=NOW,
        source_field_tick_id="tick_test",
        source_field_generated_at=NOW,
        source_attention_frame_id="frame_test",
        source_attention_generated_at=NOW,
        overall_condition="loaded",
        overall_intensity=0.65,
        overall_confidence=0.9,
        dimensions=dimensions,
    )


def test_high_execution_pressure_raises_inspect_match() -> None:
    state = _self_state(execution_pressure=1.0)
    tmpl = POLICY.proposal_templates["inspect_execution_pressure"]
    match, _ = template_match_score(self_state=state, template=tmpl)
    low_state = _self_state(execution_pressure=0.1)
    low_match, _ = template_match_score(self_state=low_state, template=tmpl)
    assert match > low_match


def test_high_field_resource_raises_summarize_match() -> None:
    state = _self_state(field_intensity=0.9, resource_pressure=0.9)
    tmpl = POLICY.proposal_templates["summarize_loaded_state"]
    match, _ = template_match_score(self_state=state, template=tmpl)
    assert match >= 0.4


def test_read_only_proposals_low_risk() -> None:
    state = _self_state(execution_pressure=1.0, reliability_pressure=0.8)
    tmpl = POLICY.proposal_templates["inspect_execution_pressure"]
    assert proposal_risk(base_risk=tmpl.base_risk, self_state=state, template=tmpl) <= 0.15


def test_policy_review_higher_risk() -> None:
    state = _self_state(agency_readiness=0.8, execution_pressure=1.0)
    review = POLICY.proposal_templates["request_policy_review_for_action"]
    inspect = POLICY.proposal_templates["inspect_execution_pressure"]
    assert proposal_risk(
        base_risk=review.base_risk, self_state=state, template=review
    ) > proposal_risk(
        base_risk=inspect.base_risk, self_state=state, template=inspect
    )


def test_scores_clamped() -> None:
    assert clamp01(2.0) == 1.0
    assert proposal_priority(
        base_priority=0.9,
        match_score=0.9,
        urgency=0.9,
        confidence=0.9,
    ) == 1.0
