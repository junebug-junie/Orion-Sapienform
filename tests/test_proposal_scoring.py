from datetime import datetime, timezone
from pathlib import Path

from orion.field.pressure import field_pressures
from orion.proposals.policy import load_proposal_policy
from orion.proposals.scoring import (
    clamp01,
    proposal_priority,
    proposal_risk,
    template_match_score,
)
from orion.schemas.field_state import FieldStateV1

REPO = Path(__file__).resolve().parents[1]
POLICY = load_proposal_policy(REPO / "config" / "proposals" / "proposal_policy.v1.yaml")
NOW = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


def _pressures(**channel_values: float) -> dict[str, float]:
    """2026-07-22 (SelfStateV1 burn): builds real field_pressures() output from
    a synthetic FieldStateV1 instead of hand-setting SelfStateV1 dimension
    scores directly. Channel names, not dimension names -- e.g.
    execution_pressure=1.0 maps straight through (channel and dimension share
    a name for that one); resource pressure comes via the "pressure" channel."""
    field = FieldStateV1(
        generated_at=NOW,
        tick_id="tick_test",
        node_vectors={"node:test": channel_values},
    )
    return field_pressures(field)


def test_high_execution_pressure_raises_inspect_match() -> None:
    pressures = _pressures(execution_pressure=1.0)
    tmpl = POLICY.proposal_templates["inspect_execution_pressure"]
    match, _ = template_match_score(field_pressures=pressures, template=tmpl)
    low_pressures = _pressures(execution_pressure=0.1)
    low_match, _ = template_match_score(field_pressures=low_pressures, template=tmpl)
    assert match > low_match


def test_high_resource_pressure_raises_summarize_match() -> None:
    """field_intensity (the other dimension summarize_loaded_state scores on)
    is a composite SelfStateV1 dimension with no post-burn replacement -- it
    always reads 0.0 now (orion/field/pressure.py's module docstring). Only
    resource_pressure can move this template's match score post-burn."""
    high = _pressures(pressure=0.9)
    low = _pressures(pressure=0.1)
    tmpl = POLICY.proposal_templates["summarize_loaded_state"]
    high_match, _ = template_match_score(field_pressures=high, template=tmpl)
    low_match, _ = template_match_score(field_pressures=low, template=tmpl)
    assert high_match > low_match


def test_read_only_proposals_low_risk() -> None:
    pressures = _pressures(execution_pressure=1.0, reliability_pressure=0.8)
    tmpl = POLICY.proposal_templates["inspect_execution_pressure"]
    assert proposal_risk(base_risk=tmpl.base_risk, field_pressures=pressures, template=tmpl) <= 0.15


def test_policy_review_higher_risk() -> None:
    """agency_readiness (request_policy_review_for_action's other scoring
    dimension) is composite and gone post-burn -- reliability_pressure alone
    now drives the risk-bump comparison between these two template kinds."""
    pressures = _pressures(execution_pressure=1.0, reliability_pressure=0.8)
    review = POLICY.proposal_templates["request_policy_review_for_action"]
    inspect = POLICY.proposal_templates["inspect_execution_pressure"]
    assert proposal_risk(
        base_risk=review.base_risk, field_pressures=pressures, template=review
    ) > proposal_risk(
        base_risk=inspect.base_risk, field_pressures=pressures, template=inspect
    )


def test_scores_clamped() -> None:
    assert clamp01(2.0) == 1.0
    assert proposal_priority(
        base_priority=0.9,
        match_score=0.9,
        urgency=0.9,
        confidence=0.9,
    ) == 1.0
