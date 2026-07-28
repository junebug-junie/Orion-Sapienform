from datetime import datetime, timezone
from pathlib import Path

from orion.field.pressure import field_pressures
from orion.proposals.policy import load_proposal_policy
from orion.proposals.scoring import (
    DIMENSION_PRECISION_EWMA_MIN_SAMPLES,
    DIMENSION_PRECISION_ZSCORE_SATURATION,
    clamp01,
    dimension_confidence,
    proposal_confidence,
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


# 2026-07-28 precision-weighted confidence fix (docs/superpowers/specs/2026-
# 07-28-precision-weighted-proposal-scoring-design.md). dimension_confidence()
# used to be a binary presence flag with no state; it now reads a per-
# dimension EWMA baseline persisted on FieldStateV1.


def test_dimension_confidence_zero_when_dimension_absent_this_tick() -> None:
    """Preserves the old binary flag's "no reading this tick == no
    confidence" semantics, regardless of any stale baseline that might exist
    on the field from a previous tick."""
    field = FieldStateV1(
        generated_at=NOW,
        tick_id="tick_test",
        dimension_precision_ewma_n={"execution_pressure": 50},
        dimension_precision_zscore={"execution_pressure": 0.0},
    )
    assert dimension_confidence(field, {}, "execution_pressure") == 0.0


def test_dimension_confidence_zero_during_cold_start() -> None:
    """Fewer than DIMENSION_PRECISION_EWMA_MIN_SAMPLES real observations
    absorbed into the baseline -- an early z-score is not reliable (hand-
    verified, see that constant's own comment) and must not be reported as
    a real confidence reading."""
    field = FieldStateV1(
        generated_at=NOW,
        tick_id="tick_test",
        dimension_precision_ewma_n={"execution_pressure": DIMENSION_PRECISION_EWMA_MIN_SAMPLES - 1},
        dimension_precision_zscore={"execution_pressure": 0.1},
    )
    pressures = {"execution_pressure": 0.5}
    assert dimension_confidence(field, pressures, "execution_pressure") == 0.0


def test_dimension_confidence_high_when_reading_matches_baseline() -> None:
    """A z-score near 0 (this tick matches this dimension's own recent
    normal) -> confidence near 1.0."""
    field = FieldStateV1(
        generated_at=NOW,
        tick_id="tick_test",
        dimension_precision_ewma_n={"execution_pressure": DIMENSION_PRECISION_EWMA_MIN_SAMPLES},
        dimension_precision_zscore={"execution_pressure": 0.0},
    )
    pressures = {"execution_pressure": 0.5}
    assert dimension_confidence(field, pressures, "execution_pressure") == 1.0


def test_dimension_confidence_low_when_reading_is_a_real_surprise() -> None:
    """A z-score at or beyond DIMENSION_PRECISION_ZSCORE_SATURATION (a real
    surprise relative to this dimension's own recent trajectory) ->
    confidence 0.0, clamped rather than going negative."""
    field = FieldStateV1(
        generated_at=NOW,
        tick_id="tick_test",
        dimension_precision_ewma_n={"execution_pressure": DIMENSION_PRECISION_EWMA_MIN_SAMPLES},
        dimension_precision_zscore={"execution_pressure": DIMENSION_PRECISION_ZSCORE_SATURATION * 5},
    )
    pressures = {"execution_pressure": 0.9}
    assert dimension_confidence(field, pressures, "execution_pressure") == 0.0


def test_dimension_confidence_scales_between_calm_and_surprising() -> None:
    """Monotonic: a bigger real deviation from baseline yields a lower
    confidence, not a step function."""
    field_small_z = FieldStateV1(
        generated_at=NOW,
        tick_id="tick_a",
        dimension_precision_ewma_n={"execution_pressure": DIMENSION_PRECISION_EWMA_MIN_SAMPLES},
        dimension_precision_zscore={"execution_pressure": 0.5},
    )
    field_big_z = FieldStateV1(
        generated_at=NOW,
        tick_id="tick_b",
        dimension_precision_ewma_n={"execution_pressure": DIMENSION_PRECISION_EWMA_MIN_SAMPLES},
        dimension_precision_zscore={"execution_pressure": 2.0},
    )
    pressures = {"execution_pressure": 0.5}
    conf_small = dimension_confidence(field_small_z, pressures, "execution_pressure")
    conf_big = dimension_confidence(field_big_z, pressures, "execution_pressure")
    assert 0.0 < conf_big < conf_small < 1.0


def test_proposal_confidence_averages_real_dimension_confidences() -> None:
    template = POLICY.proposal_templates["inspect_execution_pressure"]
    assert "execution_pressure" in template.dimensions
    field = FieldStateV1(
        generated_at=NOW,
        tick_id="tick_test",
        dimension_precision_ewma_n={"execution_pressure": DIMENSION_PRECISION_EWMA_MIN_SAMPLES},
        dimension_precision_zscore={"execution_pressure": 0.0},
    )
    pressures = {"execution_pressure": 0.6}
    assert proposal_confidence(field=field, field_pressures=pressures, template=template) == 1.0


def test_proposal_confidence_zero_for_cold_start_field() -> None:
    """A brand-new FieldStateV1 (e.g. right after a schema upgrade, or a
    fresh deploy with no accumulated history yet) has empty precision dicts
    -- proposal_confidence() must degrade to 0.0, not raise or fabricate a
    default confidence."""
    template = POLICY.proposal_templates["inspect_execution_pressure"]
    field = FieldStateV1(generated_at=NOW, tick_id="tick_fresh")
    pressures = {"execution_pressure": 0.6}
    assert proposal_confidence(field=field, field_pressures=pressures, template=template) == 0.0
