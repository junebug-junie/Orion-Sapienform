from datetime import datetime, timezone
from pathlib import Path

from orion.bus.ewma import compute_ewma_update
from orion.field.pressure import field_pressures
from orion.proposals.policy import load_proposal_policy
from orion.proposals.scoring import (
    DIMENSION_PRECISION_EWMA_ALPHA,
    DIMENSION_PRECISION_EWMA_MIN_SAMPLES,
    DIMENSION_PRECISION_MIN_VARIANCE,
    DIMENSION_PRECISION_ZSCORE_SATURATION,
    PRESSURE_DIMENSIONS,
    clamp01,
    dimension_confidence,
    proposal_confidence,
    proposal_priority,
    proposal_risk,
    proposal_urgency,
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
    """summarize_loaded_state used to also score on field_intensity, a
    composite SelfStateV1 dimension with no post-burn replacement (always
    0.0). Removed from the template's own `dimensions` 2026-07-30 (config/
    proposals/proposal_policy.v1.yaml) -- resource_pressure is now its only,
    real scoring dimension."""
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
    """request_policy_review_for_action used to also score on
    agency_readiness, composite and gone post-burn. Removed from the
    template's own `dimensions` 2026-07-30 -- execution_pressure is now its
    only, real scoring dimension. `proposal_risk()` doesn't read
    `dimensions` at all (only `reliability_pressure` directly and
    `template.kind`/`required_policy_gate`), so this test's own risk-bump
    comparison was never affected by that dimension either way."""
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


def test_proposal_confidence_no_longer_capped_by_a_dead_dimension() -> None:
    """Regression test (2026-07-30): summarize_loaded_state used to also
    score on field_intensity, a composite dimension field_pressures() never
    produces -- dimension_confidence() returns 0.0 PERMANENTLY for a
    dimension absent from field_pressures this tick, and
    proposal_confidence() averages across a template's dimensions, so this
    template's real confidence was silently halved forever (capped at 0.5
    even with a perfect real resource_pressure reading). field_intensity was
    removed from the template's own `dimensions`
    (config/proposals/proposal_policy.v1.yaml) -- confidence should now
    equal resource_pressure's own real confidence directly, not half of it."""
    template = POLICY.proposal_templates["summarize_loaded_state"]
    assert template.dimensions == {"resource_pressure": 0.40}
    field = FieldStateV1(
        generated_at=NOW,
        tick_id="tick_test",
        dimension_precision_ewma_n={"resource_pressure": DIMENSION_PRECISION_EWMA_MIN_SAMPLES},
        dimension_precision_zscore={"resource_pressure": 0.0},
    )
    pressures = {"resource_pressure": 0.6}
    assert proposal_confidence(field=field, field_pressures=pressures, template=template) == 1.0


def test_proposal_urgency_wakes_up_once_the_dead_only_dimension_is_removed() -> None:
    """Regression test (2026-07-30): inspect_bus_channel_catalog used to
    score ONLY on contract_pressure, a dimension field_pressures() never
    produces. proposal_urgency()'s own dimension filter
    (`dim_id in PRESSURE_DIMENSIONS or dim_id.endswith("_pressure")`) let
    "contract_pressure" pass through on the suffix check alone, contributing
    a permanent dimension_score() of 0.0 -- a non-empty `scores` list, so
    proposal_urgency()'s real all-4-core-dimensions fallback (`if not
    scores`) never triggered, and urgency was permanently 0.0 regardless of
    real field pressure. Now that the template's `dimensions` is honestly
    empty, the fallback correctly triggers and urgency reflects the real
    field."""
    template = POLICY.proposal_templates["inspect_bus_channel_catalog"]
    assert template.dimensions == {}
    quiet = proposal_urgency(field_pressures={"execution_pressure": 0.05}, template=template)
    loud = proposal_urgency(field_pressures={"execution_pressure": 0.95}, template=template)
    assert quiet < loud
    assert loud > 0.5


# Real historical max per dimension, scripts/analysis/measure_proposal_
# dimension_variance.py's 16h real substrate_field_state window, 2026-07-28.
_REAL_MEASURED_MAX = {
    "execution_pressure": 0.392273,
    "resource_pressure": 0.9,
    "reliability_pressure": 0.9,
    "reasoning_pressure": 0.045,
}


def test_cold_start_min_samples_sufficient_across_jump_magnitude_sweep() -> None:
    """Regression guard for DIMENSION_PRECISION_EWMA_MIN_SAMPLES's own
    comment: code review (2026-07-29) found the original single-jump
    synthetic check overclaimed an "identical across all four dimensions"
    transient shape that does not hold for every jump magnitude (a small
    jump relative to a dimension's own min_variance floor breaks it). This
    test replaces that overclaim with what's actually needed and true: swept
    across a wide range of realistic cold-start jump magnitudes (0.1%-100%
    of each dimension's own real measured historical max) using the REAL
    shipped alpha/min_variance constants, |z| must be back under
    DIMENSION_PRECISION_ZSCORE_SATURATION (and stay there) by
    DIMENSION_PRECISION_EWMA_MIN_SAMPLES observations, for every dimension
    and every jump magnitude tested -- not just one hand-picked scenario."""
    fractions = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    for dim in sorted(PRESSURE_DIMENSIONS):
        min_var = DIMENSION_PRECISION_MIN_VARIANCE[dim]
        for frac in fractions:
            jump = _REAL_MEASURED_MAX[dim] * frac
            ewma, var, n = 0.0, 0.0, 0
            # Exactly DIMENSION_PRECISION_EWMA_MIN_SAMPLES total
            # observations: the cold-start seed tick (value=0.0) plus
            # MIN_SAMPLES-1 more held-constant ticks at the jump value.
            values = [0.0] + [jump] * (DIMENSION_PRECISION_EWMA_MIN_SAMPLES - 1)
            for value in values:
                update = compute_ewma_update(
                    prev_ewma=ewma,
                    prev_variance=var,
                    prev_count=n,
                    value=value,
                    alpha=DIMENSION_PRECISION_EWMA_ALPHA,
                    min_variance=min_var,
                )
                ewma, var = update.ewma, update.variance
                n += 1
            assert n == DIMENSION_PRECISION_EWMA_MIN_SAMPLES
            assert update.zscore is not None
            assert abs(update.zscore) < DIMENSION_PRECISION_ZSCORE_SATURATION, (
                f"{dim} at jump fraction {frac} did not settle by "
                f"n={DIMENSION_PRECISION_EWMA_MIN_SAMPLES}: |z|={abs(update.zscore):.3f}"
            )
