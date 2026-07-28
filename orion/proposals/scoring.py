from __future__ import annotations

from orion.proposals.policy import ProposalPolicyV1, ProposalTemplateV1
from orion.schemas.field_state import FieldStateV1

# 2026-07-22, SelfStateV1 burn: field_intensity and uncertainty removed. Both
# were composite, hand-tuned SelfStateV1 dimensions with no principled
# non-hand-tuned replacement (see orion/field/pressure.py's module docstring).
# The 4 remaining categories are real, direct channel-merge reads, unaffected
# by the burn.
PRESSURE_DIMENSIONS = frozenset({
    "execution_pressure",
    "resource_pressure",
    "reasoning_pressure",
    "reliability_pressure",
})


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def dimension_score(field_pressures: dict[str, float], dimension_id: str) -> float:
    return clamp01(field_pressures.get(dimension_id, 0.0))


# 2026-07-28 precision-weighted confidence fix (docs/superpowers/specs/2026-
# 07-28-precision-weighted-proposal-scoring-design.md's "Recommended next
# patch"): dimension_confidence() used to be a binary presence flag (see git
# history for the old docstring, preserved in that design doc's "Current
# architecture" section). Real historical variance was measured first, not
# assumed, per this repo's metric-quality-gate discipline
# (scripts/analysis/measure_proposal_dimension_variance.py, 2026-07-28,
# artifacts under /tmp/measure-proposal-dimension-variance/): all four
# PRESSURE_DIMENSIONS show real, non-degenerate historical variance AND real,
# periodically-recurring perturbation-driven refreshes over a real 16h
# substrate_field_state window (none are the node:substrate.route "zero real
# events ever, decaying unopposed" disease -- that script's own decay-
# artifact + producer-liveness detectors confirmed this by hand, not by
# eyeballing a variance number).
#
# alpha reuses RECENT_PERTURBATION_RATE_EWMA_ALPHA's fixed ~2s-digestion-tick
# cadence rationale (services/orion-field-digester/app/digestion/
# perturbation.py) -- this baseline is updated once per digestion tick
# regardless of whether a real event landed that tick, same producer shape,
# not execution_prediction_error's per-real-observation cadence.
DIMENSION_PRECISION_EWMA_ALPHA = 0.02

# Same z>=3.0 "anomalous" convention already live in bus_synaptic_prediction_
# error, execution_prediction_error, and recent_perturbation_zscore's
# consumers -- not a new calibration.
DIMENSION_PRECISION_ZSCORE_SATURATION = 3.0

# Hand-verified 2026-07-28 via a synthetic cold-start-from-zero replay
# (seed baseline at 0.0, jump to a typical real value on the next tick, hold):
# |z| stays >= DIMENSION_PRECISION_ZSCORE_SATURATION through the 7th
# post-seed observation and first drops below it at the 8th, IDENTICALLY
# across all four dimensions despite their very different real value scales
# -- this transient shape is scale-invariant w.r.t. alpha alone once a
# domain-appropriate min_variance floor is used (verified: both the z-score's
# numerator and denominator scale with the seed jump's magnitude and cancel),
# so a single shared threshold is correct here, unlike RECENT_PERTURBATION_
# EWMA_MIN_SAMPLES which was independently calibrated for its own domain and
# not assumed transferable.
DIMENSION_PRECISION_EWMA_MIN_SAMPLES = 8

# Domain-specific EWMA variance floors -- NOT the shared orion.bus.ewma
# default (1e-6), per the execution_prediction_error lesson that a borrowed
# floor can silently dominate a real z-score. Each set to ~1/10th of that
# dimension's own real measured historical population variance (scripts/
# analysis/measure_proposal_dimension_variance.py, 16h real window,
# 2026-07-28: execution_pressure=9.63e-4, resource_pressure=4.88e-4,
# reasoning_pressure=2.11e-6, reliability_pressure=1.09e-2) and hand-verified
# by replaying each dimension's own real historical series through
# compute_ewma_update at this floor: no permanent near-zero-variance blowup
# during long held/flat stretches (a real, live risk here specifically --
# unlike execution_prediction_error's per-event cadence, this baseline
# updates every digestion tick including ticks where a channel's value is
# unchanged, which drives the EWMA's own internal variance toward zero over
# a long-enough hold unless the floor is large enough to bound it), and no
# permanent saturation either -- real p99 |z| in the 1.3-4.3 range across
# all four dimensions at this floor, a genuinely discriminating spread.
DIMENSION_PRECISION_MIN_VARIANCE: dict[str, float] = {
    "execution_pressure": 1e-4,
    "resource_pressure": 5e-5,
    "reasoning_pressure": 2e-7,
    "reliability_pressure": 1e-3,
}


def dimension_confidence(
    field: FieldStateV1, field_pressures: dict[str, float], dimension_id: str
) -> float:
    """Precision-weighted confidence: how well this tick's real reading for
    `dimension_id` matches its own recent EWMA baseline
    (`field.dimension_precision_ewma`/`_var`/`_n`, updated once per digestion
    tick by `services/orion-field-digester/app/digestion/precision.py::
    update_dimension_precision_baseline()`), inverted from a surprise/anomaly
    z-score into a [0, 1] confidence -- a z-score near 0 (this tick matches
    this dimension's own recent normal) means confidence near 1.0; a z-score
    at or beyond `DIMENSION_PRECISION_ZSCORE_SATURATION` (a real, genuine
    surprise relative to this dimension's own recent trajectory) means
    confidence 0.0. Same clamp/saturation shape as `execution_prediction_
    error`'s surprise score, inverted (confidence, not surprise) -- reuses
    the already-proven mechanism (`orion.bus.ewma.compute_ewma_update`)
    rather than inventing a new one.

    Returns 0.0 (never a fabricated mid-range guess -- this repo's "no
    empty-shell cognition" rule) when:
      - `dimension_id` has no reading at all this tick (absent from
        `field_pressures` -- preserves the old binary flag's "can't be
        confident about data you don't have this tick" semantics), or
      - fewer than `DIMENSION_PRECISION_EWMA_MIN_SAMPLES` real observations
        have been absorbed into this dimension's baseline yet (cold start --
        an early, unreliable z-score must not be reported as a real
        confidence reading; see that constant's own comment for the
        hand-verified evidence).
    """
    if dimension_id not in field_pressures:
        return 0.0
    n = field.dimension_precision_ewma_n.get(dimension_id, 0)
    if n < DIMENSION_PRECISION_EWMA_MIN_SAMPLES:
        return 0.0
    zscore = field.dimension_precision_zscore.get(dimension_id)
    if zscore is None:
        return 0.0
    surprise = min(1.0, abs(zscore) / DIMENSION_PRECISION_ZSCORE_SATURATION)
    return clamp01(1.0 - surprise)


def template_match_score(
    *,
    field_pressures: dict[str, float],
    template: ProposalTemplateV1,
    policy: ProposalPolicyV1 | None = None,
) -> tuple[float, dict[str, float]]:
    contributions: dict[str, float] = {}
    for dim_id, weight in template.dimensions.items():
        policy_weight = 1.0
        if policy is not None:
            policy_weight = float(policy.dimension_weights.get(dim_id, 1.0))
        contributions[dim_id] = clamp01(
            dimension_score(field_pressures, dim_id) * float(weight) * abs(policy_weight)
        )
    match = max(contributions.values()) if contributions else 0.0
    return clamp01(match), contributions


def proposal_urgency(
    *,
    field_pressures: dict[str, float],
    template: ProposalTemplateV1,
) -> float:
    scores = [
        dimension_score(field_pressures, dim_id)
        for dim_id in template.dimensions
        if dim_id in PRESSURE_DIMENSIONS or dim_id.endswith("_pressure")
    ]
    if not scores:
        scores = [dimension_score(field_pressures, d) for d in PRESSURE_DIMENSIONS]
    # No SelfStateV1.overall_intensity fallback survives the burn -- honest
    # 0.0 ("no pressure data this tick") rather than a fabricated rollup.
    return clamp01(max(scores) if scores else 0.0)


def proposal_confidence(
    *,
    field: FieldStateV1,
    field_pressures: dict[str, float],
    template: ProposalTemplateV1,
) -> float:
    confs = [
        dimension_confidence(field, field_pressures, dim_id)
        for dim_id in template.dimensions
    ]
    if not confs:
        return 0.0
    return clamp01(sum(confs) / len(confs))


def proposal_priority(
    *,
    base_priority: float,
    match_score: float,
    urgency: float,
    confidence: float,
) -> float:
    return clamp01(
        base_priority + 0.4 * match_score + 0.2 * urgency + 0.1 * confidence
    )


def proposal_risk(
    *,
    base_risk: float,
    field_pressures: dict[str, float],
    template: ProposalTemplateV1,
) -> float:
    risk = float(base_risk)
    if template.kind in ("prepare_action", "request_policy_review"):
        risk += 0.10
    if template.required_policy_gate not in ("none", "read_only"):
        risk += 0.05
    if dimension_score(field_pressures, "reliability_pressure") >= 0.5:
        risk += 0.10
    # The old "uncertainty" dimension risk bump is gone, not silently reading
    # 0.0 forever (2026-07-22 burn -- see PRESSURE_DIMENSIONS note above).
    if template.kind in ("observe", "inspect", "summarize") and template.required_policy_gate == "read_only":
        risk = min(risk, 0.15)
    return clamp01(risk)
