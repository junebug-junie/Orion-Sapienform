from __future__ import annotations

from orion.proposals.policy import ProposalPolicyV1, ProposalTemplateV1
from orion.schemas.field_state import FieldStateV1

# 2026-07-22, SelfStateV1 burn: field_intensity and uncertainty removed. Both
# were composite, hand-tuned SelfStateV1 dimensions with no principled
# non-hand-tuned replacement (see orion/field/pressure.py's module docstring).
# The 4 remaining categories are real, direct channel-merge reads, unaffected
# by the burn.
#
# 2026-07-28 precision-weighted confidence note, flagged by code review
# (2026-07-29): orion.field.pressure.field_pressures() can produce UP TO 7
# dimension keys via CHANNEL_DIMENSION_MAP (this 4-entry set plus
# continuity_pressure/introspection_pressure/social_pressure), but only
# these 4 are measured and EWMA-tracked by dimension_confidence()'s
# precision baseline (services/orion-field-digester/app/digestion/
# precision.py). No current config/proposals/proposal_policy.v1.yaml
# template scores on any of the other 3 (checked live), so this has no
# current effect -- but if a future template ever does, dimension_
# confidence() will PERMANENTLY return 0.0 for it (falls into the cold-
# start/n<MIN_SAMPLES branch forever, since an untracked dimension's `n`
# never advances), unlike the old binary presence flag this replaced, which
# would have correctly returned 1.0 whenever that dimension was present.
# Extending PRESSURE_DIMENSIONS (and DIMENSION_PRECISION_MIN_VARIANCE,
# enforced in sync by the assertion below) to cover a newly-scored dimension
# is required in the same patch that adds it to a template.
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

# Hand-verified 2026-07-28 via a synthetic cold-start-from-zero replay (seed
# baseline at 0.0, jump to a real value on the next tick, hold), swept across
# a wide range of jump magnitudes per dimension (0.1%-100% of that
# dimension's own real measured historical max, not just one hand-picked
# value): the worst case found across ALL FOUR dimensions and every jump
# magnitude tested was n=8 -- |z| first drops below
# DIMENSION_PRECISION_ZSCORE_SATURATION and stays there by the 8th post-seed
# observation, never later. This is NOT a claim that the transient is
# identical across dimensions for every jump size -- a first attempt at this
# comment claimed that, and code review (2026-07-29) reproduced the
# simulation with a *different* jump choice (each dimension's real measured
# MEAN rather than its max) and found reliability_pressure's small mean
# relative to its own min_variance floor breaks the identical-trajectory
# shape (its z never even approaches saturation at that jump size). The
# invariance only holds when the jump is large enough that
# alpha * jump**2 dominates that dimension's min_variance floor; for smaller
# jumps the shape differs by domain but resolves EQUALLY FAST OR FASTER in
# every case checked, never slower than n=8. See tests/test_proposal_
# scoring.py::test_cold_start_min_samples_sufficient_across_jump_magnitude_
# sweep for the swept regression test that locks this in against the real
# shipped constants, not just a synthetic one-off assertion.
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
#
# 2026-08-11 (Patch A; design doc in PR #1554, not merged as of this commit --
# see orion/field/pressure.py's CHANNEL_DIMENSION_MAP tombstone for why this
# cites a PR number rather than a path): `resource_pressure` re-derived from
# 5e-5 to 2e-3.
# Mandatory, not opportunistic -- removing `thermal_pressure` from
# orion/field/pressure.py's CHANNEL_DIMENSION_MAP changes this dimension's
# distribution, so a floor calibrated against the old one would silently
# mis-scale every confidence it produces. Re-measured through the real
# production path (the same measure_proposal_dimension_variance.py cited
# above, 6h / 10,562 real ticks, run on the patched code):
#
#     resource_pressure   variance 0.0037973 -> 0.0174205  (4.6x)
#                         mean     0.538208  -> 0.32551
#                         min      0.314286  -> 0.144535
#                         real upward transitions 270 -> 431 (+60%)
#                         classification NON_DEGENERATE_USABLE, unchanged
#
# 2e-3 is ~1/10th of the new variance, same rule as the four originals.
# Hand-verified by the same replay method: at this floor real p99 |z| is
# 3.116 (inside the 1.3-4.3 discriminating band the originals were checked
# against), median |z| 0.741, 1.13% of ticks at |z| >= 3, and the tail is
# bounded from max |z| 17.195 (at the old 5e-5) down to 10.023. The raw
# unfloored EWMA variance does reach ~0 during flat holds in this series, so
# the floor is load-bearing here, not decorative.
#
# KNOWN STALE, deliberately NOT fixed in this patch: the other three floors
# have drifted from their 2026-07-28 derivation independently of this change
# (measured on UNPATCHED code, same 6h window) --
#     execution_pressure    9.63e-4 -> 1.987e-2   (20.6x, floor 20x too low)
#     reasoning_pressure    2.11e-6 -> 3.494e-7   (0.17x, floor 6x too high)
#     reliability_pressure  1.09e-2 -> 5.520e-3   (0.51x, floor 2x too high)
# Left alone here so this patch's effect stays attributable to the one map
# entry it removes; re-deriving all four at once would make the post-deploy
# data impossible to read.
#
# This is a real, live mis-scaling and it is NOT tracked only in an unmerged
# doc -- the numbers above are the tracking. execution_pressure matters most:
# it carries the highest dimension_weight (0.30) AND is one of the four
# PRESSURE_DIMENSIONS that _pressure_dimension_ids() falls back to for every
# template declaring `dimensions: {}`, so a 20x-too-low floor inflates its
# z-scores across most of the arena. Re-derive all three by re-running
# scripts/analysis/measure_proposal_dimension_variance.py; that script needs
# no new instrument and produced every figure above.
DIMENSION_PRECISION_MIN_VARIANCE: dict[str, float] = {
    "execution_pressure": 1e-4,
    "resource_pressure": 2e-3,
    "reasoning_pressure": 2e-7,
    "reliability_pressure": 1e-3,
}

# Code review (2026-07-29) flagged this as unguarded: services/orion-field-
# digester/app/digestion/precision.py indexes this dict with a plain `[]`
# (not `.get`) for every dim_id in PRESSURE_DIMENSIONS that has a reading --
# if PRESSURE_DIMENSIONS ever gains an entry without a matching floor here,
# every subsequent digestion tick that dimension appears in raises KeyError
# (caught by that service's broad tick-level except, so it wouldn't crash
# the service, but it would silently break the WHOLE digestion tick, not
# just precision tracking, on every run until fixed -- an empty-shell-
# cognition failure shape, not a clean one). This isn't hypothetical: the
# 2026-07-22 SelfStateV1 burn already changed PRESSURE_DIMENSIONS once (see
# that frozenset's own comment). Fails fast at import time instead.
assert set(DIMENSION_PRECISION_MIN_VARIANCE) == PRESSURE_DIMENSIONS, (
    "DIMENSION_PRECISION_MIN_VARIANCE's keys must exactly match PRESSURE_DIMENSIONS "
    "-- add or remove a floor here in the same change that edits PRESSURE_DIMENSIONS"
)


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

    NOTE (see `PRESSURE_DIMENSIONS`' own comment for the full account): if
    `dimension_id` is a real, present `field_pressures` key that is NOT in
    `PRESSURE_DIMENSIONS` (e.g. `continuity_pressure`), this returns 0.0
    PERMANENTLY, every tick -- `n` never advances for a dimension the
    producer doesn't track. Unlike the absent-this-tick and cold-start cases
    above, this is not a transient "no data yet" state; it is a real
    behavior difference from the old binary presence flag, which would have
    returned 1.0 whenever such a dimension was present. Not a bug for
    today's live templates (none score on those 3 dimensions), but a real
    trap for a future one that does.
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


def _pressure_dimension_ids(template: ProposalTemplateV1) -> list[str]:
    """The dimension set proposal_urgency() and proposal_confidence() both
    read: template.dimensions filtered to real pressure dimensions, falling
    back to all of PRESSURE_DIMENSIONS if the template scores on none (e.g.
    an honestly-empty `dimensions: {}` post-2026-07-30 dead-dimension fix).

    Shared by both functions so they can never again drift out of sync the
    way they did before that fix: proposal_urgency()'s own filter used to
    let a dead dimension ending in "_pressure" (contract_pressure) pass
    through and suppress this exact fallback, while proposal_confidence()
    had no fallback at all and returned a bare 0.0 for the same template.
    One shared dimension-set means one fallback behavior for both signals.
    """
    dims = [
        dim_id
        for dim_id in template.dimensions
        if dim_id in PRESSURE_DIMENSIONS or dim_id.endswith("_pressure")
    ]
    return dims if dims else list(PRESSURE_DIMENSIONS)


def proposal_urgency(
    *,
    field_pressures: dict[str, float],
    template: ProposalTemplateV1,
) -> float:
    scores = [
        dimension_score(field_pressures, dim_id)
        for dim_id in _pressure_dimension_ids(template)
    ]
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
        for dim_id in _pressure_dimension_ids(template)
    ]
    if not confs:
        return 0.0
    return clamp01(sum(confs) / len(confs))


# 2026-07-30 (docs/superpowers/specs/2026-07-30-proposal-priority-theory-
# anchor-design.md): replaces the original hand-picked
# `base_priority + 0.4*match_score + 0.2*urgency + 0.1*confidence` blend --
# three constants with no theory anchor or citation, unlike every other
# scoring function in this module. Precision-weighting (Feldman & Friston
# 2010 -- same anchor already shipped for dimension_confidence() above and
# for Candidate A's attention salience, orion/attention/field_attention/
# candidate_precision_weighted.py) says a signal should be gated by how much
# it can be trusted, not added to it as an independent term: confidence acts
# as a multiplicative precision weight on whichever real pressure signal is
# stronger (match_score, the template-weighted relevance read, or urgency,
# the raw unweighted severity read -- max() because either alone is a
# sufficient reason to raise priority, same max()-not-sum() shape
# template_match_score() and proposal_urgency() already use internally).
#
# This is NOT yet verified to fix the 98%-single-template dispatch capture
# documented in that design doc (open question 2 there: real pressure
# elevation vs. a scoring-floor artifact) -- it removes the arbitrary
# constants, it does not by itself guarantee dispatch diversity. That is a
# separate, deliberately deferred follow-up (direction (b), diversity-aware
# dispatch), pending real post-deploy dispatch data under this formula.
def proposal_priority(
    *,
    base_priority: float,
    match_score: float,
    urgency: float,
    confidence: float,
) -> float:
    return clamp01(base_priority + confidence * max(match_score, urgency))


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
