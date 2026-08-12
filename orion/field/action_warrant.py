"""`action_warrant`: how unusual is Orion's current internal state, on a scale
whose rest point is defined rather than guessed.

WHY THIS EXISTS (2026-08-12)

Work Outcome O1 (`orion/sentience_striving_program/README.md`) asks that Orion's
autonomous-action budget rise and fall with real internal pressure. It never has.
The proposal arena dispatched exactly `max_dispatches_per_tick` actions every
tick, forever, and `thresholds.min_priority` never rejected a single candidate in
the arena's recorded history.

The reason was not a mis-set threshold. It was that the quantity being compared
against the threshold had no defined rest point. `resource_pressure = 0.3255`
is not interpretable on its own -- calm or busy depends entirely on what this
machine normally does -- so any fixed cut on it is a guess about scale, and the
guess was wrong by enough that the gate could never bind (measured floor 0.3035
against a threshold of 0.10).

Five candidate statistics were measured against real history before this one.
All five are recorded here because each failure is the reason for a specific
design decision below, and re-deriving them costs a deploy cycle each:

  raw max over dimensions        0.00% of ticks could read calm
  max |z|                        0.00%  -- absolute value destroys the rest state
  mean |z|                       0.00%  -- E[|Z|] = sqrt(2/pi) for a calm
                                    population, a permanent floor. This is the
                                    exact failure CLAUDE.md 0A already documents
                                    for bus_synaptic_prediction_error().
  max signed z                   reached "calm" only when a dimension was PINNED;
                                    the apparent 30% rest was reading a degenerate
                                    thermal-dominated resource_pressure, and went
                                    to 0.00% the moment that dimension was fixed.
  Mahalanobis p-value            reached rest (65.75%) but was unstable as
                                    dimensions were added (rest swung 44% -> 31%
                                    -> 66%) and a pinned dimension still faked
                                    calm.

The common defect in the first four is the extreme-value operator: P(all N
sources simultaneously at or below their own normal) shrinks geometrically in N,
so **adding a healthy sensor makes the system look busier**. Standardising the
inputs does not fix it, because the units were never the problem.

WHAT THIS COMPUTES

Each dimension's z-score against its own EWMA baseline (already produced every
digestion tick by `services/orion-field-digester/app/digestion/precision.py`)
is converted to a one-sided upper-tail probability:

    u_i = P(Z > z_i)

Under "this dimension is at its own normal", u_i is Uniform(0, 1) by
construction. That is the property the whole design rests on: a calibrated
rest point, not an assumed one.

Those are then combined by Fisher's method (Fisher 1925, *Statistical Methods
for Research Workers*, section 21.1 -- the standard combined-probability test):

    X = -2 * sum(ln u_i)        ~  chi-square with 2N degrees of freedom
    action_warrant = P(chi2_2N <= X)

Fisher's method is chosen specifically because **the number of contributing
dimensions enters the null distribution** (2N degrees of freedom) rather than
biasing the result. It requires the inputs to be independent, which was
verified against real data rather than assumed: the four dimensions' z-scores
have max pairwise |r| = 0.151 and a participation ratio of 3.95 out of 4
(n = 17,642). They are genuinely four signals, not one wearing four names.

This is NOT scale-invariant, and claiming otherwise would be the same kind of
overclaim this module exists to replace. A tick where every dimension sits
exactly on its own median scores:

    N = 1   0.5000        N = 4   0.3020        N = 7   0.2165
    N = 2   0.4034        N = 5   0.2681        N = 10  0.1626

because N independent readings all landing exactly on their medians is
genuinely more remarkable than one doing so. The statistic gets more decisive
as evidence accumulates -- calm converges to 0, elevated converges to 1 -- which
is the correct behaviour for a combined-probability test, not a defect.

What matters is the DIRECTION of that drift: adding a dimension moves an
average tick toward calm. Every max()-shaped predecessor drifted the other way,
each new sensor pushing the reading toward "act" until nothing could ever rest.
Here the failure mode of adding dimensions is under-firing, which is both the
safe direction and a visible one. The practical consequence for any future
gate: a fixed threshold on this score is only meaningful alongside N, which is
why `ActionWarrant` carries `contributing` rather than returning a bare float.

The result is a [0, 1] score where 0.5 means "a median normal day for this
machine", high means "genuinely unusual", and -- unlike every predecessor --
low is reachable.

LIVENESS IS PART OF THE STATISTIC, NOT A GARNISH

A dimension that is dead reads as perfectly average, because a pinned value sits
exactly on its own baseline. Measured: pinning `execution_pressure` inflates the
apparent rest fraction from 63.19% to 74.97% -- the system looks calmer because a
sensor stopped working. That is the same trap that killed the max-signed-z
candidate, and it is not hypothetical: this repo has shipped
decayed-to-zero-looks-like-calm before (`node:substrate.route`'s prediction_error,
CLAUDE.md 0A).

So a dimension contributes only if it passes `_is_live()`. Excluding it is safe
precisely BECAUSE the null is count-aware -- N drops, the degrees of freedom drop
with it, and the scale is preserved. A max()-shaped statistic could not do this.

Honest limit, disclosed rather than buried: exclusion recovers most but not all
of the distortion (execution_pressure: 74.97% naive -> 68.68% excluded, against a
true 63.19%). An empirical-ECDF variant tracked the truth far more closely
(within 0.5-1.8pp) but requires persisting a per-dimension quantile history --
new state, new schema, and something expensive to unwind if this signal turns out
wrong. The stateless variant was chosen deliberately under CLAUDE.md 0A's
reversibility rule; the residual bias is a known, measured, bounded error, not an
unknown one.

NOT A GATE YET

This module computes and exposes a number. It deliberately does not decide
anything. Wiring it into `thresholds.min_priority`'s role is a separate patch
with its own acceptance checks, so that the signal can be observed against real
traffic before anything depends on it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field

from orion.schemas.field_state import FieldStateV1

# The dimensions this signal reads. Deliberately imported rather than
# re-declared -- if PRESSURE_DIMENSIONS gains a member, this statistic must see
# it (and its count-aware null means it can, without recalibration).
from orion.proposals.scoring import (
    DIMENSION_PRECISION_EWMA_MIN_SAMPLES,
    DIMENSION_PRECISION_MIN_VARIANCE,
    PRESSURE_DIMENSIONS,
)

# Floor on u_i so a single extreme dimension cannot produce -inf and saturate
# the whole statistic. 1e-9 corresponds to |z| ~ 6, well beyond anything
# observed (max |z| in 17,642 real ticks was 28.5, which this floor caps).
# Not a tuning knob: it exists to keep log() finite, and its only effect is to
# bound one dimension's maximum contribution to X at -2*ln(1e-9) ~= 41.4.
U_FLOOR = 1e-9

# Detecting a genuinely stuck dimension needs BOTH conditions below, and the
# reason is a live measurement, not caution.
#
# `orion.bus.ewma.compute_ewma_update` applies `min_variance` when computing
# the z-score but STORES the unfloored variance. So any channel that holds a
# constant value for long enough drives its stored variance toward zero,
# regardless of whether it is dead or merely quiet. Measured 2026-08-12 over
# 3,512 live ticks:
#
#     reasoning_pressure     median stored variance  9.750e-32
#     reliability_pressure   median stored variance  5.593e-91
#
# Neither is dead. Both are BURSTY -- long flat stretches punctuated by real
# excursions (reliability_pressure's z reaches 15.8). Their z-scores remain
# informative precisely because the floor rescues them at compute time.
#
# An earlier version of this module treated "stored variance <= floor" as
# pinned, and consequently discarded two of four live dimensions on every
# tick. That is the same class of error as the signals this module replaces:
# a plausible-looking check measuring the wrong quantity.
#
# A genuinely stuck dimension has a value identical to its own baseline, which
# produces a z of exactly 0.0. That is the real signature, and requiring it
# alongside a collapsed variance separates "stuck" from "quiet".
VARIANCE_COLLAPSE_EPS = 1e-12


@dataclass(frozen=True)
class ActionWarrant:
    """The signal plus everything needed to explain it.

    `contributing`/`excluded` are not debug garnish: the score is only
    interpretable alongside N, because the null distribution depends on it.
    A warrant of 0.9 over 4 dimensions and one over 1 dimension are different
    claims, and a consumer that logs only the float cannot tell them apart.
    """

    score: float | None
    contributing: list[str] = dc_field(default_factory=list)
    excluded: dict[str, str] = dc_field(default_factory=dict)

    @property
    def degrees_of_freedom(self) -> int:
        return 2 * len(self.contributing)


def _chi2_sf(x: float, df: int) -> float:
    """Survival function of chi-square, exact for even df.

    Fisher's X always has df = 2N, so df is even by construction and the
    closed form applies -- no scipy dependency, no numerical integration, no
    approximation error to reason about.

        P(chi2_2k > x) = exp(-x/2) * sum_{i=0}^{k-1} (x/2)^i / i!
    """
    if x <= 0.0:
        return 1.0
    k = df // 2
    half = x / 2.0
    term = 1.0
    total = 1.0
    for i in range(1, k):
        term *= half / i
        total += term
    return min(1.0, math.exp(-half) * total)


def _upper_tail(z: float) -> float:
    """P(Z > z) for a standard normal, via erfc. Floored at U_FLOOR."""
    return max(0.5 * math.erfc(z / math.sqrt(2.0)), U_FLOOR)


def _is_live(field: FieldStateV1, dim_id: str) -> str | None:
    """Return a reason string if this dimension must NOT contribute, else None.

    Three ways a dimension is not reporting, each one an observed failure mode
    in this repo rather than a defensive guess:

      no_reading      absent from this tick entirely. `map_channels_to_
                      dimensions` only emits a dimension a real channel mapped
                      to, so absence is a fact, not a 0.0 to fabricate.
      cold_baseline   fewer than DIMENSION_PRECISION_EWMA_MIN_SAMPLES real
                      observations. A z-score against a baseline built from one
                      prior sample is not a measurement (simulated 2026-07-28:
                      z=1000 on the second observation of a steady ramp).
      pinned          stored EWMA variance collapsed AND this tick's z is
                      exactly 0.0 -- the value is sitting on its own baseline
                      and carries no information, so including it would drag
                      the statistic toward "perfectly normal" for free. This is
                      the dead-vs-calm trap this module exists to avoid. Both
                      conditions are required; see VARIANCE_COLLAPSE_EPS for
                      the measurement showing why variance alone is not enough.
    """
    if dim_id not in field.dimension_precision_zscore:
        return "no_reading"
    if field.dimension_precision_ewma_n.get(dim_id, 0) < DIMENSION_PRECISION_EWMA_MIN_SAMPLES:
        return "cold_baseline"
    variance = field.dimension_precision_ewma_var.get(dim_id, 0.0)
    zscore = float(field.dimension_precision_zscore[dim_id])
    if variance <= VARIANCE_COLLAPSE_EPS and zscore == 0.0:
        return "pinned"
    return None


def action_warrant(field: FieldStateV1) -> ActionWarrant:
    """Fisher-combined upper-tail probability across the live pressure dimensions.

    Returns `score=None` -- never a fabricated mid-range value, never 0.0 --
    when no dimension is live enough to contribute. "I cannot tell" and "nothing
    is happening" are different states, and collapsing them is how a dead
    pipeline comes to look like a calm one (CLAUDE.md 0A, no empty-shell
    cognition).
    """
    contributing: list[str] = []
    excluded: dict[str, str] = {}
    statistic = 0.0

    for dim_id in sorted(PRESSURE_DIMENSIONS):
        reason = _is_live(field, dim_id)
        if reason is not None:
            excluded[dim_id] = reason
            continue
        z = float(field.dimension_precision_zscore[dim_id])
        statistic += -2.0 * math.log(_upper_tail(z))
        contributing.append(dim_id)

    if not contributing:
        return ActionWarrant(score=None, contributing=[], excluded=excluded)

    score = 1.0 - _chi2_sf(statistic, 2 * len(contributing))
    return ActionWarrant(
        score=max(0.0, min(1.0, score)),
        contributing=contributing,
        excluded=excluded,
    )
