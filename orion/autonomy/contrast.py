"""Baseline-matched contrast: what an action DID, not what happened after it.

WHY THIS EXISTS
---------------
Phase 1 of the action-outcome ledger (PR #1798) recorded, for every declared
action, the unconditional field delta across its window: `after - before`.
Adversarial review killed that number and this module replaces it.

The defect is selection on the signal. An action is dispatched *because* a
pressure is high, and a high pressure falls on its own. So the raw
post-action delta is dominated by mean reversion, not by the action.
Measured live over 3 days on `prune_dangling_images`:

    pruned?   n        mean resource_pressure delta
    no      66036      -0.0257
    yes      3426      -0.1481      <- reads as a 5.8x effect

Condition on where the pressure started and it INVERTS: in 6 of 8 comparable
baseline deciles the prune arm falls *less* than the no-prune arm. The whole
raw gap is regression to the mean.

That failure mode is nastier than the ones this repo already guards. A
zero-filled or dead metric announces itself as an implausible 0.0. A
confounded estimator converges on a plausible, low-variance, CONFIDENT
number -- and the Bayesian surprise term riding on top decays to ~0 exactly
because the wrong estimate is stable. It looks maximally trustworthy
precisely when it is wrong.

WHAT THE CONTROL ARM ACTUALLY IS
--------------------------------
The design spec (docs/superpowers/specs/2026-08-21-action-value-control-arm-
design.md) proposed using capacity-blocked candidates -- actions approved by
policy that lost the `max_dispatch_candidates:5` race -- as the control arm.
Checked against live frames before building it, that arm does not work:

    n_dispatched  n_blocked  frames (2h live sample)
    0             0          667
    5             5           34
    3             7            5

Blocked candidates only ever exist in ticks where five OTHER candidates DID
go out, because the cap only binds when there is a queue. The field delta is
measured frame-wide, so a "control" record from such a tick carries the delta
produced by its five dispatched siblings. There is no clean capacity-blocked
control frame at all -- the arm is contaminated by construction, and a
within-tick contrast between a dispatched and a blocked candidate is
identically zero, since both read the same before and the same after.

The control population that genuinely exists is the 94% of ticks in which
NOTHING claiming that signal ran. `arm='no_action'` is that arm: one
observation per (feedback frame, signal) where zero dispatched candidates
declared the signal. Same field, same clock, same measurement path,
untreated.

`capacity_blocked` rows are still written to the ledger, because "the action
that lost the race saw the signal do X" is worth having; they are NOT used
as the control arm, and `contrast()` will not accept them as one.

THE ESTIMATOR
-------------
Bin by where the signal started. For an action A on signal S:

    effect(A, S) = sum_b  w_b * ( mean_delta[treated, b] - mean_delta[control, b] )

with `w_b` the treated arm's share of volume in bin b, so the answer is
"what did A do over the conditions A actually runs in" rather than over a
uniform prior on conditions. Variance adds across the difference:

    var = sum_b  w_b^2 * ( var[treated, b] + var[control, b] )

Bins where the control arm has no coverage contribute NOTHING and are
reported as `uncovered_weight`. They are never silently backfilled with the
raw delta -- that would reintroduce the exact bias this module exists to
remove, in the bins where it is worst.

This is a QUASI-EXPERIMENT, not a randomized one. Ticks in which nothing
fired are systematically calmer ticks; binning on the baseline absorbs most
of that and does not absorb all of it. Every artifact derived from it must
say `quasi_experimental`. Only `arm='randomized_holdback'` -- deliberately
withholding a fraction of actions that won their slot -- licenses the word
causal, and the two arms must never be merged into one number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from orion.autonomy.prediction import EffectPosterior

ActionArm = Literal["dispatched", "capacity_blocked", "no_action", "randomized_holdback"]

# The arms admissible as a comparison group, strongest first. Order is
# load-bearing: contrast() picks the first one with any coverage and never
# pools two, because pooling an experimental arm with a quasi-experimental
# one produces a number that is neither and gets described as the better of
# the two.
CONTROL_ARM_PRECEDENCE: tuple[ActionArm, ...] = ("randomized_holdback", "no_action")

BASELINE_BIN_COUNT = 10

# A control cell that has never once seen the signal move is not a calm
# baseline, it is a frozen instrument, and contrasting against it hands the
# treated arm's entire raw delta back as if it were an effect.
#
# This is not hypothetical. `resource_pressure` -- the signal carrying 26.5%
# of declared-claim dispatch volume, and the one the three docker-prune
# templates claim -- sat at EXACTLY 0.85 with stddev exactly 0.0 across
# ~12,000 consecutive frames on 2026-08-21 (08:00-15:00 UTC and beyond),
# against 2,600 distinct values/day on every preceding day. Root cause:
# `node:substrate.vision.prediction_error` saturated at exactly 1.0 for 12+
# hours (the vision channel's staleness pressure, correctly reporting a
# blind camera), times the 0.85 `node:substrate.vision -> capability:vision`
# edge weight in config/field/orion_field_topology.v1.yaml, winning the
# max() merge into resource_pressure. A freshness check cannot catch it: the
# value was rewritten every single tick. It was fresh, present, and constant.
#
# Below this many observations the cell is merely young, and a run of
# genuine calm is unremarkable. Above it, zero movement is an instrument
# report, not a measurement.
FROZEN_CONTROL_MIN_N = 200

# Same dead band the feedback path already uses for these exact channels
# (orion.feedback.extractors.classify_pressure_deltas).
MOVEMENT_EPSILON = 1e-6


def baseline_bin(value: float) -> int:
    """Fixed-width decile of a [0, 1] pressure, stamped at write time.

    Deliberately fixed edges rather than the trailing-window quantiles the
    design spec proposed. Quantile edges make the bin identity mean
    something different at different times -- bin 3 in August is not bin 3 in
    September -- so pooling records across time would silently mix
    conditions, which is the same class of defect as the confound this whole
    module exists to remove. Fixed edges also let the writer stamp the bin
    and the reader trust it without holding any distribution state.

    Values outside [0, 1] are clamped rather than rejected: every
    PredictableSignal is a normalised pressure, but a producer bug should
    degrade to an edge bin, not crash the feedback runtime.
    """
    if not math.isfinite(value):
        raise ValueError(f"baseline must be finite, got {value!r}")
    scaled = int(math.floor(value * BASELINE_BIN_COUNT))
    return max(0, min(BASELINE_BIN_COUNT - 1, scaled))


# (dispatch_kind, target_id, signal_id, baseline_bin)
TreatedCellKey = tuple[str, str, str, int]
# (signal_id, arm, baseline_bin)
ControlCellKey = tuple[str, str, int]


@dataclass(frozen=True)
class ControlCell:
    """One untreated cell: the belief, plus proof the instrument is alive.

    `moved_n` is the number of observations that left the dead band. It is
    carried alongside the posterior rather than derived from it because a
    Normal-Normal posterior with a FIXED observation variance cannot tell
    the difference: its variance shrinks as 1/n whether the underlying data
    varies or is a single repeated constant. A frozen channel therefore
    produces a cell that looks MORE trustworthy the longer it stays broken.
    """

    posterior: EffectPosterior
    moved_n: int = 0

    @property
    def is_frozen(self) -> bool:
        return self.moved_n == 0 and self.posterior.n >= FROZEN_CONTROL_MIN_N


@dataclass(frozen=True)
class BinContrast:
    """One baseline bin's share of the answer."""

    baseline_bin: int
    treated_mean: float
    treated_n: int
    control_mean: float
    control_n: int
    weight: float

    @property
    def difference(self) -> float:
        return self.treated_mean - self.control_mean


@dataclass(frozen=True)
class ContrastEstimate:
    """A baseline-matched effect estimate, with its own coverage attached.

    `uncovered_weight` is not a footnote. It is the share of the action's
    real volume that this number does NOT describe, and a caller that
    ignores it is reporting a partial estimate as a whole one.
    """

    dispatch_kind: str
    target_id: str
    signal_id: str

    value: float
    variance: float

    treated_n: int
    control_n: int
    control_arm: ActionArm
    bins: tuple[BinContrast, ...]
    uncovered_weight: float

    @property
    def sd(self) -> float:
        return math.sqrt(max(self.variance, 0.0))

    @property
    def evidence_class(self) -> Literal["experimental", "quasi_experimental"]:
        return (
            "experimental"
            if self.control_arm == "randomized_holdback"
            else "quasi_experimental"
        )


def pooled_treated_mean(
    treated: dict[TreatedCellKey, EffectPosterior],
    dispatch_kind: str,
    target_id: str,
    signal_id: str,
) -> EffectPosterior | None:
    """Volume-weighted mean of the treated cells, across baseline bins.

    This is what an action expects to OBSERVE when it runs, which is what
    `ExpectedEffectV1.predicted_delta` has always meant and still means. It
    is deliberately NOT the contrast: `prediction_error` on a ledger row is
    `observed_delta - predicted_delta`, and comparing a raw observed delta
    against a counterfactual-adjusted contrast would be a units mismatch
    dressed up as a residual. The contrast is what a BUDGET reads; the
    pooled mean is what a PREDICTION claims.
    """
    cells = [
        (key[3], post)
        for key, post in treated.items()
        if key[0] == dispatch_kind and key[1] == target_id and key[2] == signal_id
    ]
    cells = [(b, p) for b, p in cells if p.n > 0]
    if not cells:
        return None
    total_n = sum(p.n for _, p in cells)
    mean = sum(p.mean * p.n for _, p in cells) / total_n
    # Variance of a weighted mean of independent cell posteriors.
    variance = sum((p.n / total_n) ** 2 * p.variance for _, p in cells)
    return EffectPosterior(mean=mean, variance=variance, n=total_n)


def contrast(
    treated: dict[TreatedCellKey, EffectPosterior],
    control: dict[ControlCellKey, ControlCell],
    dispatch_kind: str,
    target_id: str,
    signal_id: str,
    *,
    min_control_n: int = 1,
) -> ContrastEstimate | None:
    """Baseline-matched effect of one action on one signal.

    Returns None when no bin has coverage in BOTH arms -- the `NO CONTROL`
    state. A None here must be reported as "not measured", never rendered as
    0.0 and never quietly replaced by the raw delta.

    Frozen control cells (see ControlCell.is_frozen) are refused as coverage
    rather than used, because a control arm that has never seen the signal
    move contributes a constant and hands the treated arm's raw delta back
    unchanged -- wearing the contrast's name and the contrast's confidence.
    """
    treated_cells = {
        key[3]: post
        for key, post in treated.items()
        if key[0] == dispatch_kind
        and key[1] == target_id
        and key[2] == signal_id
        and post.n > 0
    }
    if not treated_cells:
        return None

    for arm in CONTROL_ARM_PRECEDENCE:
        control_cells = {
            key[2]: cell.posterior
            for key, cell in control.items()
            if key[0] == signal_id
            and key[1] == arm
            and cell.posterior.n >= min_control_n
            and not cell.is_frozen
        }
        covered = sorted(set(treated_cells) & set(control_cells))
        if not covered:
            continue

        covered_volume = sum(treated_cells[b].n for b in covered)
        total_volume = sum(p.n for p in treated_cells.values())

        bins: list[BinContrast] = []
        value = 0.0
        variance = 0.0
        for b in covered:
            t = treated_cells[b]
            c = control_cells[b]
            weight = t.n / covered_volume
            bins.append(
                BinContrast(
                    baseline_bin=b,
                    treated_mean=t.mean,
                    treated_n=t.n,
                    control_mean=c.mean,
                    control_n=c.n,
                    weight=weight,
                )
            )
            value += weight * (t.mean - c.mean)
            variance += (weight**2) * (t.variance + c.variance)

        return ContrastEstimate(
            dispatch_kind=dispatch_kind,
            target_id=target_id,
            signal_id=signal_id,
            value=value,
            variance=variance,
            treated_n=covered_volume,
            control_n=sum(control_cells[b].n for b in covered),
            control_arm=arm,
            bins=tuple(bins),
            uncovered_weight=(
                (total_volume - covered_volume) / total_volume if total_volume else 0.0
            ),
        )

    return None
