"""Tests for `orion/field/action_warrant.py`.

The three PROPERTY tests below are the load-bearing ones. They are not generic
sanity checks -- each encodes a specific failure that killed a specific earlier
candidate for this signal, measured against real history. If a future patch
changes the combining operator, these are what should fail.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from orion.field.action_warrant import (
    VARIANCE_COLLAPSE_EPS,
    ActionWarrant,
    _chi2_sf,
    action_warrant,
)
from orion.proposals.scoring import (
    DIMENSION_PRECISION_EWMA_MIN_SAMPLES,
    DIMENSION_PRECISION_MIN_VARIANCE,
    PRESSURE_DIMENSIONS,
)
from orion.schemas.field_state import FieldStateV1

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)
SCORED = sorted(DIMENSION_PRECISION_MIN_VARIANCE)  # the 4 with real floors


def _field(zscores: dict[str, float], *, n: int = 128, var_scale: float = 100.0) -> FieldStateV1:
    """A field whose named dimensions are live and carry the given z-scores.

    `var_scale` multiplies each dimension's own DIMENSION_PRECISION_MIN_VARIANCE
    so the stored variance is well clear of VARIANCE_COLLAPSE_EPS. Pinning is
    simulated by the property-3 test setting BOTH a collapsed variance and a
    z of exactly 0.0, which is what a genuinely stuck value produces.
    """
    return FieldStateV1(
        generated_at=NOW,
        tick_id="field.tick:warrant-test",
        dimension_precision_zscore=dict(zscores),
        dimension_precision_ewma_n={d: n for d in zscores},
        dimension_precision_ewma_var={
            d: DIMENSION_PRECISION_MIN_VARIANCE.get(d, 1e-3) * var_scale for d in zscores
        },
    )


# --- the combining maths, checked against known values ----------------------


def test_chi2_sf_matches_closed_form_for_known_points():
    # chi2 with 2 df is Exponential(1/2): P(X > x) = exp(-x/2), exactly.
    for x in (0.5, 1.0, 4.0, 9.0):
        assert _chi2_sf(x, 2) == pytest.approx(math.exp(-x / 2), rel=1e-12)
    # chi2 with 4 df: P(X > x) = exp(-x/2) * (1 + x/2).
    for x in (0.5, 3.0, 7.0):
        assert _chi2_sf(x, 4) == pytest.approx(math.exp(-x / 2) * (1 + x / 2), rel=1e-12)
    assert _chi2_sf(0.0, 8) == 1.0


def test_median_z_gives_mid_scale_warrant():
    """z=0 on every dimension is, by definition, a perfectly average tick.

    u_i = P(Z > 0) = 0.5 for each, so X = -2*N*ln(0.5) = 2*N*ln 2. This is the
    anchor that makes the scale interpretable at all: it must land mid-range,
    not at either rail.
    """
    w = action_warrant(_field({d: 0.0 for d in SCORED}))
    assert w.score is not None
    assert 0.2 < w.score < 0.8
    assert w.contributing == SCORED
    assert w.degrees_of_freedom == 2 * len(SCORED)


# --- PROPERTY 1: both rest and firing are reachable -------------------------


def test_property_1_rest_and_fire_are_both_reachable():
    """The single requirement all five predecessors failed.

    raw max, max|z| and mean|z| each measured 0.00% of real ticks able to read
    calm -- a gate on them can never rest, which looks identical to a busy
    system from the outside and hides a dead pipeline.
    """
    calm = action_warrant(_field({d: -1.5 for d in SCORED}))
    busy = action_warrant(_field({d: 3.0 for d in SCORED}))
    assert calm.score is not None and busy.score is not None
    assert calm.score < 0.05, "must be able to report genuine calm"
    assert busy.score > 0.95, "must still be able to fire"
    assert calm.score < busy.score


def test_monotonic_in_elevation():
    scores = [
        action_warrant(_field({d: z for d in SCORED})).score
        for z in (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    ]
    assert all(a < b for a, b in zip(scores, scores[1:])), scores


# --- PROPERTY 2: stable as dimensions are added -----------------------------


def test_property_2_adding_a_dimension_never_makes_an_average_tick_look_busier():
    """Adding a healthy sensor must not make the system look busier.

    This is the defect shared by every max()-shaped candidate: P(all N sources
    simultaneously at or below normal) shrinks geometrically in N, so each new
    sensor pushed the reading toward "act". Measured: 0.00% of real ticks could
    read calm at N=4.

    Fisher's method puts N into the degrees of freedom instead. The result is
    NOT scale-invariant, and this test deliberately does not assert that it is
    -- an earlier version did, and was wrong. An all-at-median tick scores
    0.5000 at N=1 and 0.1626 at N=10, because N independent readings all
    landing exactly on their own medians is genuinely more remarkable than one
    doing so. The statistic gets more decisive as evidence accumulates, which
    is the correct behaviour for a combined-probability test.

    What must hold, and what this asserts, is the DIRECTION: the drift is
    monotonically toward calm, never toward act. The failure mode of adding
    dimensions is under-firing, not the runaway over-firing that killed every
    predecessor.
    """
    scores = []
    for k in range(1, len(SCORED) + 1):
        w = action_warrant(_field({d: 0.0 for d in SCORED[:k]}))
        assert w.score is not None
        assert len(w.contributing) == k
        scores.append(w.score)

    assert scores[0] == pytest.approx(0.5, abs=1e-12), (
        "a single dimension at its own median must anchor the scale at 0.5"
    )
    assert all(a > b for a, b in zip(scores, scores[1:])), (
        f"drift must be monotonically toward calm, never toward act: {scores}"
    )
    assert min(scores) > 0.15, (
        f"an average tick must stay mid-scale, not collapse to a rail: {scores}"
    )


def test_property_2_more_evidence_sharpens_rather_than_biases():
    """The flip side: repeated calm readings should converge on calm, and
    repeated elevated readings on act. A statistic that did not do this would
    be ignoring evidence rather than combining it."""
    calm = [action_warrant(_field({d: -1.5 for d in SCORED[:k]})).score for k in range(1, 5)]
    busy = [action_warrant(_field({d: 3.0 for d in SCORED[:k]})).score for k in range(1, 5)]
    assert all(a > b for a, b in zip(calm, calm[1:])), calm
    assert all(a <= b for a, b in zip(busy, busy[1:])), busy
    assert calm[-1] < 0.01 and busy[-1] > 0.99


def test_property_2_calm_stays_calm_as_dimensions_are_added():
    scores = [
        action_warrant(_field({d: -1.5 for d in SCORED[:k]})).score
        for k in range(1, len(SCORED) + 1)
    ]
    assert all(s < 0.2 for s in scores), scores


# --- PROPERTY 3: a dead dimension must not manufacture calm -----------------


def test_property_3_pinned_dimension_is_excluded_not_counted_as_calm():
    """A pinned dimension sits exactly on its own baseline, so it reads as
    perfectly average and drags the whole statistic toward "normal".

    Measured on real history: pinning execution_pressure inflated the apparent
    rest fraction from 63.19% to 74.97% -- the system looks calmer because a
    sensor stopped working. Excluding it is only safe because the null is
    count-aware.
    """
    zs = {d: 2.5 for d in SCORED}
    live = action_warrant(_field(zs))

    pinned_field = _field(zs)
    dead = SCORED[0]
    # A stuck value sits on its own baseline: variance collapses AND z is 0.0.
    pinned_field.dimension_precision_ewma_var[dead] = 0.0
    pinned_field.dimension_precision_zscore[dead] = 0.0
    pinned = action_warrant(pinned_field)

    assert pinned.excluded.get(dead) == "pinned"
    assert dead not in pinned.contributing
    assert pinned.degrees_of_freedom == live.degrees_of_freedom - 2
    # The elevated signal from the remaining live dimensions must survive.
    assert pinned.score is not None and pinned.score > 0.9


def test_cold_baseline_dimension_is_excluded():
    zs = {d: 1.0 for d in SCORED}
    f = _field(zs)
    cold = SCORED[1]
    f.dimension_precision_ewma_n[cold] = DIMENSION_PRECISION_EWMA_MIN_SAMPLES - 1
    w = action_warrant(f)
    assert w.excluded.get(cold) == "cold_baseline"
    assert cold not in w.contributing


def test_dimension_absent_this_tick_is_excluded_not_zeroed():
    zs = {d: 1.0 for d in SCORED if d != SCORED[2]}
    w = action_warrant(_field(zs))
    assert w.excluded.get(SCORED[2]) == "no_reading"
    assert SCORED[2] not in w.contributing


# --- the "I cannot tell" state ----------------------------------------------


def test_no_live_dimensions_returns_none_not_zero():
    """None, never 0.0. "I cannot tell" and "nothing is happening" are
    different states; collapsing them is exactly how a dead pipeline comes to
    look like a calm one."""
    w = action_warrant(
        FieldStateV1(generated_at=NOW, tick_id="field.tick:empty")
    )
    assert w.score is None
    assert w.contributing == []
    assert set(w.excluded) == set(PRESSURE_DIMENSIONS)
    assert all(r == "no_reading" for r in w.excluded.values())


def test_extreme_z_is_bounded_not_infinite():
    """One saturated dimension must not swamp the statistic into a hard 1.0
    that no other reading can move -- the U_FLOOR exists for this."""
    w = action_warrant(_field({d: (50.0 if d == SCORED[0] else 0.0) for d in SCORED}))
    assert w.score is not None
    assert math.isfinite(w.score)
    assert w.score <= 1.0


def test_warrant_is_pure_and_does_not_mutate_the_field():
    f = _field({d: 1.25 for d in SCORED})
    before = f.model_dump_json()
    action_warrant(f)
    assert f.model_dump_json() == before


def test_dataclass_is_frozen():
    with pytest.raises(Exception):
        ActionWarrant(score=0.5).score = 0.9  # type: ignore[misc]


def test_bursty_dimension_with_collapsed_variance_stays_live():
    """A collapsed stored variance alone must NOT exclude a dimension.

    `compute_ewma_update` stores the UNFLOORED variance, so any channel that
    holds flat for a while drives it toward zero whether or not it is dead.
    Measured live 2026-08-12: reasoning_pressure sat at 9.75e-32 and
    reliability_pressure at 5.59e-91, and both are bursty-but-real -- their
    z-scores stay informative because the floor rescues them at compute time.
    An earlier version of this module excluded both on every tick, discarding
    half the signal.
    """
    zs = {d: 0.0 for d in SCORED}
    zs[SCORED[0]] = 2.0  # collapsed variance, but genuinely deviating
    f = _field(zs)
    f.dimension_precision_ewma_var[SCORED[0]] = VARIANCE_COLLAPSE_EPS / 1000.0
    w = action_warrant(f)
    assert SCORED[0] in w.contributing
    assert "pinned" not in w.excluded.values()
