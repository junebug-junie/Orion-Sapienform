"""The field's pressure dimensions and precision constants (orion/field/pressure_dimensions.py).

Moved from tests/test_proposal_scoring.py on 2026-11-07, when the constants moved out of the
retired orion.proposals.scoring (ladder retirement PR 6). The proposal-scoring tests retired with
the arena; these two pin what the field digester and action_warrant still read.
"""
from __future__ import annotations

import importlib
import sys

from orion.bus.ewma import compute_ewma_update
from orion.field.pressure_dimensions import (
    DIMENSION_PRECISION_EWMA_ALPHA,
    DIMENSION_PRECISION_EWMA_MIN_SAMPLES,
    DIMENSION_PRECISION_MIN_VARIANCE,
    DIMENSION_PRECISION_ZSCORE_SATURATION,
    PRESSURE_DIMENSIONS,
)


# Real historical max per dimension, scripts/analysis/measure_proposal_
# dimension_variance.py's 16h real substrate_field_state window, 2026-07-28.
_REAL_MEASURED_MAX = {
    "execution_pressure": 0.392273,
    "resource_pressure": 0.9,
    "reliability_pressure": 0.9,
    "reasoning_pressure": 0.045,
    # deviation_pressure's own real measured max (not that same 2026-07-28
    # window -- it did not exist yet): the 41,973-tick / 2026-08-16 window
    # cited in orion/field/pressure_dimensions.py's PRESSURE_DIMENSIONS comment
    # reports a real max of 1.0 (the saturation ceiling, clamped by
    # orion.attention.tension.competition.deviation_pressure() itself), not
    # an approximation.
    "deviation_pressure": 1.0,
    # sustained_load_pressure's own real measured max: the 24h/1,395-point/
    # 34,316-real-row window cited in orion/field/pressure_dimensions.py's
    # PRESSURE_DIMENSIONS comment (scripts/analysis/measure_sustained_load_
    # pressure.py, 2026-08-18) reports a real max of 0.7703 -- the ceiling
    # was set by node:athena's disk_capacity_pressure, the one channel
    # driving nearly the whole reading over that window.
    "sustained_load_pressure": 0.7703,
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




def test_every_dimension_has_a_precision_floor():
    assert set(DIMENSION_PRECISION_MIN_VARIANCE) == PRESSURE_DIMENSIONS
    assert PRESSURE_DIMENSIONS == frozenset({
        "execution_pressure", "resource_pressure", "reasoning_pressure",
        "reliability_pressure", "deviation_pressure", "sustained_load_pressure",
    })


def test_the_retired_proposal_package_is_not_needed_to_score_the_field():
    """The field digester and action_warrant must not reach for orion.proposals (deleted)."""
    for name in [m for m in sys.modules if m == "orion.proposals" or m.startswith("orion.proposals.")]:
        del sys.modules[name]
    importlib.import_module("orion.field.action_warrant")
    importlib.import_module("orion.field.pressure_dimensions")
    assert not any(m == "orion.proposals" or m.startswith("orion.proposals.") for m in sys.modules)
