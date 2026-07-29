"""Producer-side update for `orion.proposals.scoring.dimension_confidence()`'s
precision-weighted EWMA baseline (2026-07-28, docs/superpowers/specs/2026-07-
28-precision-weighted-proposal-scoring-design.md's "Recommended next patch").

See `orion/schemas/field_state.py`'s `dimension_precision_ewma*` docstring
and `orion/proposals/scoring.py`'s `DIMENSION_PRECISION_*` constants for the
full account -- this module only wires the already-shared
`orion.bus.ewma.compute_ewma_update` mechanism into this service's per-tick
digestion pipeline, same shape as `app/digestion/perturbation.py::
_update_recent_perturbation_baseline`.
"""

from __future__ import annotations

from orion.bus.ewma import compute_ewma_update
from orion.field.pressure import field_pressures
from orion.proposals.scoring import (
    DIMENSION_PRECISION_EWMA_ALPHA,
    DIMENSION_PRECISION_MIN_VARIANCE,
    PRESSURE_DIMENSIONS,
)
from orion.schemas.field_state import FieldStateV1


def update_dimension_precision_baseline(state: FieldStateV1) -> FieldStateV1:
    """EWMA-baselines each of `PRESSURE_DIMENSIONS`' real per-tick
    `field_pressures(state)` reading, once per digestion tick, so
    `orion.proposals.scoring.dimension_confidence()` can score deviation
    from each dimension's own recent normal instead of the old binary
    presence flag.

    Called from `run_digestion_tick()` AFTER `apply_perturbations()` /
    `apply_decay()` / `apply_diffusion()` / `apply_suppression()` -- this
    must score the tick's FINAL `field_pressures()` reading, the same one
    `orion/proposals/builder.py::build_proposal_frame()` later computes from
    the persisted `FieldStateV1` it loads, not an intermediate one.

    A dimension absent from this tick's `field_pressures()` (no real channel
    mapped to it this tick, see `orion/field/pressure.py::
    map_channels_to_dimensions`) is left untouched -- its existing baseline
    holds rather than absorbing a fabricated observation, same "absent this
    tick" convention `_update_recent_perturbation_baseline` already uses.
    """
    pressures = field_pressures(state)
    for dim_id in PRESSURE_DIMENSIONS:
        if dim_id not in pressures:
            continue
        update = compute_ewma_update(
            prev_ewma=state.dimension_precision_ewma.get(dim_id, 0.0),
            prev_variance=state.dimension_precision_ewma_var.get(dim_id, 0.0),
            prev_count=state.dimension_precision_ewma_n.get(dim_id, 0),
            value=pressures[dim_id],
            alpha=DIMENSION_PRECISION_EWMA_ALPHA,
            min_variance=DIMENSION_PRECISION_MIN_VARIANCE[dim_id],
        )
        state.dimension_precision_ewma[dim_id] = update.ewma
        state.dimension_precision_ewma_var[dim_id] = update.variance
        state.dimension_precision_ewma_n[dim_id] = (
            state.dimension_precision_ewma_n.get(dim_id, 0) + 1
        )
        # zscore is None only on this dimension's very first-ever observation
        # (prev_count == 0) -- no baseline existed yet to be anomalous
        # against. Explicitly dropping the key (not writing a fabricated
        # 0.0) preserves dimension_confidence()'s "no empty-shell cognition"
        # cold-start read.
        if update.zscore is not None:
            state.dimension_precision_zscore[dim_id] = update.zscore
        else:
            state.dimension_precision_zscore.pop(dim_id, None)
    return state
