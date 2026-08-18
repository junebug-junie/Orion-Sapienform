from __future__ import annotations

from orion.schemas.field_state import FieldStateV1

from app.digestion.decay import apply_decay
from app.digestion.diffusion import apply_diffusion
from app.digestion.perturbation import apply_perturbations
from app.digestion.precision import update_dimension_precision_baseline
from app.digestion.significance import update_significance_pressure
from app.digestion.suppression import apply_suppression
from app.digestion.tension import update_tension_pressure
from app.ingest.state_deltas import Perturbation


def run_digestion_tick(
    state: FieldStateV1,
    *,
    perturbations: list[Perturbation],
    decay_rate: float,
    diffusion_rate: float,
    staleness_threshold_sec: float,
    store,
    significance_window_seconds: float,
    significance_check_interval_sec: float,
) -> FieldStateV1:
    apply_perturbations(state, perturbations)
    # now=state.generated_at, NOT datetime.now(): apply_perturbations() above
    # already defaults to state.generated_at as its own source of truth (see
    # that function's docstring), and worker.py sets state.generated_at = now
    # immediately before calling this tick -- keeps decay's staleness check
    # deterministic/replay-safe, no wall-clock call introduced here.
    apply_decay(
        state,
        decay_rate=decay_rate,
        now=state.generated_at,
        staleness_threshold_sec=staleness_threshold_sec,
    )
    apply_diffusion(state, diffusion_rate=diffusion_rate)
    apply_suppression(state)
    # Must run before update_dimension_precision_baseline(): that call scores
    # this tick's FINAL field_pressures() reading, which includes
    # deviation_pressure only once update_tension_pressure() has written it
    # (see orion/field/pressure.py::field_pressures_with_provenance()).
    update_tension_pressure(state)
    # Same ordering requirement as update_tension_pressure above -- must run
    # before update_dimension_precision_baseline() so precision tracking sees
    # THIS tick's sustained_load_pressure, not a stale carried-forward one,
    # on the tick where the throttle actually recomputes it.
    update_significance_pressure(
        state,
        store,
        window_seconds=significance_window_seconds,
        check_interval_sec=significance_check_interval_sec,
    )
    # Must run LAST: scores this tick's FINAL field_pressures() reading (see
    # update_dimension_precision_baseline()'s own docstring for why it can't
    # run before decay/diffusion/suppression -- and now tension/significance --
    # have already settled the tick).
    update_dimension_precision_baseline(state)
    return state
