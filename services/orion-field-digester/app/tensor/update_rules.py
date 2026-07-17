from __future__ import annotations

from orion.schemas.field_state import FieldStateV1

from app.digestion.decay import apply_decay
from app.digestion.diffusion import apply_diffusion
from app.digestion.perturbation import apply_perturbations
from app.digestion.suppression import apply_suppression
from app.ingest.state_deltas import Perturbation


def run_digestion_tick(
    state: FieldStateV1,
    *,
    perturbations: list[Perturbation],
    decay_rate: float,
    diffusion_rate: float,
    staleness_threshold_sec: float,
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
    return state
