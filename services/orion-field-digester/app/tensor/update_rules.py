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
) -> FieldStateV1:
    apply_perturbations(state, perturbations)
    apply_decay(state, decay_rate=decay_rate)
    apply_diffusion(state, diffusion_rate=diffusion_rate)
    apply_suppression(state)
    return state
