from __future__ import annotations

from orion.schemas.field_state import FieldStateV1

from app.ingest.state_deltas import Perturbation


def apply_perturbations(state: FieldStateV1, perturbations: list[Perturbation]) -> FieldStateV1:
    for p in perturbations:
        node_vec = state.node_vectors.setdefault(p.node_id, {})
        if p.mode == "replace":
            node_vec[p.channel] = max(0.0, min(1.0, p.intensity))
        elif p.channel == "availability":
            node_vec[p.channel] = min(node_vec.get(p.channel, 1.0), p.intensity)
        else:
            node_vec[p.channel] = min(1.0, node_vec.get(p.channel, 0.0) + p.intensity)
        if p.label not in state.recent_perturbations:
            state.recent_perturbations.append(p.label)
    state.recent_perturbations = state.recent_perturbations[-20:]
    return state
