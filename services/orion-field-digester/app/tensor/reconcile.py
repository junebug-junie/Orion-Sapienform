from __future__ import annotations

from copy import deepcopy

from orion.schemas.field_state import FieldEdgeV1, FieldStateV1

from app.graph.lattice import LatticeGraph
from app.tensor.channels import (
    CAPABILITY_CHANNELS,
    DEFAULT_CAPABILITY_VECTOR,
    DEFAULT_NODE_VECTOR,
    NODE_CHANNELS,
)


def _ensure_node_vector(
    node_vectors: dict[str, dict[str, float]],
    node_id: str,
) -> dict[str, float]:
    existing = deepcopy(node_vectors.get(node_id, {}))
    merged = deepcopy(DEFAULT_NODE_VECTOR)
    merged.update(existing)
    for channel in NODE_CHANNELS:
        if channel not in merged:
            merged[channel] = DEFAULT_NODE_VECTOR[channel]
    for key, val in existing.items():
        if key not in NODE_CHANNELS:
            merged[key] = val
    node_vectors[node_id] = merged
    return merged


def _ensure_capability_vector(
    capability_vectors: dict[str, dict[str, float]],
    capability_id: str,
) -> dict[str, float]:
    existing = deepcopy(capability_vectors.get(capability_id, {}))
    merged = deepcopy(DEFAULT_CAPABILITY_VECTOR)
    merged.update(existing)
    for channel in CAPABILITY_CHANNELS:
        if channel not in merged:
            merged[channel] = DEFAULT_CAPABILITY_VECTOR[channel]
    for key, val in existing.items():
        if key not in CAPABILITY_CHANNELS:
            merged[key] = val
    capability_vectors[capability_id] = merged
    return merged


def reconcile_field_state_with_lattice(
    state: FieldStateV1,
    *,
    lattice: LatticeGraph,
) -> FieldStateV1:
    updated = deepcopy(state)
    for node_id in lattice.nodes:
        _ensure_node_vector(updated.node_vectors, node_id)
    for capability_id in lattice.capabilities:
        _ensure_capability_vector(updated.capability_vectors, capability_id)
    updated.edges = [
        FieldEdgeV1.model_validate(edge.model_dump(mode="json"))
        for edge in lattice.edges
    ]
    return updated
