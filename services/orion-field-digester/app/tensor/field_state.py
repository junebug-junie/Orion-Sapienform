from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from uuid import uuid4

from orion.schemas.field_state import FieldStateV1

from app.graph.lattice import LatticeGraph
from app.tensor.channels import DEFAULT_CAPABILITY_VECTOR, DEFAULT_NODE_VECTOR


def new_tick_id() -> str:
    return f"tick_{uuid4().hex[:12]}"


def empty_field_state(*, lattice: LatticeGraph, now: datetime, tick_id: str) -> FieldStateV1:
    node_vectors = {nid: deepcopy(DEFAULT_NODE_VECTOR) for nid in lattice.nodes}
    capability_vectors = {cid: deepcopy(DEFAULT_CAPABILITY_VECTOR) for cid in lattice.capabilities}
    return FieldStateV1(
        generated_at=now,
        tick_id=tick_id,
        node_vectors=node_vectors,
        capability_vectors=capability_vectors,
        edges=list(lattice.edges),
        recent_perturbations=[],
    )
