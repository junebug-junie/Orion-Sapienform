from __future__ import annotations

from copy import deepcopy

from orion.schemas.field_state import FieldEdgeV1, FieldStateV1

from app.graph.lattice import LatticeGraph
from app.tensor.channels import (
    CAPABILITY_CHANNELS,
    DEFAULT_CAPABILITY_VECTOR,
    DEFAULT_NODE_VECTOR,
    NODE_CHANNELS,
    RETIRED_LATTICE_NODES,
    RETIRED_NODE_CHANNELS,
    SINGLE_OBSERVER_NODE_CHANNELS,
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
    # SINGLE_OBSERVER_NODE_CHANNELS (channels.py): some channels can only
    # ever be legitimately reported by one specific node (stream_backlog_health/
    # delivery_confidence -> node:athena, the only bus-observer). Every
    # other node gets the key pruned here, every tick -- not just skipped
    # at seed time -- so a stale value already persisted on a non-owner
    # node from before this channel was added to this set (or from before
    # its owner was assigned) self-heals on the next reconcile instead of
    # living forever via the "preserve any existing key" loop above.
    for channel, owner_node_id in SINGLE_OBSERVER_NODE_CHANNELS.items():
        if node_id != owner_node_id:
            merged.pop(channel, None)
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


def _prune_every_node_vector(state: FieldStateV1) -> None:
    """Prune retired names and misplaced single-observer channels from EVERY
    node vector, including nodes absent from the lattice.

    `_ensure_node_vector()` above only runs for `lattice.nodes`, so its
    single-observer pruning has never reached a pseudo-node. Confirmed live
    2026-08-14: `node:rpc_timeout` is not in the lattice (which holds exactly
    atlas/athena/circe/prometheus) and had been carrying `delivery_confidence`
    and `stream_backlog_health` at 0.5 with a write 774s old, while
    `node:athena` -- the declared single observer -- reported 1.0 fresh. Both
    channels are HIGHER_IS_BETTER, so the merge takes min() and that stale 0.5
    won, pinning both at "half confident" indefinitely.

    The staleness rule (orion/field/pressure.py, 2026-08-14) stopped it
    WINNING. This stops it existing, which is the actual contract
    SINGLE_OBSERVER_NODE_CHANNELS always claimed: "every other node gets the
    key pruned here, every tick".

    Retired names are pruned from all nodes for the reason in
    channels.py::RETIRED_NODE_CHANNELS -- deliberately a named set rather than
    "anything not in NODE_CHANNELS", because _ensure_node_vector's
    preserve-undeclared-keys loop is load-bearing and must keep working.

    Mutates in place; the caller already deep-copied.
    """
    for node_id, vector in state.node_vectors.items():
        stamps = state.node_vector_updated_at.get(node_id)
        for channel in RETIRED_NODE_CHANNELS:
            vector.pop(channel, None)
            if stamps is not None:
                stamps.pop(channel, None)
        for channel, owner_node_id in SINGLE_OBSERVER_NODE_CHANNELS.items():
            if node_id != owner_node_id:
                vector.pop(channel, None)
                if stamps is not None:
                    stamps.pop(channel, None)


def _prune_retired_lattice_nodes(state: FieldStateV1) -> None:
    """Drop the whole node_vectors/node_vector_updated_at entry for a node
    that has been permanently removed from the lattice (RETIRED_LATTICE_NODES
    in channels.py), not just a renamed channel on a still-live node.

    _ensure_node_vector() above only ever runs for `lattice.nodes`, so once a
    node_id leaves the yaml nothing seeds, decays-with-intent, or perturbs its
    entry again -- it just sits in node_vectors forever at whatever value
    decay last left it, which every generic consumer iterating node_vectors
    reads as a real (if quiet) node. Unlike a pseudo-node (node:rpc_timeout,
    node:substrate.*), which is off-lattice by design and must be left alone,
    a retired lattice node has no reason to still be reported at all.
    """
    for node_id in RETIRED_LATTICE_NODES:
        state.node_vectors.pop(node_id, None)
        state.node_vector_updated_at.pop(node_id, None)


def reconcile_field_state_with_lattice(
    state: FieldStateV1,
    *,
    lattice: LatticeGraph,
) -> FieldStateV1:
    updated = deepcopy(state)
    for node_id in lattice.nodes:
        _ensure_node_vector(updated.node_vectors, node_id)
    _prune_every_node_vector(updated)
    _prune_retired_lattice_nodes(updated)
    for capability_id in lattice.capabilities:
        _ensure_capability_vector(updated.capability_vectors, capability_id)
    updated.edges = [
        FieldEdgeV1.model_validate(edge.model_dump(mode="json"))
        for edge in lattice.edges
    ]
    return updated
