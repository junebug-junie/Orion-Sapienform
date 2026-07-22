from __future__ import annotations

from datetime import datetime, timezone

from app.graph.lattice import LatticeGraph
from app.tensor.channels import SINGLE_OBSERVER_NODE_CHANNELS
from app.tensor.reconcile import _ensure_node_vector, reconcile_field_state_with_lattice

from orion.schemas.field_state import FieldStateV1

NOW = datetime(2026, 7, 22, tzinfo=timezone.utc)


def _lattice(nodes: list[str]) -> LatticeGraph:
    return LatticeGraph(nodes=nodes, capabilities=[], edges=[])


def _state(node_vectors: dict[str, dict[str, float]] | None = None) -> FieldStateV1:
    return FieldStateV1(
        generated_at=NOW,
        tick_id="tick_reconcile_test",
        node_vectors=node_vectors or {},
        edges=[],
    )


def test_owner_node_gets_seeded_with_single_observer_channels() -> None:
    vec = _ensure_node_vector({}, "node:athena")
    for channel in SINGLE_OBSERVER_NODE_CHANNELS:
        assert channel in vec


def test_non_owner_node_never_gets_single_observer_channels_seeded() -> None:
    vec = _ensure_node_vector({}, "node:atlas")
    for channel in SINGLE_OBSERVER_NODE_CHANNELS:
        assert channel not in vec


def test_stale_pre_fix_value_on_non_owner_node_self_heals_on_reconcile() -> None:
    """The live bug this fix closes: a non-owner node (atlas) already had a
    stale bus_health=0.0/delivery_confidence=0.0 persisted from before this
    fix existed. reconcile must actively prune it, not just stop adding it
    to new nodes -- the pre-existing "preserve any existing key" behavior
    would otherwise carry it forward forever."""
    node_vectors = {
        "node:atlas": {"bus_health": 0.0, "delivery_confidence": 0.0, "cpu_pressure": 0.3},
    }
    vec = _ensure_node_vector(node_vectors, "node:atlas")
    assert "bus_health" not in vec
    assert "delivery_confidence" not in vec
    # Unrelated, legitimately-per-node channels are untouched.
    assert vec["cpu_pressure"] == 0.3


def test_owner_nodes_real_value_is_preserved_across_reconcile() -> None:
    node_vectors = {
        "node:athena": {"bus_health": 1.0, "delivery_confidence": 1.0},
    }
    vec = _ensure_node_vector(node_vectors, "node:athena")
    assert vec["bus_health"] == 1.0
    assert vec["delivery_confidence"] == 1.0


def test_full_reconcile_prunes_stale_values_across_the_whole_lattice() -> None:
    """End-to-end: reconcile_field_state_with_lattice() over a 4-node
    lattice (matching the real athena/atlas/circe/prometheus topology)
    leaves only node:athena carrying bus_health/delivery_confidence."""
    state = _state(
        node_vectors={
            "node:atlas": {"bus_health": 0.0, "delivery_confidence": 0.0},
            "node:circe": {"bus_health": 0.0, "delivery_confidence": 0.0},
            "node:athena": {"bus_health": 1.0, "delivery_confidence": 1.0},
            "node:prometheus": {"bus_health": 0.0, "delivery_confidence": 0.0},
        }
    )
    lattice = _lattice(["node:atlas", "node:circe", "node:athena", "node:prometheus"])
    updated = reconcile_field_state_with_lattice(state, lattice=lattice)

    for node_id in ["node:atlas", "node:circe", "node:prometheus"]:
        assert "bus_health" not in updated.node_vectors[node_id]
        assert "delivery_confidence" not in updated.node_vectors[node_id]
    assert updated.node_vectors["node:athena"]["bus_health"] == 1.0
    assert updated.node_vectors["node:athena"]["delivery_confidence"] == 1.0
