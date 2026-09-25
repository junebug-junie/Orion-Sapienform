"""SINGLE_OBSERVER_NODE_CHANNELS mechanism tests.

The map's only two real entries (stream_backlog_health, delivery_confidence)
were retired 2026-09-25 (fix/bus-observer-scope), leaving it empty. The
mechanism is kept for the next genuinely single-observer channel, so these
tests register a synthetic one rather than going vacuous over an empty dict.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.graph.lattice import LatticeGraph
from app.tensor.channels import SINGLE_OBSERVER_NODE_CHANNELS
from app.tensor.reconcile import _ensure_node_vector, reconcile_field_state_with_lattice

from orion.schemas.field_state import FieldStateV1

NOW = datetime(2026, 7, 22, tzinfo=timezone.utc)
_CH = "synthetic_single_observer_health"


@pytest.fixture(autouse=True)
def _register_synthetic_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    # setitem on the shared dict object: reconcile.py imported the same dict.
    monkeypatch.setitem(SINGLE_OBSERVER_NODE_CHANNELS, _CH, "node:athena")


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
    # rpc_timeout_pressure's owner is the off-lattice node:substrate.rpc_delivery,
    # which reconcile never seeds (it only exists once the bridge writes it).
    for channel, owner in SINGLE_OBSERVER_NODE_CHANNELS.items():
        if owner == "node:athena":
            assert channel in vec
        else:
            assert channel not in vec


def test_retired_entries_are_gone_from_the_real_map() -> None:
    real = {k: v for k, v in SINGLE_OBSERVER_NODE_CHANNELS.items() if k != _CH}
    assert "stream_backlog_health" not in real
    assert "delivery_confidence" not in real


def test_stale_value_on_non_owner_node_self_heals_on_reconcile() -> None:
    node_vectors = {"node:circe": {_CH: 0.0, "cpu_pressure": 0.3}}
    vec = _ensure_node_vector(node_vectors, "node:circe")
    assert _CH not in vec
    # Unrelated, legitimately-per-node channels are untouched.
    assert vec["cpu_pressure"] == 0.3


def test_owner_nodes_real_value_is_preserved_across_reconcile() -> None:
    vec = _ensure_node_vector({"node:athena": {_CH: 1.0}}, "node:athena")
    assert vec[_CH] == 1.0


def test_full_reconcile_prunes_stale_values_across_the_whole_lattice() -> None:
    state = _state(
        node_vectors={
            "node:circe": {_CH: 0.0},
            "node:athena": {_CH: 1.0},
            "node:prometheus": {_CH: 0.0},
        }
    )
    lattice = _lattice(["node:circe", "node:athena", "node:prometheus"])
    updated = reconcile_field_state_with_lattice(state, lattice=lattice)

    for node_id in ["node:circe", "node:prometheus"]:
        assert _CH not in updated.node_vectors[node_id]
    assert updated.node_vectors["node:athena"][_CH] == 1.0
