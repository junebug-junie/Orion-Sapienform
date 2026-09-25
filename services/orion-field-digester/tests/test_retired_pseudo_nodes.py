"""node:rpc_timeout (phantom) and node:substrate.transport (retired) must be
pruned from persisted field state and must never be re-created by a delta.

Live shape, 2026-09-22/25: both keys present in the latest
substrate_field_state.field_json->'node_vectors'; node:rpc_timeout was a
dominant attention target in 42% of attention frames on 2026-09-19, and its
last write came from a transport_bus delta for bus "rpc_timeout" minted off
the RPC-timeout grammar trace `bus.transport:rpc_timeout:<corr>`.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.graph.lattice import load_lattice
from app.ingest.state_deltas import delta_to_perturbations
from app.tensor.channels import (
    PRUNED_NODE_IDS,
    RETIRED_LATTICE_NODES,
    RETIRED_PSEUDO_NODES,
)
from app.tensor.reconcile import reconcile_field_state_with_lattice
from orion.schemas.field_state import FieldStateV1
from orion.schemas.state_delta import StateDeltaV1

REPO = Path(__file__).resolve().parents[3]
NOW = datetime(2026, 9, 25, 0, 9, 30, tzinfo=timezone.utc)

# Exact live pressure_hints the reducer wrote for bus:rpc_timeout
# (substrate_transport_bus_projection, 2026-09-24T20:02:50Z).
_PHANTOM_HINTS = {
    "stream_backlog_health": 0.5,
    "delivery_confidence": 0.5,
    "stream_depth_pressure": 0.0,
    "backpressure": 0.0,
    "catalog_drift_pressure": 0.0,
    "observer_failure_pressure": 0.0,
    "stream_backlog_pressure": 0.0,
    "contract_pressure": 0.0,
    "reliability_pressure": 0.5,
}


def _lattice():
    return load_lattice(REPO / "config" / "field" / "orion_field_topology.v1.yaml")


def _transport_delta(node_id: str) -> StateDeltaV1:
    return StateDeltaV1(
        delta_id=f"delta_transport_{node_id}",
        target_projection="active_transport_bus_projection",
        target_kind="transport_bus",
        target_id=f"bus:{node_id}",
        operation="update",
        after={"node_id": node_id, "pressure_hints": dict(_PHANTOM_HINTS)},
        caused_by_event_ids=["gev_x"],
        reducer_id="transport_bus_reducer",
    )


def test_live_field_shape_prunes_both_phantoms_and_keeps_real_nodes() -> None:
    state = FieldStateV1(
        generated_at=NOW,
        tick_id="tick_f8d60e6d9db3",
        node_vectors={
            "node:athena": {"cpu_pressure": 0.3, "reliability_pressure": 0.0},
            "node:rpc_timeout": {
                "contract_pressure": 0.0,
                "reliability_pressure": 2.3886981790503957e-300,
                "catalog_drift_pressure": 0.0,
                "stream_backlog_pressure": 0.0,
                "observer_failure_pressure": 0.0,
            },
            "node:substrate.transport": {"prediction_error": 0.0},
            "node:substrate.bus_synaptic": {"prediction_error": 0.2},
            "node:substrate.chat": {"prediction_error": 0.1},
        },
        node_vector_updated_at={
            "node:rpc_timeout": {"reliability_pressure": NOW - timedelta(hours=4)},
            "node:substrate.transport": {"prediction_error": datetime(2026, 7, 26, 12, 0, 28, tzinfo=timezone.utc)},
        },
    )
    out = reconcile_field_state_with_lattice(state, lattice=_lattice())
    for gone in ("node:rpc_timeout", "node:substrate.transport"):
        assert gone not in out.node_vectors
        assert gone not in out.node_vector_updated_at
    # Live pseudo-nodes (incl. the transport successor) are untouched.
    assert out.node_vectors["node:substrate.bus_synaptic"]["prediction_error"] == 0.2
    assert out.node_vectors["node:substrate.chat"]["prediction_error"] == 0.1
    assert out.node_vectors["node:athena"]["cpu_pressure"] == 0.3


def test_stale_phantom_transport_delta_cannot_resurrect_node_rpc_timeout() -> None:
    """Reconcile runs before perturbations each tick, so a pre-fix delta still
    unapplied in the receipts table would otherwise re-create the node."""
    assert delta_to_perturbations(_transport_delta("rpc_timeout")) == []


def test_real_bus_delta_still_perturbs_athena() -> None:
    out = delta_to_perturbations(_transport_delta("athena"))
    assert out and {p.node_id for p in out} == {"node:athena"}


def test_pruned_set_is_the_union_and_never_a_live_lattice_node() -> None:
    assert PRUNED_NODE_IDS == frozenset(RETIRED_LATTICE_NODES) | frozenset(RETIRED_PSEUDO_NODES)
    assert not (PRUNED_NODE_IDS & set(_lattice().nodes))
    # The successor of node:substrate.transport must never be caught by this.
    assert "node:substrate.bus_synaptic" not in PRUNED_NODE_IDS
