from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from orion.schemas.field_state import FieldEdgeV1, FieldStateV1
from orion.field.pressure import collect_field_channel_pressures

from app.graph.lattice import load_lattice
from app.tensor.reconcile import reconcile_field_state_with_lattice

REPO = Path(__file__).resolve().parents[1]
LATTICE_PATH = REPO / "config" / "field" / "orion_field_topology.v1.yaml"
FIXED_TS = datetime(2026, 5, 24, 15, 0, tzinfo=timezone.utc)


def _stale_athena_state() -> FieldStateV1:
    """Simulates live stack: pre-execution topology edge + missing execution channels."""
    return FieldStateV1(
        generated_at=FIXED_TS,
        tick_id="tick_stale",
        node_vectors={
            "node:athena": {
                "availability": 0.95,
                "cpu_pressure": 0.42,
                "memory_pressure": 0.1,
                "custom_probe": 0.77,
            }
        },
        capability_vectors={
            "capability:orchestration": {
                "pressure": 0.2,
                "confidence": 0.9,
                "available_capacity": 0.85,
            }
        },
        edges=[
            FieldEdgeV1(
                source_id="node:athena",
                target_id="capability:orchestration",
                edge_type="node_capability",
                weight=0.90,
                channel_map={"cpu_pressure": "pressure"},
            )
        ],
        recent_perturbations=["perturb_exec_recent"],
    )


def test_reconcile_adds_cortex_exec_step_load_channel() -> None:
    lattice = load_lattice(LATTICE_PATH)
    reconciled = reconcile_field_state_with_lattice(_stale_athena_state(), lattice=lattice)
    assert "cortex_exec_step_load" in reconciled.node_vectors["node:athena"]
    assert reconciled.node_vectors["node:athena"]["cortex_exec_step_load"] == 0.0
    assert reconciled.node_vectors["node:athena"]["cpu_pressure"] == 0.42


def test_reconcile_refreshes_athena_orchestration_edge_mappings() -> None:
    lattice = load_lattice(LATTICE_PATH)
    reconciled = reconcile_field_state_with_lattice(_stale_athena_state(), lattice=lattice)
    edge = next(
        e
        for e in reconciled.edges
        if e.source_id == "node:athena" and e.target_id == "capability:orchestration"
    )
    assert edge.channel_map.get("cortex_exec_step_load") == "execution_pressure"
    assert edge.channel_map.get("execution_friction") == "reliability_pressure"
    assert edge.channel_map.get("failure_pressure") == "reliability_pressure"
    assert edge.channel_map.get("cpu_pressure") == "pressure"


def test_reconcile_preserves_existing_values_and_unknown_channels() -> None:
    lattice = load_lattice(LATTICE_PATH)
    reconciled = reconcile_field_state_with_lattice(_stale_athena_state(), lattice=lattice)
    assert reconciled.node_vectors["node:athena"]["availability"] == 0.95
    assert reconciled.node_vectors["node:athena"]["custom_probe"] == 0.77
    assert reconciled.recent_perturbations == ["perturb_exec_recent"]


def test_reconcile_adds_capability_pressure_channels_with_defaults() -> None:
    lattice = load_lattice(LATTICE_PATH)
    reconciled = reconcile_field_state_with_lattice(_stale_athena_state(), lattice=lattice)
    cap = reconciled.capability_vectors["capability:orchestration"]
    assert cap["execution_pressure"] == 0.0
    assert cap["reasoning_pressure"] == 0.0
    assert cap["reliability_pressure"] == 0.0
    assert cap["pressure"] == 0.2
    assert cap["confidence"] == 0.9


def test_reconciled_state_validates() -> None:
    lattice = load_lattice(LATTICE_PATH)
    reconciled = reconcile_field_state_with_lattice(_stale_athena_state(), lattice=lattice)
    roundtrip = FieldStateV1.model_validate(reconciled.model_dump(mode="json"))
    assert roundtrip.tick_id == "tick_stale"


def test_reconcile_prunes_retired_bus_observer_channels_from_every_node() -> None:
    """stream_backlog_health / delivery_confidence / stream_backlog_pressure were
    retired 2026-09-25 (fix/bus-observer-scope). They used to be single-observer
    channels seeded on node:athena only (2026-07-22 design correction); now no
    node may carry them, including the persisted 1.0s already sitting on
    node:athena in the live field state, and the merged field pressures must not
    report them either."""
    lattice = load_lattice(LATTICE_PATH)
    state = FieldStateV1(
        generated_at=FIXED_TS,
        tick_id="tick_retired_bus_observer_channels",
        node_vectors={
            # Verbatim live node:athena values, 2026-09-25T05:59:20Z.
            "node:athena": {
                "stream_backlog_health": 1.0,
                "delivery_confidence": 1.0,
                "stream_backlog_pressure": 0.0016,
            },
            "node:circe": {"stream_backlog_health": 0.0, "delivery_confidence": 0.0},
        },
    )
    reconciled = reconcile_field_state_with_lattice(state, lattice=lattice)
    for node_id, vec in reconciled.node_vectors.items():
        for retired in ("stream_backlog_health", "delivery_confidence", "stream_backlog_pressure"):
            assert retired not in vec, f"{node_id} still carries retired {retired}"

    channels, _ = collect_field_channel_pressures(reconciled)
    for retired in ("stream_backlog_health", "delivery_confidence", "stream_backlog_pressure"):
        assert retired not in channels


def test_reconcile_prunes_retired_capability_channels() -> None:
    """Live 2026-09-25: every capability vector still held transport_pressure
    (renamed 2026-07-24) and capability:orchestration held a dead
    stream_backlog_pressure. Both must go, provenance included."""
    lattice = load_lattice(LATTICE_PATH)
    state = FieldStateV1(
        generated_at=FIXED_TS,
        tick_id="tick_retired_capability_channels",
        capability_vectors={
            "capability:orchestration": {
                "pressure": 0.24,
                "transport_pressure": 0.0,
                "stream_backlog_pressure": 0.0,
            },
            "capability:transport": {"pressure": 0.31, "transport_pressure": 0.0},
        },
        capability_provenance={
            "capability:orchestration": {"stream_backlog_pressure": "capability:transport"},
        },
    )
    reconciled = reconcile_field_state_with_lattice(state, lattice=lattice)
    for cap_id, vec in reconciled.capability_vectors.items():
        assert "transport_pressure" not in vec, cap_id
        assert "stream_backlog_pressure" not in vec, cap_id
    assert reconciled.capability_vectors["capability:orchestration"]["pressure"] == 0.24
    assert "stream_backlog_pressure" not in reconciled.capability_provenance.get(
        "capability:orchestration", {}
    )
