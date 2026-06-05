from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from orion.schemas.state_delta import StateDeltaV1

from app.ingest.state_deltas import delta_to_perturbations
from app.graph.lattice import load_lattice
from app.tensor.field_state import empty_field_state
from app.tensor.update_rules import run_digestion_tick

REPO_ROOT = Path(__file__).resolve().parents[1]
LATTICE = REPO_ROOT / "config" / "field" / "orion_field_topology.v1.yaml"
FIXED_TS = datetime(2026, 5, 25, 23, 30, 10, tzinfo=timezone.utc)


def test_transport_bus_delta_maps_to_node_channels() -> None:
    delta = StateDeltaV1(
        delta_id="delta_transport_1",
        target_projection="active_transport_bus_projection",
        target_kind="transport_bus",
        target_id="bus:athena",
        operation="update",
        after={
            "node_id": "athena",
            "pressure_hints": {
                "bus_health": 1.0,
                "delivery_confidence": 1.0,
                "catalog_drift_pressure": 1.0,
                "contract_pressure": 1.0,
                "transport_pressure": 0.0,
            },
        },
        caused_by_event_ids=["gev_1"],
        reducer_id="transport_bus_reducer",
    )
    perturbations = delta_to_perturbations(delta)
    channels = {p.channel: p.intensity for p in perturbations}
    assert channels["contract_pressure"] == 1.0
    assert channels["bus_health"] == 1.0
    assert perturbations[0].node_id == "node:athena"


def test_transport_perturbations_diffuse_to_transport_capability() -> None:
    lattice = load_lattice(LATTICE)
    assert "capability:transport" in lattice.capabilities
    delta = StateDeltaV1(
        delta_id="delta_transport_2",
        target_projection="active_transport_bus_projection",
        target_kind="transport_bus",
        target_id="bus:athena",
        operation="update",
        after={
            "node_id": "athena",
            "pressure_hints": {
                "catalog_drift_pressure": 1.0,
                "contract_pressure": 1.0,
            },
        },
        caused_by_event_ids=["gev_2"],
        reducer_id="transport_bus_reducer",
    )
    field = empty_field_state(lattice=lattice, now=FIXED_TS, tick_id="tick_transport")
    field = run_digestion_tick(
        field,
        perturbations=delta_to_perturbations(delta),
        decay_rate=1.0,
        diffusion_rate=1.0,
    )
    cap = field.capability_vectors.get("capability:transport") or {}
    assert cap.get("contract_pressure", 0.0) > 0.0


def test_field_digester_store_does_not_query_grammar_events() -> None:
    store_path = REPO_ROOT / "services" / "orion-field-digester" / "app" / "store.py"
    src = store_path.read_text(encoding="utf-8")
    assert "grammar_events" not in src
