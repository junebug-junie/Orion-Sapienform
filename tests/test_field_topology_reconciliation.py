from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from orion.schemas.field_state import FieldEdgeV1, FieldStateV1
from orion.self_state.scoring import collect_field_channel_pressures

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


def test_reconcile_adds_execution_load_channel() -> None:
    lattice = load_lattice(LATTICE_PATH)
    reconciled = reconcile_field_state_with_lattice(_stale_athena_state(), lattice=lattice)
    assert "execution_load" in reconciled.node_vectors["node:athena"]
    assert reconciled.node_vectors["node:athena"]["execution_load"] == 0.0
    assert reconciled.node_vectors["node:athena"]["cpu_pressure"] == 0.42


def test_reconcile_refreshes_athena_orchestration_edge_mappings() -> None:
    lattice = load_lattice(LATTICE_PATH)
    reconciled = reconcile_field_state_with_lattice(_stale_athena_state(), lattice=lattice)
    edge = next(
        e
        for e in reconciled.edges
        if e.source_id == "node:athena" and e.target_id == "capability:orchestration"
    )
    assert edge.channel_map.get("execution_load") == "execution_pressure"
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


def test_reconcile_seeds_bus_health_and_delivery_confidence_to_one_not_zero() -> None:
    # Regression guard, live post-deploy finding (2026-07-17): bus_health/
    # delivery_confidence are HIGHER_IS_BETTER_CHANNELS (min()-wins merge,
    # orion/self_state/scoring.py) but were missing from DEFAULT_NODE_VECTOR's
    # per-channel override table -- every node got the generic 0.0 default
    # from `{ch: 0.0 for ch in NODE_CHANNELS}`. Only node:athena (the
    # transport-bus observer) ever reports a real value for these two
    # channels; every other lattice node's untouched 0.0 always won the
    # min()-merge, permanently masking athena's real reading regardless of
    # actual bus health. Confirmed live: substrate_transport_bus_projection
    # showed bus_health=1.0 for node:athena while the merged
    # field_channel_corpus.v1 row read 0.0, 100% of rows, for the entire
    # post-deploy window -- this test reproduces that exact shape.
    lattice = load_lattice(LATTICE_PATH)
    state = FieldStateV1(
        generated_at=FIXED_TS,
        tick_id="tick_bus_health_default",
        node_vectors={"node:athena": {"bus_health": 1.0, "delivery_confidence": 1.0}},
    )
    reconciled = reconcile_field_state_with_lattice(state, lattice=lattice)
    # Every other lattice node must be seeded to 1.0 (presumed healthy until
    # reported otherwise), matching `availability`'s existing precedent --
    # not the generic 0.0 every other untouched channel gets.
    for node_id, vec in reconciled.node_vectors.items():
        if node_id == "node:athena":
            continue
        assert vec.get("bus_health") == 1.0, f"{node_id} bus_health should default to 1.0, not mask a real report"
        assert vec.get("delivery_confidence") == 1.0, f"{node_id} delivery_confidence should default to 1.0"

    channels, _ = collect_field_channel_pressures(reconciled)
    assert channels["bus_health"] == 1.0
    assert channels["delivery_confidence"] == 1.0
