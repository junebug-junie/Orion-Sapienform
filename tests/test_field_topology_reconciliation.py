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


def test_reconcile_scopes_stream_backlog_health_and_delivery_confidence_to_node_athena_only() -> None:
    # Design correction (2026-07-22), superseding the 2026-07-17 fix this
    # test used to guard. That fix (default 0.0 -> 1.0 for non-reporting
    # nodes) treated the symptom as a data problem, but it only helped
    # *newly* reconciled nodes -- reconcile's "preserve any existing value"
    # behavior meant a node with an already-persisted stale 0.0 from before
    # that patch deployed could never self-correct. Confirmed live: athena's
    # real, fresh report was stream_backlog_health=1.0, but atlas's persisted 0.0 (never
    # once perturbed, from before the fix) still won the min()-merge and
    # permanently masked it in field_channel_corpus.v1 and every SelfStateV1
    # coherence score (self_state_policy.v1.yaml maps both channels to
    # `coherence`).
    #
    # The real fix: there is exactly one bus, it runs on athena, and only
    # athena's bus-observer ever produces a real reading -- atlas/circe/
    # prometheus have no standing to have an opinion on bus health at all.
    # SINGLE_OBSERVER_NODE_CHANNELS (app/tensor/channels.py) makes this
    # explicit: only node:athena ever gets these two channels seeded, and
    # they're actively pruned from every other node on every reconcile tick
    # -- self-healing for exactly the stale-persisted-value case above,
    # without needing a manual data migration.
    lattice = load_lattice(LATTICE_PATH)
    state = FieldStateV1(
        generated_at=FIXED_TS,
        tick_id="tick_stream_backlog_health_default",
        node_vectors={
            "node:athena": {"stream_backlog_health": 1.0, "delivery_confidence": 1.0},
            # Simulates the live bug: a stale pre-fix 0.0 already persisted
            # on a non-owner node.
            "node:atlas": {"stream_backlog_health": 0.0, "delivery_confidence": 0.0},
        },
    )
    reconciled = reconcile_field_state_with_lattice(state, lattice=lattice)
    # Every other lattice node must have no opinion at all -- not a default
    # value, an absent key -- so it can never win (or lose) the min()-merge.
    for node_id, vec in reconciled.node_vectors.items():
        if node_id == "node:athena":
            continue
        assert "stream_backlog_health" not in vec, f"{node_id} should have no stream_backlog_health entry at all"
        assert "delivery_confidence" not in vec, f"{node_id} should have no delivery_confidence entry at all"

    channels, _ = collect_field_channel_pressures(reconciled)
    assert channels["stream_backlog_health"] == 1.0
    assert channels["delivery_confidence"] == 1.0
