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
FIXED_TS = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


def test_execution_run_delta_maps_to_node_channels() -> None:
    delta = StateDeltaV1(
        delta_id="delta_exec_1",
        target_projection="active_execution_trajectory",
        target_kind="execution_run",
        target_id="cortex.exec:athena:corr-1",
        operation="update",
        after={
            "node_id": "athena",
            "pressure_hints": {
                "execution_load": 0.45,
                "execution_friction": 0.05,
                "reasoning_load": 0.35,
                "failure_pressure": 0.0,
            },
        },
        caused_by_event_ids=["gev_1"],
        reducer_id="execution_trajectory_reducer",
    )
    perturbations = delta_to_perturbations(delta)
    channels = {p.channel: p.intensity for p in perturbations}
    assert channels["execution_load"] == 0.45
    assert channels["reasoning_load"] == 0.35
    assert perturbations[0].node_id == "node:athena"


def test_execution_perturbations_use_replace_mode() -> None:
    """Two consecutive execution_run deltas should not accumulate — second replaces first."""
    delta_a = StateDeltaV1(
        delta_id="delta_exec_a",
        target_projection="active_execution_trajectory",
        target_kind="execution_run",
        target_id="cortex.exec:athena:corr-a",
        operation="update",
        after={"node_id": "athena", "pressure_hints": {"execution_load": 1.0, "failure_pressure": 1.0}},
        caused_by_event_ids=["gev_a"],
        reducer_id="execution_trajectory_reducer",
    )
    delta_b = StateDeltaV1(
        delta_id="delta_exec_b",
        target_projection="active_execution_trajectory",
        target_kind="execution_run",
        target_id="cortex.exec:athena:corr-b",
        operation="update",
        after={"node_id": "athena", "pressure_hints": {"execution_load": 0.25, "failure_pressure": 0.0}},
        caused_by_event_ids=["gev_b"],
        reducer_id="execution_trajectory_reducer",
    )
    lattice = load_lattice(LATTICE)
    field = empty_field_state(lattice=lattice, now=FIXED_TS, tick_id="tick_replace")

    # Apply first (maxed) delta then a second (low) delta in the same tick
    all_perturbations = delta_to_perturbations(delta_a) + delta_to_perturbations(delta_b)
    field = run_digestion_tick(field, perturbations=all_perturbations, decay_rate=1.0, diffusion_rate=1.0)

    node = field.node_vectors.get("node:athena") or {}
    # replace mode: last write wins — execution_load should be 0.25, not 1.0
    assert node.get("execution_load") == 0.25, f"expected 0.25 got {node.get('execution_load')}"
    assert node.get("failure_pressure") == 0.0, f"expected 0.0 got {node.get('failure_pressure')}"

    # verify all execution perturbations carry replace mode
    for p in delta_to_perturbations(delta_a):
        assert p.mode == "replace", f"channel {p.channel} should be replace mode"


def test_execution_perturbations_diffuse_to_orchestration_capability() -> None:
    lattice = load_lattice(LATTICE)
    delta = StateDeltaV1(
        delta_id="delta_exec_2",
        target_projection="active_execution_trajectory",
        target_kind="execution_run",
        target_id="cortex.exec:athena:corr-2",
        operation="update",
        after={
            "node_id": "athena",
            "pressure_hints": {
                "execution_load": 0.8,
                "execution_friction": 0.1,
                "failure_pressure": 0.2,
            },
        },
        caused_by_event_ids=["gev_2"],
        reducer_id="execution_trajectory_reducer",
    )
    field = empty_field_state(lattice=lattice, now=FIXED_TS, tick_id="tick_exec")
    field = run_digestion_tick(
        field,
        perturbations=delta_to_perturbations(delta),
        decay_rate=1.0,
        diffusion_rate=1.0,
    )
    cap = field.capability_vectors.get("capability:orchestration") or {}
    assert cap.get("execution_pressure", 0.0) > 0.0
