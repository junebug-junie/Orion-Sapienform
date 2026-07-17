from __future__ import annotations

from datetime import datetime, timezone

from orion.schemas.field_state import FieldStateV1
from orion.schemas.state_delta import StateDeltaV1

from app.digestion.perturbation import apply_perturbations
from app.ingest.state_deltas import delta_to_perturbations


def _make_node_biometrics_delta(
    *,
    operation: str = "update",
    node_id: str = "atlas",
    pressure_hints: dict | None = None,
    availability_status: str = "online",
    expected_online: bool | None = True,
) -> StateDeltaV1:
    after: dict = {
        "node_id": node_id,
        "pressure_hints": pressure_hints or {},
        "availability_status": availability_status,
        "expected_online": expected_online,
    }
    return StateDeltaV1(
        delta_id="delta_node_bio_1",
        target_projection="node_biometrics",
        target_kind="node_biometrics",
        target_id=f"node:{node_id}",
        operation=operation,  # type: ignore[arg-type]
        after=after,
        caused_by_event_ids=["gev_bio_1"],
        reducer_id="biometrics_node_reducer",
    )


def test_node_biometrics_delta_produces_memory_thermal_disk_pressure() -> None:
    delta = _make_node_biometrics_delta(
        pressure_hints={
            "memory_pressure": 0.61,
            "thermal_pressure": 0.33,
            "disk_pressure": 0.12,
        }
    )
    perturbations = delta_to_perturbations(delta)
    channels = {p.channel: p.intensity for p in perturbations}
    assert channels["memory_pressure"] == 0.61
    assert channels["thermal_pressure"] == 0.33
    assert channels["disk_pressure"] == 0.12
    assert all(p.node_id == "node:atlas" for p in perturbations)


def test_node_biometrics_delta_still_produces_gpu_and_cpu_pressure() -> None:
    # gpu/strain (-> cpu_pressure) are a different agent's precedent pattern;
    # confirm the additive change did not disturb them.
    delta = _make_node_biometrics_delta(pressure_hints={"gpu": 0.75, "strain": 0.4})
    perturbations = delta_to_perturbations(delta)
    channels = {p.channel: p.intensity for p in perturbations}
    assert channels["gpu_pressure"] == 0.75
    assert channels["cpu_pressure"] == 0.4
    assert "memory_pressure" not in channels
    assert "thermal_pressure" not in channels
    assert "disk_pressure" not in channels


def test_node_biometrics_delta_missing_hardware_hints_produces_no_perturbation() -> None:
    delta = _make_node_biometrics_delta(pressure_hints={})
    perturbations = delta_to_perturbations(delta)
    channels = [p.channel for p in perturbations]
    assert "memory_pressure" not in channels
    assert "thermal_pressure" not in channels
    assert "disk_pressure" not in channels


def test_node_biometrics_noop_delta_skipped() -> None:
    delta = _make_node_biometrics_delta(
        operation="noop",
        pressure_hints={"memory_pressure": 0.9, "thermal_pressure": 0.9, "disk_pressure": 0.9},
    )
    perturbations = delta_to_perturbations(delta)
    assert perturbations == []


def test_memory_thermal_disk_perturbations_use_replace_mode() -> None:
    # node_reducer.py emits one StateDeltaV1 per grammar event in a
    # node_biometrics trace (trace_started/atoms/edges/trace_ended -- not
    # just the atom that first sets a hint), and each carries the cumulative
    # pressure_hints forward. A single ~22-event trace therefore produces
    # well over a dozen deltas that all still contain "memory_pressure"/
    # "thermal_pressure"/"disk_pressure" once those hints are first set. If
    # these used the default mode="add", replaying all of those deltas in
    # one field-digester tick would re-add the same intensity that many
    # times and saturate the channel to 1.0 regardless of real load -- so
    # they must use mode="replace" (same precedent as execution_run/
    # chat_turn below).
    delta = _make_node_biometrics_delta(
        pressure_hints={
            "memory_pressure": 0.61,
            "thermal_pressure": 0.33,
            "disk_pressure": 0.12,
        }
    )
    perturbations = delta_to_perturbations(delta)
    modes = {p.channel: p.mode for p in perturbations}
    assert modes["memory_pressure"] == "replace"
    assert modes["thermal_pressure"] == "replace"
    assert modes["disk_pressure"] == "replace"


def test_memory_thermal_disk_perturbations_do_not_saturate_across_repeated_deltas() -> None:
    # Regression for the fan-out described above: simulate one field-digester
    # tick that (correctly, per how node_reducer.py actually behaves) sees
    # the same node_biometrics hint value repeated across many deltas from
    # one trace, and confirm the lattice channel ends up at the real
    # intensity -- not clamped to 1.0 by repeated addition.
    delta = _make_node_biometrics_delta(
        pressure_hints={"memory_pressure": 0.3, "thermal_pressure": 0.3, "disk_pressure": 0.3}
    )
    perturbations = []
    for _ in range(16):  # representative of the per-trace event fan-out
        perturbations.extend(delta_to_perturbations(delta))

    state = FieldStateV1(
        generated_at=datetime(2026, 7, 17, tzinfo=timezone.utc),
        tick_id="tick_saturation_regression",
        node_vectors={},
        capability_vectors={},
        edges=[],
    )
    apply_perturbations(state, perturbations)
    node_vec = state.node_vectors["node:atlas"]
    assert node_vec["memory_pressure"] == 0.3
    assert node_vec["thermal_pressure"] == 0.3
    assert node_vec["disk_pressure"] == 0.3
