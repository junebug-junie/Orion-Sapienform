from __future__ import annotations

from orion.schemas.state_delta import StateDeltaV1

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
