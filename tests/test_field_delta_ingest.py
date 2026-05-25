from datetime import datetime, timezone

from orion.schemas.state_delta import StateDeltaV1

from app.ingest.state_deltas import delta_to_perturbations


def test_active_node_pressure_strain_maps_to_gpu_pressure() -> None:
    delta = StateDeltaV1(
        delta_id="delta_test1",
        target_projection="active_node_pressure_projection",
        target_kind="active_node_pressure",
        target_id="atlas",
        operation="reinforce",
        before=None,
        after={
            "node_id": "atlas",
            "active_pressures": ["strain"],
            "pressure_score": 0.72,
            "availability_status": "online",
            "suppressed_pressures": [],
            "capability_impacts": [],
            "evidence_event_ids": ["gev_1"],
            "last_updated_at": "2026-05-24T12:00:00+00:00",
        },
        caused_by_event_ids=["gev_1"],
        reducer_id="node_pressure_reducer",
    )
    perturbations = delta_to_perturbations(delta)
    assert len(perturbations) == 1
    assert perturbations[0].node_id == "node:atlas"
    assert perturbations[0].channel == "gpu_pressure"
    assert perturbations[0].intensity == 0.72
    assert perturbations[0].label == "delta_test1"


def test_node_biometrics_pressure_hints_and_expected_offline() -> None:
    delta = StateDeltaV1(
        delta_id="delta_bio_circe",
        target_projection="node_biometrics_projection",
        target_kind="node_biometrics",
        target_id="circe",
        operation="update",
        after={
            "node_id": "circe",
            "pressure_hints": {"gpu": 0.55, "strain": 0.3},
            "availability_status": "stale",
            "expected_online": False,
        },
        caused_by_event_ids=["gev_2"],
        reducer_id="node_biometrics_reducer",
    )
    perturbations = delta_to_perturbations(delta)
    channels = {p.channel: p.intensity for p in perturbations}
    assert channels["gpu_pressure"] == 0.55
    assert channels["cpu_pressure"] == 0.3
    assert channels["staleness"] == 0.5
    assert channels["expected_offline_suppression"] == 1.0
