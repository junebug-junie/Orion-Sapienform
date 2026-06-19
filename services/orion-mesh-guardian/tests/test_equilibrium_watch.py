from __future__ import annotations

from datetime import datetime, timezone

from orion.schemas.telemetry.system_health import EquilibriumServiceState, EquilibriumSnapshotV1

from app.equilibrium_watch import equilibrium_status_for_service


def _snapshot(*services: EquilibriumServiceState) -> EquilibriumSnapshotV1:
    return EquilibriumSnapshotV1(
        source_service="equilibrium",
        producer_boot_id="boot",
        generated_at=datetime.now(timezone.utc),
        grace_multiplier=3.0,
        windows_sec=[300],
        expected_services=[s.service for s in services],
        services=list(services),
    )


def test_equilibrium_ok_when_no_snapshot_yet() -> None:
    bad, reason = equilibrium_status_for_service(None, heartbeat_name="landing-pad", grace_sec=30)
    assert bad is False
    assert reason is None


def test_equilibrium_bad_when_down() -> None:
    svc = EquilibriumServiceState(
        service="landing-pad",
        status="down",
        last_seen_ts=datetime.now(timezone.utc),
        heartbeat_interval_sec=10.0,
        down_for_ms=5000,
    )
    bad, reason = equilibrium_status_for_service(_snapshot(svc), heartbeat_name="landing-pad", grace_sec=30)
    assert bad is True
    assert reason == "down"


def test_equilibrium_bad_when_degraded_beyond_grace() -> None:
    svc = EquilibriumServiceState(
        service="landing-pad",
        status="degraded",
        last_seen_ts=datetime.now(timezone.utc),
        heartbeat_interval_sec=10.0,
        down_for_ms=31000,
    )
    bad, reason = equilibrium_status_for_service(_snapshot(svc), heartbeat_name="landing-pad", grace_sec=30)
    assert bad is True
    assert reason == "degraded_beyond_grace"
