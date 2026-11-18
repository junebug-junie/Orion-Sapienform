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


def test_fresh_degraded_heartbeat_never_triggers_remediation() -> None:
    """A heartbeat now says "degraded" while a bus listener reconnects or stays
    down (bus_service_chassis._listener_heartbeat_status). That must not restart
    the container: equilibrium's down_for_ms only measures heartbeat lateness, so
    an on-time degraded heartbeat has down_for_ms=0. If down_for_ms ever starts
    measuring time-in-degraded, this test must be revisited before shipping --
    otherwise a stuck listener turns into a restart loop after grace_sec."""
    svc = EquilibriumServiceState(
        service="landing-pad",
        status="degraded",
        last_seen_ts=datetime.now(timezone.utc),
        heartbeat_interval_sec=10.0,
        down_for_ms=0,
        details={"listeners": {"ok": False, "down": ["x"], "count": 1}},
    )
    bad, reason = equilibrium_status_for_service(_snapshot(svc), heartbeat_name="landing-pad", grace_sec=30)
    assert bad is False
    assert reason is None
