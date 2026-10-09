from __future__ import annotations

from app.main import build_heartbeat_chassis
from app.settings import Settings
from orion.core.bus.bus_service_chassis import HeartbeatOnly


def test_heartbeat_chassis_builds_with_service_identity() -> None:
    chassis = build_heartbeat_chassis(Settings(ORION_BUS_ENABLED=False))
    assert isinstance(chassis, HeartbeatOnly)
