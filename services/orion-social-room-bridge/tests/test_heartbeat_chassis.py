"""Regression coverage for social-room-bridge's bus-native SystemHealthV1 heartbeat wiring.

Part of the service-heartbeat rollout, see
docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md, following the
pilot-5 pattern proved out in PR #1350. orion-social-room-bridge had no bus-native heartbeat
before this patch (confirmed via `grep -rl "BaseChassis"` returning nothing for this service).
`service` (the real SocialRoomBridgeService, owning its own bus connection) is mocked out here
so this test exercises only the new heartbeat wiring in app/main.py's lifespan, without opening
a real bus connection.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

import app.main as bridge_main
from orion.core.bus.bus_service_chassis import HeartbeatOnly


def test_build_heartbeat_chassis_uses_bridge_settings() -> None:
    chassis = bridge_main.build_heartbeat_chassis()
    assert isinstance(chassis, HeartbeatOnly)
    assert chassis.cfg.service_name == bridge_main.settings.service_name
    assert chassis.cfg.service_version == bridge_main.settings.service_version
    assert chassis.cfg.node_name == bridge_main.settings.node_name
    assert chassis.cfg.bus_url == bridge_main.settings.orion_bus_url
    assert chassis.cfg.bus_enabled == bridge_main.settings.orion_bus_enabled
    assert chassis.cfg.heartbeat_interval_sec == bridge_main.settings.heartbeat_interval_sec
    assert chassis.cfg.health_channel == "orion:system:health"


@pytest.mark.asyncio
async def test_lifespan_starts_and_stops_heartbeat_chassis() -> None:
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    with patch.object(bridge_main, "service") as mock_service, patch(
        "app.main.build_heartbeat_chassis", return_value=fake_chassis
    ):
        mock_service.start = AsyncMock()
        mock_service.stop = AsyncMock()

        app_stub = object()
        async with bridge_main.lifespan(app_stub):  # type: ignore[arg-type]
            assert bridge_main.heartbeat_chassis is fake_chassis
            fake_chassis.start_background.assert_awaited_once()

        fake_chassis.stop.assert_awaited_once()
        assert bridge_main.heartbeat_chassis is None
        mock_service.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_lifespan_survives_heartbeat_start_failure() -> None:
    """Heartbeat startup failure must not prevent the real bridge service from starting."""
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    fake_chassis.start_background.side_effect = RuntimeError("bus unreachable")
    with patch.object(bridge_main, "service") as mock_service, patch(
        "app.main.build_heartbeat_chassis", return_value=fake_chassis
    ):
        mock_service.start = AsyncMock()
        mock_service.stop = AsyncMock()

        app_stub = object()
        async with bridge_main.lifespan(app_stub):  # type: ignore[arg-type]
            assert bridge_main.heartbeat_chassis is None
            mock_service.start.assert_awaited_once()
