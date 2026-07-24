"""Regression coverage for vision-window's bus-native SystemHealthV1 heartbeat wiring.

Part of the service-heartbeat rollout, see
docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md, following the
pilot-5 pattern proved out in PR #1350. orion-vision-window had no bus-native heartbeat before
this patch (confirmed via `grep -rl "BaseChassis"` returning nothing for this service).
`service` (the real WindowService, owning its own bus + recovery-store connections) is mocked
out here so this test exercises only the new heartbeat wiring in app/main.py's lifespan,
without opening a real bus connection.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

import app.main as window_main
from orion.core.bus.bus_service_chassis import HeartbeatOnly


def test_build_heartbeat_chassis_uses_window_settings() -> None:
    chassis = window_main.build_heartbeat_chassis()
    assert isinstance(chassis, HeartbeatOnly)
    assert chassis.cfg.service_name == window_main.settings.SERVICE_NAME
    assert chassis.cfg.service_version == window_main.settings.SERVICE_VERSION
    assert chassis.cfg.node_name == window_main.settings.NODE_NAME
    assert chassis.cfg.bus_url == window_main.settings.ORION_BUS_URL
    assert chassis.cfg.bus_enabled is True
    assert chassis.cfg.heartbeat_interval_sec == window_main.settings.HEARTBEAT_INTERVAL_SEC
    assert chassis.cfg.health_channel == "orion:system:health"


@pytest.mark.asyncio
async def test_lifespan_starts_and_stops_heartbeat_chassis() -> None:
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    with patch.object(window_main, "service") as mock_service, patch(
        "app.main.build_heartbeat_chassis", return_value=fake_chassis
    ):
        mock_service.start = AsyncMock()
        mock_service.stop = AsyncMock()

        app_stub = object()
        async with window_main.lifespan(app_stub):  # type: ignore[arg-type]
            assert window_main.heartbeat_chassis is fake_chassis
            fake_chassis.start_background.assert_awaited_once()

        fake_chassis.stop.assert_awaited_once()
        assert window_main.heartbeat_chassis is None
        mock_service.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_lifespan_survives_heartbeat_start_failure() -> None:
    """Heartbeat startup failure must not prevent the real window service from starting."""
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    fake_chassis.start_background.side_effect = RuntimeError("bus unreachable")
    with patch.object(window_main, "service") as mock_service, patch(
        "app.main.build_heartbeat_chassis", return_value=fake_chassis
    ):
        mock_service.start = AsyncMock()
        mock_service.stop = AsyncMock()

        app_stub = object()
        async with window_main.lifespan(app_stub):  # type: ignore[arg-type]
            assert window_main.heartbeat_chassis is None
            mock_service.start.assert_awaited_once()
