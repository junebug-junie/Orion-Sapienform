"""Regression coverage for cortex-gateway's bus-native SystemHealthV1 heartbeat wiring.

Part of the pilot-5 rollout, see
docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md.
orion-cortex-gateway had no bus-native heartbeat before this patch (confirmed via
`grep -rl "BaseChassis"` returning nothing for this service). `bus_client` (the service's real
BusClient, owning its own bus/RPC forks) is mocked out here so this test exercises only the new
heartbeat wiring in app/main.py's lifespan, without opening a real bus connection.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

import app.main as gateway_main
from orion.core.bus.bus_service_chassis import HeartbeatOnly


def test_build_heartbeat_chassis_uses_gateway_settings() -> None:
    chassis = gateway_main.build_heartbeat_chassis()
    assert isinstance(chassis, HeartbeatOnly)
    assert chassis.cfg.service_name == gateway_main.settings.service_name
    assert chassis.cfg.service_version == gateway_main.settings.service_version
    assert chassis.cfg.node_name == gateway_main.settings.node_name
    assert chassis.cfg.bus_url == gateway_main.settings.orion_bus_url
    assert chassis.cfg.bus_enabled is True
    assert chassis.cfg.heartbeat_interval_sec == gateway_main.settings.heartbeat_interval_sec
    assert chassis.cfg.health_channel == "orion:system:health"


@pytest.mark.asyncio
async def test_lifespan_starts_and_stops_heartbeat_chassis() -> None:
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    with patch("app.main.bus_client") as mock_bus_client, patch(
        "app.main.build_heartbeat_chassis", return_value=fake_chassis
    ):
        mock_bus_client.connect = AsyncMock()
        mock_bus_client.start_gateway_consumer = AsyncMock()
        mock_bus_client.close = AsyncMock()

        app_stub = object()
        async with gateway_main.lifespan(app_stub):  # type: ignore[arg-type]
            assert gateway_main.heartbeat_chassis is fake_chassis
            fake_chassis.start_background.assert_awaited_once()

        fake_chassis.stop.assert_awaited_once()
        assert gateway_main.heartbeat_chassis is None
        mock_bus_client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_lifespan_survives_heartbeat_start_failure() -> None:
    """Heartbeat startup failure must not prevent the gateway's real consumer from starting."""
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    fake_chassis.start_background.side_effect = RuntimeError("bus unreachable")
    with patch("app.main.bus_client") as mock_bus_client, patch(
        "app.main.build_heartbeat_chassis", return_value=fake_chassis
    ):
        mock_bus_client.connect = AsyncMock()
        mock_bus_client.start_gateway_consumer = AsyncMock()
        mock_bus_client.close = AsyncMock()

        app_stub = object()
        async with gateway_main.lifespan(app_stub):  # type: ignore[arg-type]
            assert gateway_main.heartbeat_chassis is None
            mock_bus_client.start_gateway_consumer.assert_awaited_once()
