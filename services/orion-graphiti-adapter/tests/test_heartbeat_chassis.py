"""Regression coverage for graphiti-adapter's bus-native SystemHealthV1 heartbeat wiring.

Part of the wider service-heartbeat rollout following pilot-5 (PR #1350), see
docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md.
orion-graphiti-adapter had no bus connection at all before this patch (confirmed via
`grep -rl "BaseChassis\\|SystemHealthV1"` returning nothing for this service) -- it is pure
HTTP + Postgres + FalkorDB. This test does not open a real bus connection -- it verifies (a)
`build_heartbeat_chassis()` wires ChassisConfig from this service's own (uppercase-alias)
settings, and (b) the FastAPI lifespan actually starts and stops that chassis, using a mocked
chassis so the rest of lifespan (asyncpg pool, FalkorDB indices) stays a no-op for this test.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

import app.main as graphiti_main
from orion.core.bus.bus_service_chassis import HeartbeatOnly


def test_build_heartbeat_chassis_uses_graphiti_settings() -> None:
    chassis = graphiti_main.build_heartbeat_chassis()
    assert isinstance(chassis, HeartbeatOnly)
    assert chassis.cfg.service_name == graphiti_main.settings.SERVICE_NAME
    assert chassis.cfg.service_version == graphiti_main.settings.SERVICE_VERSION
    assert chassis.cfg.node_name == graphiti_main.settings.NODE_NAME
    assert chassis.cfg.bus_url == graphiti_main.settings.ORION_BUS_URL
    assert chassis.cfg.bus_enabled == graphiti_main.settings.ORION_BUS_ENABLED
    assert chassis.cfg.heartbeat_interval_sec == graphiti_main.settings.HEARTBEAT_INTERVAL_SEC
    assert chassis.cfg.health_channel == "orion:system:health"


@pytest.mark.asyncio
async def test_lifespan_starts_and_stops_heartbeat_chassis() -> None:
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    with patch.object(graphiti_main.settings, "POSTGRES_URI", ""), patch.object(
        graphiti_main.settings, "FALKORDB_ENABLED", False
    ), patch("app.main.build_heartbeat_chassis", return_value=fake_chassis):
        app_stub = graphiti_main.FastAPI()
        async with graphiti_main.lifespan(app_stub):
            assert graphiti_main.heartbeat_chassis is fake_chassis
            fake_chassis.start_background.assert_awaited_once()

        fake_chassis.stop.assert_awaited_once()
        assert graphiti_main.heartbeat_chassis is None


@pytest.mark.asyncio
async def test_lifespan_survives_heartbeat_start_failure() -> None:
    """Heartbeat startup failure must not prevent the rest of lifespan from completing."""
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    fake_chassis.start_background.side_effect = RuntimeError("bus unreachable")
    with patch.object(graphiti_main.settings, "POSTGRES_URI", ""), patch.object(
        graphiti_main.settings, "FALKORDB_ENABLED", False
    ), patch("app.main.build_heartbeat_chassis", return_value=fake_chassis):
        app_stub = graphiti_main.FastAPI()
        async with graphiti_main.lifespan(app_stub):
            assert graphiti_main.heartbeat_chassis is None
