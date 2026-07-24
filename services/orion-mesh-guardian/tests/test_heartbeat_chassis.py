"""Regression coverage for mesh-guardian's bus-native SystemHealthV1 heartbeat wiring.

Part of the wider service-heartbeat rollout following pilot-5 (PR #1350), see
docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md.
orion-mesh-guardian had no bus-native heartbeat before this patch (confirmed via `grep -rl
"BaseChassis"` returning nothing for this service) -- it does have a separate bus connection of
its own (`guardian.bus`, watching orion:equilibrium:snapshot), unaffected by this patch. This
patch also fixes a pre-existing bug: `orion_bus_url`'s settings.py default was
`redis://bus-core:6379/0`, not the real tailscale node IP (the checked-in `.env_example` was
already correct) -- corrected to `100.92.216.81`, matching the fix PR #1350 made for
orion-harness-governor's same class of bug. `guardian` (the real MeshGuardianService) is mocked
out here so this test exercises only the new heartbeat wiring in app/main.py's lifespan,
without opening a real bus connection.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI

import app.main as guardian_main
from orion.core.bus.bus_service_chassis import HeartbeatOnly


def test_build_heartbeat_chassis_uses_guardian_settings() -> None:
    chassis = guardian_main.build_heartbeat_chassis()
    assert isinstance(chassis, HeartbeatOnly)
    assert chassis.cfg.service_name == guardian_main.settings.service_name
    assert chassis.cfg.service_version == guardian_main.settings.service_version
    assert chassis.cfg.node_name == guardian_main.settings.node_name
    assert chassis.cfg.bus_url == guardian_main.settings.orion_bus_url
    assert chassis.cfg.bus_enabled is True
    assert chassis.cfg.heartbeat_interval_sec == guardian_main.settings.heartbeat_interval_sec
    assert chassis.cfg.health_channel == "orion:system:health"


def test_bus_url_default_is_real_tailscale_node() -> None:
    """Regression guard for the bus-core placeholder bug fixed in this patch."""
    assert "bus-core" not in guardian_main.settings.orion_bus_url


@pytest.mark.asyncio
async def test_lifespan_starts_and_stops_heartbeat_chassis(monkeypatch) -> None:
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    monkeypatch.setattr(guardian_main, "build_heartbeat_chassis", lambda: fake_chassis)
    monkeypatch.setattr(guardian_main.guardian, "start", AsyncMock())
    monkeypatch.setattr(guardian_main.guardian, "stop", AsyncMock())

    app = FastAPI()
    async with guardian_main.lifespan(app):
        assert guardian_main.heartbeat_chassis is fake_chassis
        fake_chassis.start_background.assert_awaited_once()

    fake_chassis.stop.assert_awaited_once()
    assert guardian_main.heartbeat_chassis is None
    guardian_main.guardian.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_lifespan_survives_heartbeat_start_failure(monkeypatch) -> None:
    """Heartbeat startup failure must not prevent the real guardian from starting."""
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    fake_chassis.start_background.side_effect = RuntimeError("bus unreachable")
    monkeypatch.setattr(guardian_main, "build_heartbeat_chassis", lambda: fake_chassis)
    monkeypatch.setattr(guardian_main.guardian, "start", AsyncMock())
    monkeypatch.setattr(guardian_main.guardian, "stop", AsyncMock())

    app = FastAPI()
    async with guardian_main.lifespan(app):
        assert guardian_main.heartbeat_chassis is None
        guardian_main.guardian.start.assert_awaited_once()
