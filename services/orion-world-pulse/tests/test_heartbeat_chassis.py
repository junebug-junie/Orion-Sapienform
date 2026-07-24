"""Regression coverage for world-pulse's bus-native SystemHealthV1 heartbeat wiring.

Part of the service-heartbeat rollout, see
docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md, following the
pilot-5 pattern proved out in PR #1350. orion-world-pulse had no bus-native heartbeat before
this patch (confirmed via `grep -rl "BaseChassis"` returning nothing for this service) --
app/main.py previously had no `lifespan` at all (a bare `FastAPI(...)` with routers attached
and only a `uvicorn.run(...)` call under `if __name__ == "__main__"`). This patch adds the
lifespan whose sole job (for now) is the heartbeat.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI

import app.main as world_pulse_main
from orion.core.bus.bus_service_chassis import HeartbeatOnly


def test_build_heartbeat_chassis_uses_world_pulse_settings() -> None:
    chassis = world_pulse_main.build_heartbeat_chassis()
    assert isinstance(chassis, HeartbeatOnly)
    assert chassis.cfg.service_name == world_pulse_main.settings.service_name
    assert chassis.cfg.service_version == world_pulse_main.settings.service_version
    assert chassis.cfg.node_name == world_pulse_main.settings.node_name
    assert chassis.cfg.bus_url == world_pulse_main.settings.orion_bus_url
    assert chassis.cfg.bus_enabled == world_pulse_main.settings.orion_bus_enabled
    assert chassis.cfg.heartbeat_interval_sec == world_pulse_main.settings.heartbeat_interval_sec
    assert chassis.cfg.health_channel == "orion:system:health"


@pytest.mark.asyncio
async def test_lifespan_starts_and_stops_heartbeat_chassis(monkeypatch) -> None:
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    monkeypatch.setattr(world_pulse_main, "build_heartbeat_chassis", lambda: fake_chassis)

    app = FastAPI()
    async with world_pulse_main.lifespan(app):
        assert world_pulse_main.heartbeat_chassis is fake_chassis
        fake_chassis.start_background.assert_awaited_once()

    fake_chassis.stop.assert_awaited_once()
    assert world_pulse_main.heartbeat_chassis is None


@pytest.mark.asyncio
async def test_lifespan_survives_heartbeat_start_failure(monkeypatch) -> None:
    """Heartbeat startup failure must not raise out of lifespan and take down the app."""
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    fake_chassis.start_background.side_effect = RuntimeError("bus unreachable")
    monkeypatch.setattr(world_pulse_main, "build_heartbeat_chassis", lambda: fake_chassis)

    app = FastAPI()
    async with world_pulse_main.lifespan(app):
        assert world_pulse_main.heartbeat_chassis is None
