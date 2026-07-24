"""Regression coverage for self-experiments' bus-native SystemHealthV1 heartbeat wiring.

Part of the service-heartbeat rollout, see
docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md.
orion-self-experiments had no bus-native heartbeat before this patch (confirmed via
`grep -rl "BaseChassis"` returning nothing for this service) -- its existing bus usage
(dispatch to context-exec via `SELF_EXPERIMENTS_CONTEXT_EXEC_DISPATCH_TRANSPORT=bus`) is
per-request, not a persistent connection.

`init_db()` is mocked out here so this test exercises only the new heartbeat wiring in
app/main.py's lifespan, without touching the sqlite store.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI

import app.main as self_experiments_main
from orion.core.bus.bus_service_chassis import HeartbeatOnly


def test_build_heartbeat_chassis_uses_self_experiments_settings() -> None:
    chassis = self_experiments_main.build_heartbeat_chassis()
    assert isinstance(chassis, HeartbeatOnly)
    assert chassis.cfg.service_name == self_experiments_main.settings.service_name
    assert chassis.cfg.service_version == self_experiments_main.settings.service_version
    assert chassis.cfg.node_name == self_experiments_main.settings.node_name
    assert chassis.cfg.bus_url == self_experiments_main.settings.orion_bus_url
    assert chassis.cfg.bus_enabled == self_experiments_main.settings.orion_bus_enabled
    assert (
        chassis.cfg.heartbeat_interval_sec == self_experiments_main.settings.heartbeat_interval_sec
    )
    assert chassis.cfg.health_channel == "orion:system:health"


@pytest.mark.asyncio
async def test_lifespan_starts_and_stops_heartbeat_chassis(monkeypatch) -> None:
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    monkeypatch.setattr(self_experiments_main, "build_heartbeat_chassis", lambda: fake_chassis)
    monkeypatch.setattr(self_experiments_main, "init_db", lambda: None)

    app = FastAPI()
    async with self_experiments_main.lifespan(app):
        assert self_experiments_main.heartbeat_chassis is fake_chassis
        fake_chassis.start_background.assert_awaited_once()

    fake_chassis.stop.assert_awaited_once()
    assert self_experiments_main.heartbeat_chassis is None


@pytest.mark.asyncio
async def test_lifespan_survives_heartbeat_start_failure(monkeypatch) -> None:
    """Heartbeat startup failure must not prevent the real experiment registry from booting."""
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    fake_chassis.start_background.side_effect = RuntimeError("bus unreachable")
    monkeypatch.setattr(self_experiments_main, "build_heartbeat_chassis", lambda: fake_chassis)
    monkeypatch.setattr(self_experiments_main, "init_db", lambda: None)

    app = FastAPI()
    async with self_experiments_main.lifespan(app):
        assert self_experiments_main.heartbeat_chassis is None
