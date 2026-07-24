"""Regression coverage for security-watcher's bus-native SystemHealthV1 heartbeat wiring.

Part of the service-heartbeat rollout, see
docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md.
orion-security-watcher had no bus-native heartbeat before this patch (confirmed via
`grep -rl "BaseChassis"` returning nothing for this service). This is also the first test file
this service has -- there was no `tests/` directory before this patch. This patch also adds the
service's first `@app.on_event("shutdown")` handler (there was none before, so nothing was
gracefully torn down on shutdown).

This test does not open a real bus connection -- it verifies (a) `build_heartbeat_chassis()`
wires ChassisConfig from `ctx.settings`, and (b) `on_startup`/`on_shutdown` actually start and
stop that chassis, using a mocked chassis so the real `bus_worker` task stays scoped out.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

import app.main as security_watcher_main
from app.context import ctx
from orion.core.bus.bus_service_chassis import HeartbeatOnly


def test_build_heartbeat_chassis_uses_security_watcher_settings() -> None:
    chassis = security_watcher_main.build_heartbeat_chassis()
    assert isinstance(chassis, HeartbeatOnly)
    assert chassis.cfg.service_name == ctx.settings.SERVICE_NAME
    assert chassis.cfg.service_version == ctx.settings.SERVICE_VERSION
    assert chassis.cfg.node_name == ctx.settings.NODE_NAME
    assert chassis.cfg.bus_url == ctx.settings.ORION_BUS_URL
    assert chassis.cfg.bus_enabled == ctx.settings.ORION_BUS_ENABLED
    assert chassis.cfg.heartbeat_interval_sec == ctx.settings.HEARTBEAT_INTERVAL_SEC
    assert chassis.cfg.health_channel == "orion:system:health"


@pytest.mark.asyncio
async def test_startup_starts_and_shutdown_stops_heartbeat_chassis(monkeypatch) -> None:
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    monkeypatch.setattr(security_watcher_main, "build_heartbeat_chassis", lambda: fake_chassis)
    monkeypatch.setattr(ctx.settings, "SECURITY_ENABLED", False)

    await security_watcher_main.on_startup()
    try:
        assert security_watcher_main.app.state.heartbeat_chassis is fake_chassis
        fake_chassis.start_background.assert_awaited_once()
    finally:
        await security_watcher_main.on_shutdown()

    fake_chassis.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_startup_survives_heartbeat_start_failure(monkeypatch) -> None:
    """Heartbeat startup failure must not prevent the rest of Security Watcher's boot."""
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    fake_chassis.start_background.side_effect = RuntimeError("bus unreachable")
    monkeypatch.setattr(security_watcher_main, "build_heartbeat_chassis", lambda: fake_chassis)
    monkeypatch.setattr(ctx.settings, "SECURITY_ENABLED", False)

    await security_watcher_main.on_startup()
    try:
        assert security_watcher_main.app.state.heartbeat_chassis is None
    finally:
        await security_watcher_main.on_shutdown()
