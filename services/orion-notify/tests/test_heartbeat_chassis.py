"""Regression coverage for orion-notify's bus-native SystemHealthV1 heartbeat wiring.

Part of the service-heartbeat rollout, see
docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md.
orion-notify had no bus-native heartbeat before this patch (confirmed via
`grep -rl "BaseChassis"` returning nothing for this service). This test does not open a real
bus connection -- it verifies (a) `build_heartbeat_chassis()` wires ChassisConfig from Notify's
own settings, and (b) `on_startup`/`on_shutdown` actually start and stop that chassis, using a
mocked chassis so the rest of startup (bus init, email transport, policy load) stays scoped out.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

import app.main as notify_main
from orion.core.bus.bus_service_chassis import HeartbeatOnly


def test_build_heartbeat_chassis_uses_notify_settings() -> None:
    chassis = notify_main.build_heartbeat_chassis()
    assert isinstance(chassis, HeartbeatOnly)
    assert chassis.cfg.service_name == notify_main.settings.SERVICE_NAME
    assert chassis.cfg.service_version == notify_main.settings.SERVICE_VERSION
    assert chassis.cfg.node_name == notify_main.settings.NODE_NAME
    assert chassis.cfg.bus_url == notify_main.settings.ORION_BUS_URL
    assert chassis.cfg.bus_enabled == notify_main.settings.ORION_BUS_ENABLED
    assert chassis.cfg.heartbeat_interval_sec == notify_main.settings.HEARTBEAT_INTERVAL_SEC
    assert chassis.cfg.health_channel == "orion:system:health"


@pytest.mark.asyncio
async def test_startup_starts_and_shutdown_stops_heartbeat_chassis(monkeypatch) -> None:
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    monkeypatch.setattr(notify_main, "build_heartbeat_chassis", lambda: fake_chassis)
    monkeypatch.setattr(notify_main, "_init_bus", AsyncMock(return_value=None))
    monkeypatch.setattr(notify_main, "_build_email_transport", lambda: None)
    monkeypatch.setattr(notify_main, "_load_policy", lambda: None)

    await notify_main.on_startup()
    try:
        assert notify_main.app.state.heartbeat_chassis is fake_chassis
        fake_chassis.start_background.assert_awaited_once()
    finally:
        await notify_main.on_shutdown()

    fake_chassis.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_startup_survives_heartbeat_start_failure(monkeypatch) -> None:
    """Heartbeat startup failure must not prevent the rest of Notify's boot."""
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    fake_chassis.start_background.side_effect = RuntimeError("bus unreachable")
    monkeypatch.setattr(notify_main, "build_heartbeat_chassis", lambda: fake_chassis)
    monkeypatch.setattr(notify_main, "_init_bus", AsyncMock(return_value=None))
    monkeypatch.setattr(notify_main, "_build_email_transport", lambda: None)
    monkeypatch.setattr(notify_main, "_load_policy", lambda: None)

    await notify_main.on_startup()
    try:
        assert notify_main.app.state.heartbeat_chassis is None
    finally:
        await notify_main.on_shutdown()
