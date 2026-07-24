"""Regression coverage for power-guard's bus-native SystemHealthV1 heartbeat wiring.

Part of the service-heartbeat rollout, see
docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md.
orion-power-guard already publishes real bus events (on_battery/grace_elapsed/restored) via
`monitor_ups()`'s own `bus` connection, but had no `SystemHealthV1` liveness heartbeat before
this patch (confirmed via `grep -rl "BaseChassis"` returning nothing for this service). This is
also the first test file this service has -- there was no `tests/` directory before this patch.

Power Guard has no FastAPI app/lifespan (`app/main.py` is a bare `asyncio.run()` entrypoint), so
the heartbeat chassis is started/stopped around `monitor_ups()` in `_main_async()` instead of a
lifespan hook. This test does not open a real bus connection -- it verifies (a)
`build_heartbeat_chassis()` wires ChassisConfig from power-guard's own settings, and (b)
`_main_async()` actually starts and stops that chassis around the (mocked) `monitor_ups()` call.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

import app.main as power_guard_main
from app.settings import get_settings
from orion.core.bus.bus_service_chassis import HeartbeatOnly


def test_build_heartbeat_chassis_uses_power_guard_settings() -> None:
    settings = get_settings()
    chassis = power_guard_main.build_heartbeat_chassis(settings)
    assert isinstance(chassis, HeartbeatOnly)
    assert chassis.cfg.service_name == settings.SERVICE_NAME
    assert chassis.cfg.service_version == settings.SERVICE_VERSION
    assert chassis.cfg.node_name == settings.POWER_GUARD_NODE_NAME
    assert chassis.cfg.bus_url == settings.ORION_BUS_URL
    assert chassis.cfg.bus_enabled == settings.ORION_BUS_ENABLED
    assert chassis.cfg.heartbeat_interval_sec == settings.HEARTBEAT_INTERVAL_SEC
    assert chassis.cfg.health_channel == "orion:system:health"


@pytest.mark.asyncio
async def test_main_async_starts_and_stops_heartbeat_chassis_around_monitor_ups(
    monkeypatch,
) -> None:
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    monkeypatch.setattr(power_guard_main, "build_heartbeat_chassis", lambda settings: fake_chassis)
    monitor_ups_mock = AsyncMock()
    monkeypatch.setattr(power_guard_main, "monitor_ups", monitor_ups_mock)

    await power_guard_main._main_async()

    fake_chassis.start_background.assert_awaited_once()
    monitor_ups_mock.assert_awaited_once()
    fake_chassis.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_main_async_survives_heartbeat_start_failure(monkeypatch) -> None:
    """Heartbeat startup failure must not prevent the real UPS monitor loop from running."""
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    fake_chassis.start_background.side_effect = RuntimeError("bus unreachable")
    monkeypatch.setattr(power_guard_main, "build_heartbeat_chassis", lambda settings: fake_chassis)
    monitor_ups_mock = AsyncMock()
    monkeypatch.setattr(power_guard_main, "monitor_ups", monitor_ups_mock)

    await power_guard_main._main_async()

    monitor_ups_mock.assert_awaited_once()
    # stop() must not be called on a chassis whose start_background() raised -- there is
    # nothing running to stop.
    fake_chassis.stop.assert_not_awaited()
