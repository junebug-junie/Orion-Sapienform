"""Regression coverage for bus-observer's bus-native SystemHealthV1 heartbeat wiring.

Part of the wider service-heartbeat rollout following pilot-5 (PR #1350), see
docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md. bus-observer
(services/orion-bus/app/bus_observer.py) had no bus-native heartbeat before this patch
(confirmed via `grep -rl "BaseChassis"` returning nothing for services/orion-bus). This test
does not open a real bus connection -- it verifies (a) `build_heartbeat_chassis()` wires
ChassisConfig from the observer's own Settings, and (b) `run_bus_observer_loop()` actually
starts and stops that chassis around its tick loop, using a mocked chassis so the rest of the
loop stays a single-iteration no-op for this test.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.bus_observer import build_heartbeat_chassis, run_bus_observer_loop
from app.settings import Settings
from orion.core.bus.bus_service_chassis import HeartbeatOnly


def test_build_heartbeat_chassis_uses_observer_settings() -> None:
    settings = Settings().model_copy(
        update={
            "SERVICE_NAME": "orion-bus",
            "SERVICE_VERSION": "0.1.0",
            "NODE_NAME": "athena",
            "REDIS_URL": "redis://bus-core:6379/0",
            "HEARTBEAT_INTERVAL_SEC": 10.0,
        }
    )
    chassis = build_heartbeat_chassis(settings)
    assert isinstance(chassis, HeartbeatOnly)
    assert chassis.cfg.service_name == "orion-bus"
    assert chassis.cfg.service_version == "0.1.0"
    assert chassis.cfg.node_name == "athena"
    assert chassis.cfg.bus_url == "redis://bus-core:6379/0"
    assert chassis.cfg.bus_enabled is True
    assert chassis.cfg.heartbeat_interval_sec == 10.0
    assert chassis.cfg.health_channel == "orion:system:health"


@pytest.mark.asyncio
async def test_run_bus_observer_loop_starts_and_stops_heartbeat_chassis() -> None:
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    fake_bus = AsyncMock()

    async def _tick_then_stop(*args, **kwargs):
        raise StopAsyncLoop

    class StopAsyncLoop(Exception):
        pass

    with patch("app.bus_observer.OrionBusAsync", return_value=fake_bus), patch(
        "app.bus_observer.build_heartbeat_chassis", return_value=fake_chassis
    ), patch("app.bus_observer.run_observer_tick", side_effect=_tick_then_stop):
        with pytest.raises(StopAsyncLoop):
            await run_bus_observer_loop()

    fake_chassis.start_background.assert_awaited_once()
    fake_chassis.stop.assert_awaited_once()
    fake_bus.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_bus_observer_loop_survives_heartbeat_start_failure() -> None:
    """Heartbeat startup failure must not prevent the real observer tick loop from starting."""
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    fake_chassis.start_background.side_effect = RuntimeError("bus unreachable")
    fake_bus = AsyncMock()

    class StopAsyncLoop(Exception):
        pass

    async def _tick_then_stop(*args, **kwargs):
        raise StopAsyncLoop

    with patch("app.bus_observer.OrionBusAsync", return_value=fake_bus), patch(
        "app.bus_observer.build_heartbeat_chassis", return_value=fake_chassis
    ), patch("app.bus_observer.run_observer_tick", side_effect=_tick_then_stop):
        with pytest.raises(StopAsyncLoop):
            await run_bus_observer_loop()

    fake_chassis.stop.assert_not_awaited()
    fake_bus.close.assert_awaited_once()
