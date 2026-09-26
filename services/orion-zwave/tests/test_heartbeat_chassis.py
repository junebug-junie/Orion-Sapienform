"""Regression coverage for orion-zwave's bus-native SystemHealthV1 heartbeat wiring."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

import app.main as zwave_main
from app.settings import get_settings
from orion.core.bus.bus_service_chassis import HeartbeatOnly


def test_build_heartbeat_chassis_uses_zwave_settings() -> None:
    settings = get_settings()
    chassis = zwave_main.build_heartbeat_chassis(settings)
    assert isinstance(chassis, HeartbeatOnly)
    assert chassis.cfg.service_name == settings.SERVICE_NAME
    assert chassis.cfg.service_name == "orion-zwave"
    assert chassis.cfg.service_version == settings.SERVICE_VERSION
    assert chassis.cfg.node_name == settings.INSTANCE_ID
    assert chassis.cfg.bus_url == settings.ORION_BUS_URL
    assert chassis.cfg.bus_enabled == settings.ORION_BUS_ENABLED
    assert chassis.cfg.heartbeat_interval_sec == settings.HEARTBEAT_INTERVAL_SEC
    assert chassis.cfg.health_channel == "orion:system:health"


@pytest.mark.asyncio
async def test_main_async_starts_and_stops_heartbeat_chassis_around_poll_loop(
    monkeypatch,
) -> None:
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    monkeypatch.setattr(zwave_main, "build_heartbeat_chassis", lambda settings: fake_chassis)
    poll_loop_mock = AsyncMock()
    monkeypatch.setattr(zwave_main, "poll_cooling_loop", poll_loop_mock)

    await zwave_main._main_async()

    fake_chassis.start_background.assert_awaited_once()
    poll_loop_mock.assert_awaited_once()
    fake_chassis.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_main_async_survives_heartbeat_start_failure(monkeypatch) -> None:
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    fake_chassis.start_background.side_effect = RuntimeError("bus unreachable")
    monkeypatch.setattr(zwave_main, "build_heartbeat_chassis", lambda settings: fake_chassis)
    poll_loop_mock = AsyncMock()
    monkeypatch.setattr(zwave_main, "poll_cooling_loop", poll_loop_mock)

    await zwave_main._main_async()

    poll_loop_mock.assert_awaited_once()
    fake_chassis.stop.assert_not_awaited()
