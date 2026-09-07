"""Regression coverage for harness-governor's bus-native SystemHealthV1 heartbeat wiring.

Part of the pilot-5 rollout, see
docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md.
orion-harness-governor had no bus-native heartbeat before this patch (confirmed via
`grep -rl "BaseChassis"` returning nothing for this service). The service's real bus workers
(run_bus_worker/run_cancel_worker) are mocked out here so this test exercises only the new
heartbeat wiring in app/main.py's lifespan, without opening a real bus connection.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI

import app.main as governor_main
from orion.core.bus.bus_service_chassis import HeartbeatOnly


def test_build_heartbeat_chassis_uses_governor_settings() -> None:
    chassis = governor_main.build_heartbeat_chassis()
    assert isinstance(chassis, HeartbeatOnly)
    assert chassis.cfg.service_name == governor_main.settings.service_name
    assert chassis.cfg.service_version == governor_main.settings.service_version
    assert chassis.cfg.node_name == governor_main.settings.node_name
    assert chassis.cfg.bus_url == governor_main.settings.orion_bus_url
    assert chassis.cfg.bus_enabled == governor_main.settings.orion_bus_enabled
    assert chassis.cfg.heartbeat_interval_sec == governor_main.settings.heartbeat_interval_sec
    assert chassis.cfg.health_channel == "orion:system:health"


async def _wait_for_stop(
    first: str | asyncio.Event, second: asyncio.Event | None = None, *, lane: str = "chat"
) -> None:
    """Stands in for both `run_bus_worker(channel, stop_event, lane=...)` and
    `run_cancel_worker(stop_event)` -- their calling shapes differ, so `stop_event`
    is whichever of the two positional args is actually the Event."""
    stop_event = second if second is not None else first
    await stop_event.wait()  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_lifespan_starts_and_stops_heartbeat_chassis(monkeypatch) -> None:
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    monkeypatch.setattr(governor_main, "build_heartbeat_chassis", lambda: fake_chassis)
    # Real bus workers are exercised by test_harness_governor_rpc.py -- here they are
    # replaced with a stop-event-only no-op so this test stays scoped to heartbeat wiring
    # and never opens a real bus connection.
    monkeypatch.setattr(governor_main, "run_bus_worker", _wait_for_stop)
    monkeypatch.setattr(governor_main, "run_cancel_worker", _wait_for_stop)

    app = FastAPI()
    async with governor_main.lifespan(app):
        assert app.state.heartbeat_chassis is fake_chassis
        fake_chassis.start_background.assert_awaited_once()

    fake_chassis.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_lifespan_survives_heartbeat_start_failure(monkeypatch) -> None:
    """Heartbeat startup failure must not prevent the governor's real bus workers from starting."""
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    fake_chassis.start_background.side_effect = RuntimeError("bus unreachable")
    monkeypatch.setattr(governor_main, "build_heartbeat_chassis", lambda: fake_chassis)
    monkeypatch.setattr(governor_main, "run_bus_worker", _wait_for_stop)
    monkeypatch.setattr(governor_main, "run_cancel_worker", _wait_for_stop)

    app = FastAPI()
    async with governor_main.lifespan(app):
        assert app.state.heartbeat_chassis is None
        assert not app.state.bus_task_chat.done()
        assert not app.state.bus_task_agent.done()
