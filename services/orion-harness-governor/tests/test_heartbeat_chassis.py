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


class _FakeTask:
    def __init__(self, done: bool) -> None:
        self._done = done

    def done(self) -> bool:
        return self._done


def test_lane_alive_is_none_when_disabled_not_false(monkeypatch) -> None:
    """`lane_chat_alive`/`lane_agent_alive` must distinguish "turned off on
    purpose" from "the dispatch loop died" -- both make the task `done()`,
    but only one is a problem. `None` (not `False`) is the "not applicable"
    signal a dashboard/alert has to check for separately from a real crash."""
    monkeypatch.setattr(governor_main.settings, "orion_bus_enabled", False)
    monkeypatch.setattr(governor_main.settings, "orion_harness_governor_enabled", True)
    assert governor_main._lane_alive(_FakeTask(done=True)) is None
    assert governor_main._lane_alive(None) is None


def test_lane_alive_is_a_real_bool_when_enabled(monkeypatch) -> None:
    monkeypatch.setattr(governor_main.settings, "orion_bus_enabled", True)
    monkeypatch.setattr(governor_main.settings, "orion_harness_governor_enabled", True)
    assert governor_main._lane_alive(_FakeTask(done=False)) is True
    assert governor_main._lane_alive(_FakeTask(done=True)) is False
    assert governor_main._lane_alive(None) is False


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


async def _wait_for_stop(*args: object, stop_event: asyncio.Event | None = None, **kwargs: object) -> None:
    """Stands in for both `run_bus_worker(stop_event=..., lane=..., bus=...)`
    (all-keyword, no positional args at all) and `run_cancel_worker(stop_event)`
    (one positional arg) -- their calling shapes differ, so `stop_event` is
    whichever of "the keyword" or "the first positional arg" is actually present."""
    event = stop_event if stop_event is not None else args[0]
    await event.wait()  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_lifespan_starts_and_stops_heartbeat_chassis(monkeypatch) -> None:
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    monkeypatch.setattr(governor_main, "build_heartbeat_chassis", lambda: fake_chassis)
    # Bus disabled so lifespan's own `dispatch_bus.connect()` (shared across both
    # lanes) no-ops instead of opening a real connection -- run_bus_worker/
    # run_cancel_worker are ALSO replaced below, so this test stays scoped to
    # heartbeat wiring and never touches a real bus at all.
    monkeypatch.setattr(governor_main.settings, "orion_bus_enabled", False)
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
    monkeypatch.setattr(governor_main.settings, "orion_bus_enabled", False)
    monkeypatch.setattr(governor_main, "run_bus_worker", _wait_for_stop)
    monkeypatch.setattr(governor_main, "run_cancel_worker", _wait_for_stop)

    app = FastAPI()
    async with governor_main.lifespan(app):
        assert app.state.heartbeat_chassis is None
        assert not app.state.bus_task_chat.done()
        assert not app.state.bus_task_agent.done()
