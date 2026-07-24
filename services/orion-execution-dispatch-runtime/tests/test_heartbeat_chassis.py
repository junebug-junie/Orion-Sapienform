"""Regression coverage for execution-dispatch-runtime's bus-native SystemHealthV1 heartbeat
wiring.

Part of the wider service-heartbeat rollout following pilot-5 (PR #1350), see
docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md.
orion-execution-dispatch-runtime had no bus-native heartbeat before this patch (confirmed via
`grep -rl "BaseChassis"` returning nothing for this service) -- it does have a separate RPC bus
client to cortex-exec (`orion.execution_dispatch.cortex_client`), unaffected by this patch.
`worker` (the real ExecutionDispatchRuntimeWorker) is mocked out here so this test exercises
only the new heartbeat wiring in app/main.py's lifespan, without opening a real bus connection.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI

# app.main constructs a module-level ExecutionDispatchRuntimeStore(...) at import time, which
# requires POSTGRES_URI (no default).
os.environ.setdefault("POSTGRES_URI", "postgresql://unused/unused")

import app.main as dispatch_main
from orion.core.bus.bus_service_chassis import HeartbeatOnly


def test_build_heartbeat_chassis_uses_dispatch_settings() -> None:
    import app.settings as settings_mod

    chassis = dispatch_main.build_heartbeat_chassis()
    s = settings_mod.get_settings()
    assert isinstance(chassis, HeartbeatOnly)
    assert chassis.cfg.service_name == s.service_name
    assert chassis.cfg.service_version == s.service_version
    assert chassis.cfg.node_name == s.node_name
    assert chassis.cfg.bus_url == s.orion_bus_url
    assert chassis.cfg.bus_enabled == s.orion_bus_enabled
    assert chassis.cfg.heartbeat_interval_sec == s.heartbeat_interval_sec
    assert chassis.cfg.health_channel == "orion:system:health"


@pytest.mark.asyncio
async def test_lifespan_starts_and_stops_heartbeat_chassis(monkeypatch) -> None:
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    monkeypatch.setattr(dispatch_main, "build_heartbeat_chassis", lambda: fake_chassis)
    monkeypatch.setattr(dispatch_main.worker, "start", AsyncMock())
    monkeypatch.setattr(dispatch_main.worker, "stop", AsyncMock())

    app = FastAPI()
    async with dispatch_main.lifespan(app):
        assert dispatch_main.heartbeat_chassis is fake_chassis
        fake_chassis.start_background.assert_awaited_once()

    fake_chassis.stop.assert_awaited_once()
    assert dispatch_main.heartbeat_chassis is None
    dispatch_main.worker.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_lifespan_survives_heartbeat_start_failure(monkeypatch) -> None:
    """Heartbeat startup failure must not prevent the real dispatch worker from starting."""
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    fake_chassis.start_background.side_effect = RuntimeError("bus unreachable")
    monkeypatch.setattr(dispatch_main, "build_heartbeat_chassis", lambda: fake_chassis)
    monkeypatch.setattr(dispatch_main.worker, "start", AsyncMock())
    monkeypatch.setattr(dispatch_main.worker, "stop", AsyncMock())

    app = FastAPI()
    async with dispatch_main.lifespan(app):
        assert dispatch_main.heartbeat_chassis is None
        dispatch_main.worker.start.assert_awaited_once()
