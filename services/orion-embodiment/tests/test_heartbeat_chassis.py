"""Regression coverage for embodiment's bus-native SystemHealthV1 heartbeat wiring.

Part of the wider service-heartbeat rollout following pilot-5 (PR #1350), see
docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md.
orion-embodiment had no bus-native heartbeat before this patch (confirmed via
`grep -rl "BaseChassis"` returning nothing for this service) -- it does have a separate bus
connection of its own (`EmbodimentWorker._bus`, handling intent/outcome/perception traffic),
unaffected by this patch. `worker` (the real EmbodimentWorker) is mocked out here so this test
exercises only the new heartbeat wiring in app/main.py's lifespan, without opening a real bus
connection.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI

# app.main constructs a module-level EmbodimentWorker() at import time, which reads
# EMBODIMENT_FCC_ENV_PATH (default /root/.fcc/.env) via Path.is_file() -- in a sandboxed test
# environment without root's home directory, that raises PermissionError on the traversal
# itself, not just a missing-file. Point at a path this process can actually stat.
os.environ.setdefault("EMBODIMENT_FCC_ENV_PATH", "/tmp/nonexistent-embodiment-fcc-test.env")

import app.main as embodiment_main
from orion.core.bus.bus_service_chassis import HeartbeatOnly


def test_build_heartbeat_chassis_uses_embodiment_settings() -> None:
    from app.settings import get_settings

    chassis = embodiment_main.build_heartbeat_chassis()
    s = get_settings()
    assert isinstance(chassis, HeartbeatOnly)
    assert chassis.cfg.service_name == s.service_name
    assert chassis.cfg.service_version == s.service_version
    assert chassis.cfg.node_name == s.node_name
    assert chassis.cfg.bus_url == s.bus_url
    assert chassis.cfg.bus_enabled == s.bus_enabled
    assert chassis.cfg.heartbeat_interval_sec == s.heartbeat_interval_sec
    assert chassis.cfg.health_channel == "orion:system:health"


@pytest.mark.asyncio
async def test_lifespan_starts_and_stops_heartbeat_chassis(monkeypatch) -> None:
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    monkeypatch.setattr(embodiment_main, "build_heartbeat_chassis", lambda: fake_chassis)
    monkeypatch.setattr(embodiment_main.worker, "start", AsyncMock())
    monkeypatch.setattr(embodiment_main.worker, "stop", AsyncMock())

    app = FastAPI()
    async with embodiment_main.lifespan(app):
        assert embodiment_main.heartbeat_chassis is fake_chassis
        fake_chassis.start_background.assert_awaited_once()

    fake_chassis.stop.assert_awaited_once()
    assert embodiment_main.heartbeat_chassis is None
    embodiment_main.worker.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_lifespan_survives_heartbeat_start_failure(monkeypatch) -> None:
    """Heartbeat startup failure must not prevent the real embodiment worker from starting,
    even with ORION_EMBODIMENT_ENABLED=false (worker.start() itself is a no-op in that case,
    but the heartbeat is independent of that flag and must still attempt to run)."""
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    fake_chassis.start_background.side_effect = RuntimeError("bus unreachable")
    monkeypatch.setattr(embodiment_main, "build_heartbeat_chassis", lambda: fake_chassis)
    monkeypatch.setattr(embodiment_main.worker, "start", AsyncMock())
    monkeypatch.setattr(embodiment_main.worker, "stop", AsyncMock())

    app = FastAPI()
    async with embodiment_main.lifespan(app):
        assert embodiment_main.heartbeat_chassis is None
        embodiment_main.worker.start.assert_awaited_once()
