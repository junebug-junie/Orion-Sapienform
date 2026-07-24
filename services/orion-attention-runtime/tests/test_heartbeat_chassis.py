"""Regression coverage for attention-runtime's bus-native SystemHealthV1 heartbeat wiring.

Part of the wider service-heartbeat rollout following pilot-5 (PR #1350), see
docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md.
orion-attention-runtime had no bus connection at all before this patch (confirmed via
`grep -rl "BaseChassis\\|SystemHealthV1"` returning nothing for this service) -- it is a
Postgres-poll-only worker. `worker` (the real AttentionRuntimeWorker) is mocked out here so
this test exercises only the new heartbeat wiring in app/main.py's lifespan, without opening a
real bus connection.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI

# app.main constructs a module-level AttentionRuntimeStore(...) at import time, which requires
# POSTGRES_URI (no default). Matches the fallback pattern used by other tests in this suite
# (test_health_monitor.py, test_worker_prune.py).
os.environ.setdefault("POSTGRES_URI", "postgresql://unused/unused")

import app.main as attention_main
from orion.core.bus.bus_service_chassis import HeartbeatOnly


def test_build_heartbeat_chassis_uses_attention_settings() -> None:
    import app.settings as settings_mod

    chassis = attention_main.build_heartbeat_chassis()
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
    monkeypatch.setattr(attention_main, "build_heartbeat_chassis", lambda: fake_chassis)
    monkeypatch.setattr(attention_main.worker, "start", AsyncMock())
    monkeypatch.setattr(attention_main.worker, "stop", AsyncMock())

    app = FastAPI()
    async with attention_main.lifespan(app):
        assert attention_main.heartbeat_chassis is fake_chassis
        fake_chassis.start_background.assert_awaited_once()

    fake_chassis.stop.assert_awaited_once()
    assert attention_main.heartbeat_chassis is None
    attention_main.worker.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_lifespan_survives_heartbeat_start_failure(monkeypatch) -> None:
    """Heartbeat startup failure must not prevent the real worker from starting."""
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    fake_chassis.start_background.side_effect = RuntimeError("bus unreachable")
    monkeypatch.setattr(attention_main, "build_heartbeat_chassis", lambda: fake_chassis)
    monkeypatch.setattr(attention_main.worker, "start", AsyncMock())
    monkeypatch.setattr(attention_main.worker, "stop", AsyncMock())

    app = FastAPI()
    async with attention_main.lifespan(app):
        assert attention_main.heartbeat_chassis is None
        attention_main.worker.start.assert_awaited_once()
