"""Regression coverage for feedback-runtime's bus-native SystemHealthV1 heartbeat wiring.

Part of the wider service-heartbeat rollout following pilot-5 (PR #1350), see
docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md.
orion-feedback-runtime had no bus-native heartbeat before this patch (confirmed via
`grep -rl "BaseChassis"` returning nothing for this service) -- it does have a separate bus
connection of its own (`FeedbackRuntimeWorker._bus`, publishing `FEEDBACK_BUS_CHANNEL`),
unaffected by this patch. This patch also fixes a pre-existing bug: `ORION_BUS_URL` in this
service's `.env_example`/`settings.py` default was `redis://bus-core:6379/0` /
`redis://127.0.0.1:6379/0` respectively, not the real tailscale node IP -- corrected to
`100.92.216.81`, matching the fix PR #1350 made for orion-harness-governor's same class of bug.
`worker` (the real FeedbackRuntimeWorker) is mocked out here so this test exercises only the
new heartbeat wiring in app/main.py's lifespan, without opening a real bus connection.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI

# app.main constructs a module-level FeedbackRuntimeStore(...) at import time, which requires
# POSTGRES_URI (no default).
os.environ.setdefault("POSTGRES_URI", "postgresql://unused/unused")

import app.main as feedback_main
from orion.core.bus.bus_service_chassis import HeartbeatOnly


def test_build_heartbeat_chassis_uses_feedback_settings() -> None:
    import app.settings as settings_mod

    chassis = feedback_main.build_heartbeat_chassis()
    s = settings_mod.get_settings()
    assert isinstance(chassis, HeartbeatOnly)
    assert chassis.cfg.service_name == s.service_name
    assert chassis.cfg.service_version == s.service_version
    assert chassis.cfg.node_name == s.node_name
    assert chassis.cfg.bus_url == s.bus_url
    assert chassis.cfg.bus_enabled == s.bus_enabled
    assert chassis.cfg.heartbeat_interval_sec == s.heartbeat_interval_sec
    assert chassis.cfg.health_channel == "orion:system:health"


def test_bus_url_default_is_real_tailscale_node() -> None:
    """Regression guard for the bus-core/127.0.0.1 placeholder bug fixed in this patch."""
    import app.settings as settings_mod

    s = settings_mod.get_settings()
    assert "bus-core" not in s.bus_url
    assert "127.0.0.1" not in s.bus_url


@pytest.mark.asyncio
async def test_lifespan_starts_and_stops_heartbeat_chassis(monkeypatch) -> None:
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    monkeypatch.setattr(feedback_main, "build_heartbeat_chassis", lambda: fake_chassis)
    monkeypatch.setattr(feedback_main.worker, "start", AsyncMock())
    monkeypatch.setattr(feedback_main.worker, "stop", AsyncMock())

    app = FastAPI()
    async with feedback_main.lifespan(app):
        assert feedback_main.heartbeat_chassis is fake_chassis
        fake_chassis.start_background.assert_awaited_once()

    fake_chassis.stop.assert_awaited_once()
    assert feedback_main.heartbeat_chassis is None
    feedback_main.worker.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_lifespan_survives_heartbeat_start_failure(monkeypatch) -> None:
    """Heartbeat startup failure must not prevent the real feedback worker from starting."""
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    fake_chassis.start_background.side_effect = RuntimeError("bus unreachable")
    monkeypatch.setattr(feedback_main, "build_heartbeat_chassis", lambda: fake_chassis)
    monkeypatch.setattr(feedback_main.worker, "start", AsyncMock())
    monkeypatch.setattr(feedback_main.worker, "stop", AsyncMock())

    app = FastAPI()
    async with feedback_main.lifespan(app):
        assert feedback_main.heartbeat_chassis is None
        feedback_main.worker.start.assert_awaited_once()
