"""Regression coverage for topic-foundry's bus-native SystemHealthV1 heartbeat wiring.

Part of the service-heartbeat rollout, see
docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md, following the
pilot-5 pattern proved out in PR #1350. orion-topic-foundry had no bus-native heartbeat before
this patch (confirmed via `grep -rl "BaseChassis"` returning nothing for this service).
`ensure_tables`/`drift_daemon_loop` are mocked out here so this test exercises only the new
heartbeat wiring in app/main.py's lifespan, without touching Postgres or opening a real bus
connection.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI

import app.main as topic_foundry_main
from orion.core.bus.bus_service_chassis import HeartbeatOnly


def test_build_heartbeat_chassis_uses_topic_foundry_settings() -> None:
    chassis = topic_foundry_main.build_heartbeat_chassis()
    assert isinstance(chassis, HeartbeatOnly)
    assert chassis.cfg.service_name == topic_foundry_main.settings.service_name
    assert chassis.cfg.service_version == topic_foundry_main.settings.service_version
    assert chassis.cfg.node_name == topic_foundry_main.settings.node_name
    assert chassis.cfg.bus_url == topic_foundry_main.settings.orion_bus_url
    assert chassis.cfg.bus_enabled == topic_foundry_main.settings.orion_bus_enabled
    assert chassis.cfg.heartbeat_interval_sec == topic_foundry_main.settings.heartbeat_interval_sec
    assert chassis.cfg.health_channel == "orion:system:health"


@pytest.mark.asyncio
async def test_lifespan_starts_and_stops_heartbeat_chassis(monkeypatch) -> None:
    monkeypatch.setattr(topic_foundry_main, "ensure_tables", lambda: None)
    monkeypatch.setattr(topic_foundry_main.settings, "topic_foundry_drift_daemon", False)
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    monkeypatch.setattr(topic_foundry_main, "build_heartbeat_chassis", lambda: fake_chassis)

    app = FastAPI()
    async with topic_foundry_main.lifespan(app):
        assert topic_foundry_main.heartbeat_chassis is fake_chassis
        fake_chassis.start_background.assert_awaited_once()

    fake_chassis.stop.assert_awaited_once()
    assert topic_foundry_main.heartbeat_chassis is None


@pytest.mark.asyncio
async def test_lifespan_survives_heartbeat_start_failure(monkeypatch) -> None:
    """Heartbeat startup failure must not prevent the rest of topic-foundry's boot."""
    monkeypatch.setattr(topic_foundry_main, "ensure_tables", lambda: None)
    monkeypatch.setattr(topic_foundry_main.settings, "topic_foundry_drift_daemon", False)
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    fake_chassis.start_background.side_effect = RuntimeError("bus unreachable")
    monkeypatch.setattr(topic_foundry_main, "build_heartbeat_chassis", lambda: fake_chassis)

    app = FastAPI()
    async with topic_foundry_main.lifespan(app):
        assert topic_foundry_main.heartbeat_chassis is None
