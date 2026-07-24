"""Regression coverage for notify-digest's bus-native SystemHealthV1 heartbeat wiring.

Part of the service-heartbeat rollout, see
docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md.
orion-notify-digest had no bus connection of any kind before this patch -- it talks to
orion-notify over HTTP and to Postgres directly (confirmed via `grep -rl "BaseChassis"` and a
grep for `ORION_BUS` returning nothing for this service prior to this patch). This test does not
open a real bus connection or touch Postgres -- it verifies (a) `build_heartbeat_chassis()` wires
ChassisConfig from the service's own settings, and (b) `on_startup`/`on_shutdown` actually start
and stop that chassis.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

import app.main as digest_main
from orion.core.bus.bus_service_chassis import HeartbeatOnly


def test_build_heartbeat_chassis_uses_digest_settings() -> None:
    chassis = digest_main.build_heartbeat_chassis()
    assert isinstance(chassis, HeartbeatOnly)
    assert chassis.cfg.service_name == digest_main.settings.SERVICE_NAME
    assert chassis.cfg.service_version == digest_main.settings.SERVICE_VERSION
    assert chassis.cfg.node_name == digest_main.settings.NODE_NAME
    assert chassis.cfg.bus_url == digest_main.settings.ORION_BUS_URL
    assert chassis.cfg.bus_enabled == digest_main.settings.ORION_BUS_ENABLED
    assert chassis.cfg.heartbeat_interval_sec == digest_main.settings.HEARTBEAT_INTERVAL_SEC
    assert chassis.cfg.health_channel == "orion:system:health"


@pytest.mark.asyncio
async def test_startup_starts_and_shutdown_stops_heartbeat_chassis(monkeypatch) -> None:
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    monkeypatch.setattr(digest_main, "build_heartbeat_chassis", lambda: fake_chassis)
    monkeypatch.setattr(digest_main, "init_models", lambda *_a, **_k: None)
    monkeypatch.setattr(digest_main.settings, "DIGEST_ENABLED", False)
    monkeypatch.setattr(digest_main.settings, "DRIFT_ALERTS_ENABLED", False)

    await digest_main.on_startup()
    try:
        assert digest_main.app.state.heartbeat_chassis is fake_chassis
        fake_chassis.start_background.assert_awaited_once()
    finally:
        await digest_main.on_shutdown()

    fake_chassis.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_startup_survives_heartbeat_start_failure(monkeypatch) -> None:
    """Heartbeat startup failure must not prevent the rest of notify-digest's boot."""
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    fake_chassis.start_background.side_effect = RuntimeError("bus unreachable")
    monkeypatch.setattr(digest_main, "build_heartbeat_chassis", lambda: fake_chassis)
    monkeypatch.setattr(digest_main, "init_models", lambda *_a, **_k: None)
    monkeypatch.setattr(digest_main.settings, "DIGEST_ENABLED", False)
    monkeypatch.setattr(digest_main.settings, "DRIFT_ALERTS_ENABLED", False)

    await digest_main.on_startup()
    try:
        assert digest_main.app.state.heartbeat_chassis is None
    finally:
        await digest_main.on_shutdown()
