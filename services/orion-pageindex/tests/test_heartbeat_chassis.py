"""Regression coverage for PageIndex's bus-native SystemHealthV1 heartbeat wiring.

Part of the service-heartbeat rollout, see
docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md.
orion-pageindex had no bus connection of any kind before this patch -- it talks to Postgres
directly and shells out to the PageIndex CLI (confirmed via `grep -rl "BaseChassis"` and a grep
for `ORION_BUS` returning nothing for this service prior to this patch). The Dockerfile also
never copied the shared `orion` package in (fixed in the same patch) -- this test only exercises
the import/wiring path that runs from a repo-root checkout, it does not cover the Docker image
build itself. This test does not open a real bus connection -- it verifies (a)
`build_heartbeat_chassis()` wires ChassisConfig from the service's own settings, and (b) the new
startup/shutdown handlers actually start and stop that chassis.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock

import pytest

# app.main constructs a module-level JournalPageIndexService() at import time, which
# mkdir()s PAGEINDEX_DATA_DIR (default /data/pageindex -- only writable inside the real
# container). No other pageindex test module imports app.main today (test_service.py imports
# app.service/app.models directly), so this fallback is needed here regardless of collection
# order -- same pattern as test_worker.py's POSTGRES_URI fallback in other services.
os.environ.setdefault("PAGEINDEX_DATA_DIR", "/tmp/orion-pageindex-test-data")
os.environ.setdefault(
    "CHAT_EPISODES_MARKDOWN_PATH",
    "/tmp/orion-pageindex-test-data/chat_episodes/chat_episode_corpus.md",
)

import app.main as pageindex_main
from orion.core.bus.bus_service_chassis import HeartbeatOnly


def test_build_heartbeat_chassis_uses_pageindex_settings() -> None:
    chassis = pageindex_main.build_heartbeat_chassis()
    assert isinstance(chassis, HeartbeatOnly)
    assert chassis.cfg.service_name == pageindex_main.settings.SERVICE_NAME
    assert chassis.cfg.service_version == pageindex_main.settings.SERVICE_VERSION
    assert chassis.cfg.node_name == pageindex_main.settings.NODE_NAME
    assert chassis.cfg.bus_url == pageindex_main.settings.ORION_BUS_URL
    assert chassis.cfg.bus_enabled == pageindex_main.settings.ORION_BUS_ENABLED
    assert chassis.cfg.heartbeat_interval_sec == pageindex_main.settings.HEARTBEAT_INTERVAL_SEC
    assert chassis.cfg.health_channel == "orion:system:health"


@pytest.mark.asyncio
async def test_startup_starts_and_shutdown_stops_heartbeat_chassis(monkeypatch) -> None:
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    monkeypatch.setattr(pageindex_main, "build_heartbeat_chassis", lambda: fake_chassis)

    await pageindex_main._start_heartbeat_chassis()
    try:
        assert pageindex_main.heartbeat_chassis is fake_chassis
        fake_chassis.start_background.assert_awaited_once()
    finally:
        await pageindex_main._stop_heartbeat_chassis()

    fake_chassis.stop.assert_awaited_once()
    assert pageindex_main.heartbeat_chassis is None


@pytest.mark.asyncio
async def test_startup_survives_heartbeat_start_failure(monkeypatch) -> None:
    """Heartbeat startup failure must not prevent the rest of PageIndex's boot."""
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    fake_chassis.start_background.side_effect = RuntimeError("bus unreachable")
    monkeypatch.setattr(pageindex_main, "build_heartbeat_chassis", lambda: fake_chassis)

    await pageindex_main._start_heartbeat_chassis()
    try:
        assert pageindex_main.heartbeat_chassis is None
    finally:
        await pageindex_main._stop_heartbeat_chassis()
