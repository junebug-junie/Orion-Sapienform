"""Regression coverage for Hub's bus-native SystemHealthV1 heartbeat wiring.

Part of the pilot-5 rollout, see
docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md. Hub had no
bus-native heartbeat before this patch (confirmed via `grep -rl "BaseChassis"` returning
nothing for services/orion-hub). This test does not open a real bus connection -- it verifies
(a) `build_heartbeat_chassis()` wires ChassisConfig from Hub's own settings, and (b) Hub's
FastAPI startup/shutdown events actually start and stop that chassis, using a mocked chassis so
the rest of Hub's (large) startup path stays a no-op for this test.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock

import pytest

# Settings has a handful of required (no-default) fields unrelated to this test's scope --
# same fallback pattern as test_agent_repl_bridge.py, needed here because this file can be
# collected before any other hub test module has set these as a process-wide os.environ default.
for _key, _value in {
    "CHANNEL_VOICE_TRANSCRIPT": "orion:voice:transcript",
    "CHANNEL_VOICE_LLM": "orion:voice:llm",
    "CHANNEL_VOICE_TTS": "orion:voice:tts",
    "CHANNEL_COLLAPSE_INTAKE": "orion:collapse:intake",
    "CHANNEL_COLLAPSE_TRIAGE": "orion:collapse:triage",
}.items():
    os.environ.setdefault(_key, _value)

import scripts.main as hub_main
from orion.core.bus.bus_service_chassis import HeartbeatOnly


def test_build_heartbeat_chassis_uses_hub_settings() -> None:
    chassis = hub_main.build_heartbeat_chassis()
    assert isinstance(chassis, HeartbeatOnly)
    assert chassis.cfg.service_name == hub_main.settings.SERVICE_NAME
    assert chassis.cfg.service_version == hub_main.settings.SERVICE_VERSION
    assert chassis.cfg.node_name == hub_main.settings.NODE_NAME
    assert chassis.cfg.bus_url == hub_main.settings.ORION_BUS_URL
    assert chassis.cfg.bus_enabled == hub_main.settings.ORION_BUS_ENABLED
    assert chassis.cfg.heartbeat_interval_sec == hub_main.settings.HEARTBEAT_INTERVAL_SEC
    assert chassis.cfg.health_channel == "orion:system:health"


@pytest.mark.asyncio
async def test_startup_event_starts_and_shutdown_stops_heartbeat_chassis(monkeypatch) -> None:
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    monkeypatch.setattr(hub_main, "build_heartbeat_chassis", lambda: fake_chassis)

    # Keep the rest of Hub's (large) startup path a no-op so this test stays scoped to the
    # heartbeat wiring -- these flags gate the other startup_event side effects.
    monkeypatch.setattr(hub_main.settings, "ORION_BUS_ENABLED", False)
    monkeypatch.setattr(hub_main.settings, "SUBSTRATE_CONCEPT_SEED_ENABLED", False)
    monkeypatch.setattr(hub_main.settings, "SUBSTRATE_AUTONOMY_ENABLED", False)
    monkeypatch.setattr(hub_main.settings, "SUBSTRATE_DECAY_SCHEDULER_ENABLED", False)
    monkeypatch.setattr(hub_main.settings, "SUBSTRATE_TOPIC_FOUNDRY_SCHEDULER_ENABLED", False)
    monkeypatch.setattr(hub_main.settings, "RECALL_PG_DSN", "")

    await hub_main.startup_event()
    try:
        assert hub_main.heartbeat_chassis is fake_chassis
        fake_chassis.start_background.assert_awaited_once()
    finally:
        await hub_main.shutdown_event()

    assert hub_main.heartbeat_chassis is None
    fake_chassis.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_startup_event_survives_heartbeat_start_failure(monkeypatch) -> None:
    """Heartbeat startup failure must not take down the rest of Hub's boot."""
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    fake_chassis.start_background.side_effect = RuntimeError("bus unreachable")
    monkeypatch.setattr(hub_main, "build_heartbeat_chassis", lambda: fake_chassis)

    monkeypatch.setattr(hub_main.settings, "ORION_BUS_ENABLED", False)
    monkeypatch.setattr(hub_main.settings, "SUBSTRATE_CONCEPT_SEED_ENABLED", False)
    monkeypatch.setattr(hub_main.settings, "SUBSTRATE_AUTONOMY_ENABLED", False)
    monkeypatch.setattr(hub_main.settings, "SUBSTRATE_DECAY_SCHEDULER_ENABLED", False)
    monkeypatch.setattr(hub_main.settings, "SUBSTRATE_TOPIC_FOUNDRY_SCHEDULER_ENABLED", False)
    monkeypatch.setattr(hub_main.settings, "RECALL_PG_DSN", "")

    await hub_main.startup_event()
    try:
        assert hub_main.heartbeat_chassis is None
    finally:
        await hub_main.shutdown_event()
