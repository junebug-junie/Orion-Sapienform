"""Regression coverage for spark-concept-induction's bus-native SystemHealthV1 heartbeat wiring.

Part of the service-heartbeat rollout, see
docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md, following the
pilot-5 pattern proved out in PR #1350. orion-spark-concept-induction had no bus-native
heartbeat before this patch (confirmed via `grep -rl "BaseChassis"` returning nothing for this
service) -- even though `ConceptSettings.heartbeat_interval_sec` already existed as an unused
field (a keyword-cathedral half-step: a config knob with no producer). This patch is that
producer. `ConceptWorker` (the real worker, owning its own bus connection) is mocked out here
so this test exercises only the new heartbeat wiring in app/main.py's lifespan, without opening
a real bus connection.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI

ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = ROOT / "services" / "orion-spark-concept-induction"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app import main as concept_main  # noqa: E402
from orion.core.bus.bus_service_chassis import HeartbeatOnly  # noqa: E402


class _FakeWorker:
    def __init__(self, _settings):
        self.started = asyncio.Event()
        self.stop_calls = 0

    async def start(self) -> None:
        self.started.set()
        try:
            while True:
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            raise

    async def stop(self) -> None:
        self.stop_calls += 1

    def mark_task_created(self) -> None:
        return None

    def record_task_exit(self, _task) -> None:
        return None


def test_build_heartbeat_chassis_uses_concept_settings() -> None:
    chassis = concept_main.build_heartbeat_chassis()
    assert isinstance(chassis, HeartbeatOnly)
    assert chassis.cfg.service_name == concept_main.settings.service_name
    assert chassis.cfg.service_version == concept_main.settings.service_version
    assert chassis.cfg.node_name == concept_main.settings.node_name
    assert chassis.cfg.bus_url == concept_main.settings.orion_bus_url
    assert chassis.cfg.bus_enabled == concept_main.settings.orion_bus_enabled
    assert chassis.cfg.heartbeat_interval_sec == concept_main.settings.heartbeat_interval_sec
    assert chassis.cfg.health_channel == "orion:system:health"


@pytest.mark.asyncio
async def test_lifespan_starts_and_stops_heartbeat_chassis(monkeypatch) -> None:
    monkeypatch.setattr(concept_main, "ConceptWorker", _FakeWorker)
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    monkeypatch.setattr(concept_main, "build_heartbeat_chassis", lambda: fake_chassis)

    app = FastAPI()
    async with concept_main.lifespan(app):
        assert concept_main.heartbeat_chassis is fake_chassis
        fake_chassis.start_background.assert_awaited_once()

    fake_chassis.stop.assert_awaited_once()
    assert concept_main.heartbeat_chassis is None


@pytest.mark.asyncio
async def test_lifespan_survives_heartbeat_start_failure(monkeypatch) -> None:
    """Heartbeat startup failure must not prevent the real concept-induction worker from starting."""
    monkeypatch.setattr(concept_main, "ConceptWorker", _FakeWorker)
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    fake_chassis.start_background.side_effect = RuntimeError("bus unreachable")
    monkeypatch.setattr(concept_main, "build_heartbeat_chassis", lambda: fake_chassis)

    app = FastAPI()
    async with concept_main.lifespan(app):
        assert concept_main.heartbeat_chassis is None
        assert app.state.worker.started.is_set() is True
