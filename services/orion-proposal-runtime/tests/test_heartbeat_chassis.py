"""Regression coverage for proposal-runtime's bus-native SystemHealthV1 heartbeat wiring.

Part of the service-heartbeat rollout, see
docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md.
orion-proposal-runtime had no bus connection of any kind before this patch -- it polls Postgres
directly via `ProposalRuntimeWorker` (confirmed via `grep -rl "BaseChassis"` and a grep for
`ORION_BUS` returning nothing for this service prior to this patch). This is also the first test
file this service has -- there was no `tests/` directory before this patch.

`worker` (the real ProposalRuntimeWorker, owning a Postgres engine) is mocked out here so this
test exercises only the new heartbeat wiring in app/main.py's lifespan, without opening a real
bus or database connection.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI

REPO_ROOT = Path(__file__).resolve().parents[3]

# app.main constructs a module-level ProposalRuntimeWorker() at import time, which requires
# POSTGRES_URI (no default) and loads PROPOSAL_POLICY_PATH from disk (default is a path
# relative to repo root, which does not resolve when pytest's cwd is this service directory).
os.environ.setdefault("POSTGRES_URI", "postgresql://unused:5432/unused")
os.environ.setdefault(
    "PROPOSAL_POLICY_PATH", str(REPO_ROOT / "config" / "proposals" / "proposal_policy.v1.yaml")
)

import app.main as proposal_main
from orion.core.bus.bus_service_chassis import HeartbeatOnly


def test_build_heartbeat_chassis_uses_proposal_runtime_settings() -> None:
    from app.settings import get_settings

    chassis = proposal_main.build_heartbeat_chassis()
    s = get_settings()
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
    monkeypatch.setattr(proposal_main, "build_heartbeat_chassis", lambda: fake_chassis)
    monkeypatch.setattr(proposal_main.worker, "start", AsyncMock())
    monkeypatch.setattr(proposal_main.worker, "stop", AsyncMock())

    app = FastAPI()
    async with proposal_main.lifespan(app):
        assert proposal_main.heartbeat_chassis is fake_chassis
        fake_chassis.start_background.assert_awaited_once()

    fake_chassis.stop.assert_awaited_once()
    assert proposal_main.heartbeat_chassis is None
    proposal_main.worker.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_lifespan_survives_heartbeat_start_failure(monkeypatch) -> None:
    """Heartbeat startup failure must not prevent the real proposal worker from starting."""
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    fake_chassis.start_background.side_effect = RuntimeError("bus unreachable")
    monkeypatch.setattr(proposal_main, "build_heartbeat_chassis", lambda: fake_chassis)
    monkeypatch.setattr(proposal_main.worker, "start", AsyncMock())
    monkeypatch.setattr(proposal_main.worker, "stop", AsyncMock())

    app = FastAPI()
    async with proposal_main.lifespan(app):
        assert proposal_main.heartbeat_chassis is None
        proposal_main.worker.start.assert_awaited_once()
