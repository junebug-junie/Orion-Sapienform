"""Regression coverage for orion-mind's bus-native SystemHealthV1 heartbeat wiring.

Part of the wider service-heartbeat rollout following pilot-5 (PR #1350), see
docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md.
orion-mind had no FastAPI `lifespan` at all before this patch (confirmed via `grep -rl
"BaseChassis"` returning nothing for this service, and `app = FastAPI(...)` taking no
`lifespan=` kwarg) -- it does make its own per-request bus connections for LLM-gateway RPC
(`app/llm_client.py`, gated by `MIND_LLM_USE_BUS`), unaffected by this patch. This patch also
fixes a pre-existing bug: `ORION_BUS_URL`'s settings.py default and docker-compose.yml default
were both `redis://redis:6379/0`, not the real tailscale node IP (the checked-in `.env_example`
already had the correct value) -- corrected to `100.92.216.81`, matching the fix PR #1350 made
for orion-harness-governor's same class of bug.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI

_guard_path = Path(__file__).resolve().parent / "_mind_import_guard.py"


def _mind_prep():
    spec = importlib.util.spec_from_file_location("_mind_guard_lazy_heartbeat", _guard_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ensure_orion_mind_app()
    import app.main as main_mod

    return main_mod


def test_build_heartbeat_chassis_uses_mind_settings() -> None:
    from orion.core.bus.bus_service_chassis import HeartbeatOnly

    mind_main = _mind_prep()
    chassis = mind_main.build_heartbeat_chassis()
    assert isinstance(chassis, HeartbeatOnly)
    assert chassis.cfg.service_name == mind_main.settings.SERVICE_NAME
    assert chassis.cfg.service_version == mind_main.settings.SERVICE_VERSION
    assert chassis.cfg.node_name == mind_main.settings.NODE_NAME
    assert chassis.cfg.bus_url == mind_main.settings.ORION_BUS_URL
    assert chassis.cfg.bus_enabled == mind_main.settings.ORION_BUS_ENABLED
    assert chassis.cfg.heartbeat_interval_sec == mind_main.settings.HEARTBEAT_INTERVAL_SEC
    assert chassis.cfg.health_channel == "orion:system:health"


def test_bus_url_default_is_real_tailscale_node() -> None:
    """Regression guard for the redis://redis placeholder bug fixed in this patch."""
    mind_main = _mind_prep()
    assert mind_main.settings.ORION_BUS_URL != "redis://redis:6379/0"
    assert "100.92.216.81" in mind_main.settings.ORION_BUS_URL


@pytest.mark.asyncio
async def test_lifespan_starts_and_stops_heartbeat_chassis(monkeypatch) -> None:
    from orion.core.bus.bus_service_chassis import HeartbeatOnly

    mind_main = _mind_prep()
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    monkeypatch.setattr(mind_main, "build_heartbeat_chassis", lambda: fake_chassis)

    app = FastAPI()
    async with mind_main.lifespan(app):
        assert mind_main.heartbeat_chassis is fake_chassis
        fake_chassis.start_background.assert_awaited_once()

    fake_chassis.stop.assert_awaited_once()
    assert mind_main.heartbeat_chassis is None


@pytest.mark.asyncio
async def test_lifespan_survives_heartbeat_start_failure(monkeypatch) -> None:
    """Heartbeat startup failure must not raise out of lifespan startup."""
    from orion.core.bus.bus_service_chassis import HeartbeatOnly

    mind_main = _mind_prep()
    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    fake_chassis.start_background.side_effect = RuntimeError("bus unreachable")
    monkeypatch.setattr(mind_main, "build_heartbeat_chassis", lambda: fake_chassis)

    app = FastAPI()
    async with mind_main.lifespan(app):
        assert mind_main.heartbeat_chassis is None
