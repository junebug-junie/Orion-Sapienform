"""Regression coverage for orion-thought's bus-native SystemHealthV1 heartbeat wiring.

Part of the service-heartbeat rollout, see
docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md, following the
pilot-5 pattern proved out in PR #1350. orion-thought had no bus-native heartbeat before this
patch (confirmed via `grep -rl "BaseChassis"` returning nothing for this service). This patch
also fixes a pre-existing bug found while wiring: `ORION_BUS_URL` in both `app/settings.py` and
`.env_example` was the literal placeholder `100.x.x.x`, not the real tailscale node IP --
corrected to `100.92.216.81` (same class of bug PR #1350 fixed for harness-governor).

The other four lifespan workers (bus/reverie/reverie_chain/reasoning) are disabled here via
`orion_bus_enabled=False`/`reverie_enabled=False`/`reverie_chain_enabled=False`, matching
test_main_lifespan.py's existing convention, so this test exercises only the new heartbeat
wiring without opening a real bus connection.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import app.main as thought_main
from app.settings import settings
from orion.core.bus.bus_service_chassis import HeartbeatOnly


def test_build_heartbeat_chassis_uses_thought_settings() -> None:
    chassis = thought_main.build_heartbeat_chassis()
    assert isinstance(chassis, HeartbeatOnly)
    assert chassis.cfg.service_name == settings.service_name
    assert chassis.cfg.service_version == settings.service_version
    assert chassis.cfg.node_name == settings.node_name
    assert chassis.cfg.bus_url == settings.orion_bus_url
    assert chassis.cfg.bus_enabled == settings.orion_bus_enabled
    assert chassis.cfg.heartbeat_interval_sec == settings.heartbeat_interval_sec
    assert chassis.cfg.health_channel == "orion:system:health"


def test_orion_bus_url_is_not_placeholder() -> None:
    """Regression guard: ORION_BUS_URL defaulted to the literal placeholder 100.x.x.x
    before this patch, which would have made the heartbeat chassis (and every other bus
    connection this service opens) unable to reach the real bus."""
    assert "x.x.x" not in settings.orion_bus_url


@pytest.mark.asyncio
async def test_lifespan_starts_and_stops_heartbeat_chassis(monkeypatch) -> None:
    monkeypatch.setattr(settings, "orion_bus_enabled", False)
    monkeypatch.setattr(settings, "reverie_enabled", False)
    monkeypatch.setattr(settings, "reverie_chain_enabled", False)

    async def _fake_warm_pool() -> None:
        return None

    monkeypatch.setattr(thought_main, "warm_pool", _fake_warm_pool)

    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    monkeypatch.setattr(thought_main, "build_heartbeat_chassis", lambda: fake_chassis)

    app = SimpleNamespace(state=SimpleNamespace())
    async with thought_main.lifespan(app):
        assert thought_main.heartbeat_chassis is fake_chassis
        fake_chassis.start_background.assert_awaited_once()

    fake_chassis.stop.assert_awaited_once()
    assert thought_main.heartbeat_chassis is None


@pytest.mark.asyncio
async def test_lifespan_survives_heartbeat_start_failure(monkeypatch) -> None:
    """Heartbeat startup failure must not prevent the rest of orion-thought's boot."""
    monkeypatch.setattr(settings, "orion_bus_enabled", False)
    monkeypatch.setattr(settings, "reverie_enabled", False)
    monkeypatch.setattr(settings, "reverie_chain_enabled", False)

    async def _fake_warm_pool() -> None:
        return None

    monkeypatch.setattr(thought_main, "warm_pool", _fake_warm_pool)

    fake_chassis = AsyncMock(spec=HeartbeatOnly)
    fake_chassis.start_background.side_effect = RuntimeError("bus unreachable")
    monkeypatch.setattr(thought_main, "build_heartbeat_chassis", lambda: fake_chassis)

    app = SimpleNamespace(state=SimpleNamespace())
    async with thought_main.lifespan(app):
        assert thought_main.heartbeat_chassis is None
        assert app.state.bus_stop_event is not None
