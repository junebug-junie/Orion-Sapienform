"""The heartbeat reports bus listener state, so equilibrium sees a dead listener.

Live 2026-10-20: substrate-runtime's finalize_appraisal listener died while its
heartbeat kept saying "ok", and chat lost reflection for hours with nothing in
equilibrium showing it.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from orion.core.bus import resilience
from orion.core.bus.bus_schemas import ServiceRef
from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly


async def _one_heartbeat(chassis) -> dict:
    sent: list = []

    async def publish(channel, env):
        sent.append(env)
        chassis._stop.set()

    chassis.bus = SimpleNamespace(publish=publish, reconnect=AsyncMock())
    chassis._source = lambda: ServiceRef(name="svc", node="n", version="0")
    chassis._stop.clear()
    await chassis._heartbeat_loop()
    assert len(sent) == 1
    return sent[0].payload


def _chassis() -> HeartbeatOnly:
    return HeartbeatOnly(
        ChassisConfig(
            service_name="svc", service_version="0", node_name="n",
            bus_url="redis://localhost:6379/0", bus_enabled=True, heartbeat_interval_sec=0.01,
        )
    )


@pytest.fixture
def states(monkeypatch):
    registry: dict = {}
    monkeypatch.setattr(resilience, "_SUBSCRIPTION_STATES", registry)
    return registry


@pytest.mark.asyncio
async def test_down_listener_makes_the_heartbeat_degraded(states) -> None:
    states["a"] = resilience.SubscriptionState(name="a", channel="c1", subscribed=True)
    states["b"] = resilience.SubscriptionState(name="b", channel="c2", subscribed=False)
    payload = await _one_heartbeat(_chassis())
    assert payload["status"] == "degraded"
    assert payload["details"]["listeners"] == {"ok": False, "down": ["b"], "count": 2}


@pytest.mark.asyncio
async def test_all_listeners_up_is_ok_with_a_summary(states) -> None:
    states["a"] = resilience.SubscriptionState(name="a", channel="c1", subscribed=True)
    payload = await _one_heartbeat(_chassis())
    assert payload["status"] == "ok"
    assert payload["details"]["listeners"] == {"ok": True, "down": [], "count": 1}


@pytest.mark.asyncio
async def test_process_without_listeners_is_unchanged(states) -> None:
    payload = await _one_heartbeat(_chassis())
    assert payload["status"] == "ok" and "listeners" not in payload["details"]


@pytest.mark.asyncio
async def test_listener_health_failure_never_breaks_the_heartbeat(states, monkeypatch) -> None:
    import orion.core.bus.bus_service_chassis as chassis_mod

    def boom():
        raise RuntimeError("registry broke")

    monkeypatch.setattr(chassis_mod, "subscription_health", boom)
    payload = await _one_heartbeat(_chassis())
    assert payload["status"] == "ok" and "listeners" not in payload["details"]
