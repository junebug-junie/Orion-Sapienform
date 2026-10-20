"""gpu-cluster-power publishes the standard SystemHealthV1 heartbeat via HeartbeatOnly.

Before this patch the service ran its own hand-rolled 30s loop on the command
listener's bus connection and reported the made-up node name "psu-node" (live in
equilibrium's state hash as `gpu-cluster-power@psu-node`, 2026-10-20). The heartbeat
now comes from orion.core.bus.bus_service_chassis.HeartbeatOnly on its own bus
connection, with the real NODE_NAME. Pattern:
services/orion-substrate-runtime/tests/test_heartbeat_chassis.py.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from orion.core.bus.bus_service_chassis import HeartbeatOnly

from app import api
from app.service import PsuService
from app.settings import settings


def test_build_heartbeat_chassis_uses_service_settings(monkeypatch) -> None:
    monkeypatch.setattr(settings, "orion_bus_url", "redis://example.invalid:6379/0")
    monkeypatch.setattr(settings, "orion_bus_enabled", True)
    chassis = api.build_heartbeat_chassis()
    assert isinstance(chassis, HeartbeatOnly)
    assert chassis.cfg.service_name == settings.service_name == "gpu-cluster-power"
    assert chassis.cfg.service_version == settings.service_version
    assert chassis.cfg.node_name == settings.node_name
    assert chassis.cfg.node_name != "psu-node"
    assert chassis.cfg.bus_url == "redis://example.invalid:6379/0"
    assert chassis.cfg.bus_enabled is True
    assert chassis.cfg.heartbeat_interval_sec == settings.heartbeat_interval_sec
    assert chassis.cfg.health_channel == "orion:system:health"


def test_build_heartbeat_chassis_disables_bus_without_url(monkeypatch) -> None:
    monkeypatch.setattr(settings, "orion_bus_url", None)
    chassis = api.build_heartbeat_chassis()
    assert chassis.cfg.bus_enabled is False


def test_old_hand_rolled_heartbeat_loop_is_gone() -> None:
    """Two publishers for one service name would show up as two equilibrium rows."""
    assert not hasattr(api, "heartbeat_loop")


def test_bus_disabled_starts_no_heartbeat(monkeypatch) -> None:
    built = []
    monkeypatch.setattr(api, "build_heartbeat_chassis", lambda: built.append(1))
    monkeypatch.setattr(PsuService, "init_bus", AsyncMock())
    monkeypatch.setattr(settings, "orion_bus_enabled", False)
    app = api.create_app()

    async def _run():
        await app.router.startup()
        assert app.state.heartbeat_chassis is None
        await app.router.shutdown()

    asyncio.run(_run())
    assert built == []


def test_startup_starts_and_shutdown_stops_heartbeat(monkeypatch) -> None:
    monkeypatch.setattr(settings, "orion_bus_enabled", True)
    fake = AsyncMock(spec=HeartbeatOnly)
    monkeypatch.setattr(api, "build_heartbeat_chassis", lambda: fake)
    monkeypatch.setattr(PsuService, "init_bus", AsyncMock())  # no real bus
    app = api.create_app()

    async def _run():
        await app.router.startup()
        assert app.state.heartbeat_chassis is fake
        fake.start_background.assert_awaited_once()
        await app.router.shutdown()

    asyncio.run(_run())
    fake.stop.assert_awaited_once()
    assert app.state.heartbeat_chassis is None


def test_heartbeat_start_failure_is_not_fatal(monkeypatch) -> None:
    monkeypatch.setattr(settings, "orion_bus_enabled", True)
    fake = AsyncMock(spec=HeartbeatOnly)
    fake.start_background.side_effect = RuntimeError("bus unreachable")
    monkeypatch.setattr(api, "build_heartbeat_chassis", lambda: fake)
    monkeypatch.setattr(PsuService, "init_bus", AsyncMock())
    app = api.create_app()

    async def _run():
        await app.router.startup()  # must not raise
        assert app.state.heartbeat_chassis is None
        await app.router.shutdown()

    asyncio.run(_run())
    fake.stop.assert_not_awaited()
