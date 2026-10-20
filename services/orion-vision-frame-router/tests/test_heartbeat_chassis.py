"""vision-frame-router publishes the standard SystemHealthV1 heartbeat via HeartbeatOnly.

Before this patch FrameRouterService._health_loop published its own envelope on the
frames/replies bus and sent no node at all, so equilibrium keyed it
`vision-frame-router@unknown` (live 2026-10-20). The heartbeat now comes from
orion.core.bus.bus_service_chassis.HeartbeatOnly on its own bus connection, started
and stopped by the app lifespan, with the router counters still in `details`.
"""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import AsyncMock

from fastapi import FastAPI

import app.main as router_main
from app.state import PendingTask
from orion.core.bus.bus_service_chassis import HeartbeatOnly


def test_build_heartbeat_chassis_uses_service_settings() -> None:
    chassis = router_main.build_heartbeat_chassis()
    s = router_main.service.settings
    assert isinstance(chassis, HeartbeatOnly)
    assert chassis.cfg.service_name == s.SERVICE_NAME == "vision-frame-router"
    assert chassis.cfg.service_version == s.SERVICE_VERSION
    assert chassis.cfg.node_name == s.NODE_NAME
    assert chassis.cfg.node_name  # the old loop sent node=None
    assert chassis.cfg.bus_url == s.ORION_BUS_URL
    assert chassis.cfg.heartbeat_interval_sec == s.HEALTH_INTERVAL_SECONDS
    assert chassis.cfg.health_channel == s.CHANNEL_SYSTEM_HEALTH == "orion:system:health"
    assert chassis._heartbeat_details == router_main.service.health_details


def test_old_health_loop_is_gone() -> None:
    assert not hasattr(router_main.FrameRouterService, "_health_loop")


def _pending(cid: str, lane: str) -> PendingTask:
    return PendingTask(
        correlation_id=cid, camera_id="cam0", frame_ts=None, image_path="/x.jpg",
        task_type="retina_fast", dispatched_at=0.0, reply_to="r", lane=lane,
    )


def test_health_details_keep_router_counters() -> None:
    svc = router_main.FrameRouterService(bus=AsyncMock())
    svc.metrics.frames_seen_total = 7
    svc.metrics.skip_reason_counts["cooldown"] = 3
    svc.state.pending["a"] = _pending("a", "host")
    svc.state.pending["b"] = _pending("b", "caption")
    d = svc.health_details()
    assert d["frames_seen_total"] == 7
    assert d["skip_reason_counts"] == {"cooldown": 3}
    assert d["inflight_total"] == svc.state.inflight_total() == 1
    assert d["caption_inflight"] == svc.state.inflight_lane("caption") == 1
    assert d["pending_count"] == 2


def test_health_details_survive_concurrent_mutation() -> None:
    """HeartbeatOnly calls the provider in a worker thread while the loop mutates state."""
    svc = router_main.FrameRouterService(bus=AsyncMock())
    stop = threading.Event()

    def churn():
        i = 0
        while not stop.is_set():
            svc.state.pending[f"k{i}"] = _pending(f"k{i}", "host")
            svc.state.pending.pop(f"k{i - 50}", None)
            svc.metrics.skip_reason_counts[f"r{i % 97}"] = i
            i += 1

    t = threading.Thread(target=churn)
    t.start()
    try:
        for _ in range(2000):
            svc.health_details()  # must never raise
    finally:
        stop.set()
        t.join()


def test_lifespan_starts_and_stops_heartbeat(monkeypatch) -> None:
    fake = AsyncMock(spec=HeartbeatOnly)
    monkeypatch.setattr(router_main, "build_heartbeat_chassis", lambda: fake)
    monkeypatch.setattr(router_main.service, "start", AsyncMock())
    monkeypatch.setattr(router_main.service, "stop", AsyncMock())

    async def _run():
        async with router_main.lifespan(FastAPI()):
            assert router_main.heartbeat_chassis is fake
            fake.start_background.assert_awaited_once()

    asyncio.run(_run())
    fake.stop.assert_awaited_once()
    assert router_main.heartbeat_chassis is None
    router_main.service.stop.assert_awaited_once()


def test_lifespan_survives_heartbeat_start_failure(monkeypatch) -> None:
    fake = AsyncMock(spec=HeartbeatOnly)
    fake.start_background.side_effect = RuntimeError("bus unreachable")
    monkeypatch.setattr(router_main, "build_heartbeat_chassis", lambda: fake)
    monkeypatch.setattr(router_main.service, "start", AsyncMock())
    monkeypatch.setattr(router_main.service, "stop", AsyncMock())

    async def _run():
        async with router_main.lifespan(FastAPI()):
            assert router_main.heartbeat_chassis is None
            router_main.service.start.assert_awaited_once()

    asyncio.run(_run())
    fake.stop.assert_not_awaited()
    router_main.service.stop.assert_awaited_once()
