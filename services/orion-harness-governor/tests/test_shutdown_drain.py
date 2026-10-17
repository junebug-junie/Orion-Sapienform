"""Restart drain: a turn already running when the governor shuts down gets to finish.

Live 2026-10-16: a redeploy killed a 229-step run at once, because shutdown cancelled every
in-flight turn. The worker now stops taking requests, waits up to `drain_sec`, then cancels."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

from app import bus_listener
from app.settings import HarnessGovernorSettings
from test_lane_dispatch_channels import _FakeBus

SERVICE = Path(__file__).resolve().parents[1]


def _worker(monkeypatch, drain_sec):
    queue: asyncio.Queue = asyncio.Queue()
    bus = _FakeBus({"agent:channel": queue})
    monkeypatch.setattr(bus_listener.settings, "orion_bus_enabled", True)
    monkeypatch.setattr(bus_listener.settings, "orion_harness_governor_enabled", True)
    stop = asyncio.Event()
    task = asyncio.create_task(
        bus_listener.run_bus_worker("agent:channel", stop, lane="agent", bus=bus, drain_sec=drain_sec)
    )
    return queue, stop, task


async def _until(predicate, timeout=2.0):
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while not predicate():
        assert loop.time() < end, "condition not reached"
        await asyncio.sleep(0.02)


@pytest.mark.asyncio
async def test_a_running_turn_finishes_during_the_drain(monkeypatch):
    events: list[str] = []
    release = asyncio.Event()

    async def handle(_bus, msg):
        events.append("start")
        try:
            await release.wait()
            events.append("finished")
        except asyncio.CancelledError:
            events.append("cancelled")
            raise

    monkeypatch.setattr(bus_listener, "_handle_bus_message", handle)
    queue, stop, task = _worker(monkeypatch, drain_sec=5)
    await queue.put({"type": "message", "data": {}})
    await _until(lambda: "start" in events)
    stop.set()
    await asyncio.sleep(1.5)  # past the loop noticing the stop flag
    assert not task.done(), "the worker must wait for the running turn"
    assert "cancelled" not in events
    release.set()
    await asyncio.wait_for(task, 2)
    assert events == ["start", "finished"]


@pytest.mark.asyncio
async def test_a_turn_still_running_after_the_budget_is_cancelled(monkeypatch):
    events: list[str] = []

    async def handle(_bus, msg):
        events.append("start")
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            events.append("cancelled")
            raise

    monkeypatch.setattr(bus_listener, "_handle_bus_message", handle)
    queue, stop, task = _worker(monkeypatch, drain_sec=0.3)
    await queue.put({"type": "message", "data": {}})
    await _until(lambda: "start" in events)
    stop.set()
    await asyncio.wait_for(task, 4)
    assert events == ["start", "cancelled"]


@pytest.mark.asyncio
async def test_zero_budget_cancels_at_once_as_before(monkeypatch):
    events: list[str] = []

    async def handle(_bus, msg):
        events.append("start")
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            events.append("cancelled")
            raise

    monkeypatch.setattr(bus_listener, "_handle_bus_message", handle)
    queue, stop, task = _worker(monkeypatch, drain_sec=0)
    await queue.put({"type": "message", "data": {}})
    await _until(lambda: "start" in events)
    stop.set()
    await asyncio.wait_for(task, 3)
    assert events == ["start", "cancelled"]


@pytest.mark.asyncio
async def test_no_new_request_is_taken_once_draining(monkeypatch):
    started: list[str] = []
    release = asyncio.Event()

    async def handle(_bus, msg):
        started.append(msg["data"]["name"])
        await release.wait()

    monkeypatch.setattr(bus_listener, "_handle_bus_message", handle)
    queue, stop, task = _worker(monkeypatch, drain_sec=5)
    await queue.put({"type": "message", "data": {"name": "first"}})
    await _until(lambda: started == ["first"])
    stop.set()
    await asyncio.sleep(1.5)
    await queue.put({"type": "message", "data": {"name": "late"}})
    await asyncio.sleep(0.3)
    release.set()
    await asyncio.wait_for(task, 2)
    assert started == ["first"]


def test_docker_grace_period_outlasts_the_drain_by_default():
    """Docker SIGKILLs at stop_grace_period; a shorter value silently voids the drain."""
    compose = (SERVICE / "docker-compose.yml").read_text()
    grace = int(re.search(r"stop_grace_period:\s*\$\{HARNESS_STOP_GRACE_SEC:-(\d+)\}s", compose).group(1))
    env = (SERVICE / ".env_example").read_text()
    assert f"HARNESS_STOP_GRACE_SEC={grace}" in env
    fields = HarnessGovernorSettings.model_fields
    drain = max(
        fields["harness_shutdown_drain_sec_chat"].default, fields["harness_shutdown_drain_sec_agent"].default
    )
    assert grace >= drain + 60
    for key, field in (
        ("HARNESS_SHUTDOWN_DRAIN_SEC_CHAT", "harness_shutdown_drain_sec_chat"),
        ("HARNESS_SHUTDOWN_DRAIN_SEC_AGENT", "harness_shutdown_drain_sec_agent"),
    ):
        assert f"{key}={int(fields[field].default)}" in env
        assert f"{key}=${{{key}:-{int(fields[field].default)}}}" in compose


@pytest.mark.asyncio
async def test_lifespan_keeps_the_cancel_listener_until_the_lanes_have_drained(monkeypatch):
    """A cancel from durable-runs must still reach a draining turn, so the cancel listener
    stops only after both lanes are done."""
    from app import main

    order: list[str] = []

    class _Bus:
        def __init__(self, *a, **k):
            self.enabled = True

        async def connect(self):
            return None

        async def close(self):
            return None

    class _Publisher:
        def start(self):
            return None

        async def stop(self):
            return None

    async def fake_lane(*, stop_event, lane, bus, drain_sec):
        await stop_event.wait()
        order.append(f"{lane}:draining:{drain_sec}")
        await asyncio.sleep(0.2)
        order.append(f"{lane}:done")

    async def fake_cancel(stop_event):
        try:
            await stop_event.wait()
        finally:
            order.append("cancel:stopped")

    async def _noop(*a, **k):
        return None

    monkeypatch.setattr(main, "OrionBusAsync", _Bus)
    monkeypatch.setattr(main, "build_rpc_health_publisher", lambda getter: _Publisher())
    class _Chassis:
        async def start_background(self):
            return None

        async def stop(self):
            return None

    monkeypatch.setattr(main, "build_heartbeat_chassis", lambda: _Chassis())
    monkeypatch.setattr(main, "run_bus_worker", fake_lane)
    monkeypatch.setattr(main, "run_cancel_worker", fake_cancel)
    monkeypatch.setattr(main, "stop_warm_pool", _noop)
    monkeypatch.setattr(main.settings, "harness_fcc_chat_warm_pool_enabled", False)
    monkeypatch.setattr(main.settings, "orion_bus_enabled", False)
    monkeypatch.setattr(main.settings, "harness_shutdown_drain_sec_chat", 7.0)
    monkeypatch.setattr(main.settings, "harness_shutdown_drain_sec_agent", 9.0)

    async with main.lifespan(main.app):
        await asyncio.sleep(0.05)
    assert "chat:draining:7.0" in order and "agent:draining:9.0" in order
    assert order.index("cancel:stopped") > order.index("chat:done")
    assert order.index("cancel:stopped") > order.index("agent:done")
