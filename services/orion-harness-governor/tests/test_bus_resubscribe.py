"""Both governor listeners survive a Redis drop.

Live 2026-10-20: a bare `async with bus.subscribe(...)` loop ended on one redis
ConnectionError/TimeoutError while the process stayed up and looked healthy. The
lane workers and the cancel worker now resubscribe through `run_subscription`."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest

from app import bus_listener, cancel_listener
from test_lane_dispatch_channels import _FakePubSub, _PassthroughCodec


class _DropsFirstSubscribeBus:
    """The first subscribe raises like a dropped Redis; later ones feed `queue`."""

    def __init__(self, queue: asyncio.Queue) -> None:
        self._queue = queue
        self.codec = _PassthroughCodec()
        self.subscribe_calls = 0

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    @asynccontextmanager
    async def subscribe(self, channel: str):
        self.subscribe_calls += 1
        if self.subscribe_calls == 1:
            raise ConnectionError("Error 111 connecting to redis")
        yield _FakePubSub(channel, self._queue)


async def _until(predicate, timeout=5.0):
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while not predicate():
        assert loop.time() < end, "condition not reached"
        await asyncio.sleep(0.02)


class _Fence:
    lease_id = "lease"
    generation = 1


class _Req:
    """Stands in for HarnessRunRequestV1: only the fence key fields are read."""

    def __init__(self, corr: str) -> None:
        self.correlation_id = corr
        self.gpu_lease = _Fence()

    @classmethod
    def model_validate(cls, payload: dict) -> "_Req":
        return cls(payload["n"])


@pytest.mark.asyncio
async def test_lane_worker_resubscribes_and_still_runs_turns_concurrently(monkeypatch) -> None:
    started: list[str] = []
    release = asyncio.Event()

    async def handle(_bus, msg):
        started.append(msg["data"]["n"])
        await release.wait()

    monkeypatch.setattr(bus_listener, "_handle_bus_message", handle)
    monkeypatch.setattr(bus_listener, "HarnessRunRequestV1", _Req)
    monkeypatch.setattr(bus_listener.settings, "orion_bus_enabled", True)
    monkeypatch.setattr(bus_listener.settings, "orion_harness_governor_enabled", True)
    queue: asyncio.Queue = asyncio.Queue()
    bus = _DropsFirstSubscribeBus(queue)
    stop = asyncio.Event()
    task = asyncio.create_task(bus_listener.run_bus_worker("chat:channel", stop, lane="chat", bus=bus))
    # Admitted turns skip the legacy lock, so both run at once: the handler starts
    # a task and does not hold up the next read. The duplicate of "a" is dropped.
    for n in ("a", "b", "a"):
        await queue.put({"type": "message", "data": {"n": n, "gpu_lease": {}}})
    await _until(lambda: len(started) == 2 and queue.empty())
    await asyncio.sleep(0.1)
    assert bus.subscribe_calls == 2
    assert sorted(started) == ["a", "b"]
    release.set()
    stop.set()
    await asyncio.wait_for(task, 5)


@pytest.mark.asyncio
async def test_cancel_worker_resubscribes_after_a_drop(monkeypatch) -> None:
    seen: list[dict] = []

    async def handle(_bus, msg):
        seen.append(msg)

    queue: asyncio.Queue = asyncio.Queue()
    bus = _DropsFirstSubscribeBus(queue)
    monkeypatch.setattr(cancel_listener, "OrionBusAsync", lambda url: bus)
    monkeypatch.setattr(cancel_listener, "handle_cancel_bus_message", handle)
    monkeypatch.setattr(cancel_listener.settings, "orion_bus_enabled", True)
    monkeypatch.setattr(cancel_listener.settings, "orion_harness_governor_enabled", True)
    stop = asyncio.Event()
    task = asyncio.create_task(cancel_listener.run_cancel_worker(stop))
    await queue.put({"type": "message", "data": {"correlation_id": "c1"}})
    await _until(lambda: len(seen) == 1)
    assert bus.subscribe_calls == 2
    stop.set()
    await asyncio.wait_for(task, 5)


class _DropsMidStreamBus(_DropsFirstSubscribeBus):
    """The first session hands out one message, then its read raises like a dropped socket."""

    @asynccontextmanager
    async def subscribe(self, channel: str):
        self.subscribe_calls += 1
        if self.subscribe_calls == 1:
            yield _DroppingPubSub(self._queue)
        else:
            yield _FakePubSub(channel, self._queue)


class _DroppingPubSub:
    def __init__(self, queue: asyncio.Queue) -> None:
        self._queue = queue
        self._reads = 0

    async def get_message(self, ignore_subscribe_messages: bool = True, timeout: float = 0.0):
        self._reads += 1
        if self._reads == 1:
            return await self._queue.get()
        raise ConnectionError("Connection closed by server.")


@pytest.mark.asyncio
async def test_lane_worker_keeps_inflight_turns_across_a_mid_stream_drop(monkeypatch) -> None:
    """A turn running when the socket drops keeps running, and its duplicate is still refused."""
    started: list[str] = []
    release = asyncio.Event()

    async def handle(_bus, msg):
        started.append(msg["data"]["n"])
        await release.wait()

    monkeypatch.setattr(bus_listener, "_handle_bus_message", handle)
    monkeypatch.setattr(bus_listener, "HarnessRunRequestV1", _Req)
    monkeypatch.setattr(bus_listener.settings, "orion_bus_enabled", True)
    monkeypatch.setattr(bus_listener.settings, "orion_harness_governor_enabled", True)
    queue: asyncio.Queue = asyncio.Queue()
    bus = _DropsMidStreamBus(queue)
    stop = asyncio.Event()
    task = asyncio.create_task(bus_listener.run_bus_worker("chat:channel", stop, lane="chat", bus=bus))
    await queue.put({"type": "message", "data": {"n": "a", "gpu_lease": {}}})
    await _until(lambda: started == ["a"])
    await _until(lambda: bus.subscribe_calls == 2)
    for n in ("a", "b"):  # the duplicate of the still-running "a" is dropped after the resubscribe
        await queue.put({"type": "message", "data": {"n": n, "gpu_lease": {}}})
    await _until(lambda: len(started) == 2 and queue.empty())
    await asyncio.sleep(0.1)
    assert started == ["a", "b"]
    release.set()
    stop.set()
    await asyncio.wait_for(task, 5)
