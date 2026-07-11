"""Regression coverage for OrionBusAsync._run_rpc_only.

Live incident (2026-07-11, hub): every HarnessGovernorClient.run() call had been
falling back to the ad-hoc per-turn subscribe path forever — the pooled RPC worker
added in PR #943 had never once carried a reply. Root cause: _run_rpc_only() called
pubsub.get_message() before any reply-channel was subscribed. redis-py's real
PubSub.parse_response() raises RuntimeError("pubsub connection not set: did you
forget to call subscribe() or psubscribe()?") when `self.connection is None`, which
is true until the first subscribe() call — so the worker task died on its very first
loop iteration, on every single startup, with nothing logging the exception.
"""
from __future__ import annotations

import asyncio

import pytest
from unittest.mock import AsyncMock, patch

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.codec import OrionCodec


class _FakeConnection:
    """Stand-in for redis-py's Connection: just a truthy sentinel."""


class _FakePubSub:
    """Mimics redis.asyncio's PubSub closely enough to reproduce the real bug:
    `connection` is None until subscribe()/psubscribe() is called, and get_message()
    raises RuntimeError in that state exactly like the real parse_response() does.
    """

    def __init__(self) -> None:
        self.connection: _FakeConnection | None = None
        self.subscribed: set[str] = set()
        self._queue: asyncio.Queue = asyncio.Queue()
        self.get_message_override = None

    async def subscribe(self, *channels: str) -> None:
        self.connection = _FakeConnection()
        self.subscribed.update(channels)

    async def unsubscribe(self, *channels: str) -> None:
        self.subscribed.difference_update(channels)

    async def get_message(self, ignore_subscribe_messages: bool = True, timeout: float = 1.0):
        if self.get_message_override is not None:
            override = self.get_message_override
            self.get_message_override = None
            raise override
        if self.connection is None:
            raise RuntimeError(
                "pubsub connection not set: did you forget to call subscribe() or psubscribe()?"
            )
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def close(self) -> None:
        return None

    def push(self, msg: dict) -> None:
        self._queue.put_nowait(msg)


class _FakeRedis:
    """_create_pubsub_redis() replacement. Mirrors real redis-py: each pubsub() call
    (including the one _run_rpc_only makes on reconnect) returns a fresh PubSub with
    connection=None. `.current` always points at whichever one the worker is actually
    reading from, so tests can push messages/failures into the live instance.
    """

    def __init__(self) -> None:
        self.current: _FakePubSub | None = None

    def pubsub(self) -> _FakePubSub:
        self.current = _FakePubSub()
        return self.current

    async def close(self) -> None:
        return None


def _make_bus() -> tuple[OrionBusAsync, _FakeRedis]:
    bus = OrionBusAsync("redis://127.0.0.1:6379/0", enabled=True)
    bus.connect = AsyncMock()  # avoid a real Redis command-connection dial in _rpc_subscribe
    fake_redis = _FakeRedis()
    return bus, fake_redis


@pytest.mark.asyncio
async def test_run_rpc_only_survives_get_message_before_any_subscribe() -> None:
    """The worker must not crash before its first reply-channel subscription exists.
    Pre-fix, this died within one event-loop tick (RuntimeError from parse_response),
    so the task would be done() almost immediately.
    """
    bus, fake_redis = _make_bus()
    with patch.object(bus, "_create_pubsub_redis", return_value=fake_redis):
        bus.start_rpc_worker()
        try:
            await asyncio.sleep(0.3)
            assert bus._rpc_worker_task is not None
            assert not bus._rpc_worker_task.done(), (
                "RPC worker died before any subscribe() call — "
                "pre-subscribe get_message() crash regressed"
            )
        finally:
            await bus.close()


@pytest.mark.asyncio
async def test_run_rpc_only_delivers_message_after_late_subscribe() -> None:
    """Once a real caller subscribes a reply channel (as HarnessGovernorClient._run_via_worker
    does via _rpc_subscribe), the worker must still be alive and deliver the reply.
    """
    bus, fake_redis = _make_bus()
    with patch.object(bus, "_create_pubsub_redis", return_value=fake_redis):
        bus.start_rpc_worker()
        await asyncio.sleep(0.1)  # spins in the pre-subscribe guard; must not crash

        corr_id = "00000000-0000-4000-8000-000000000001"
        reply_channel = f"orion:test:result:{corr_id}"
        fut = asyncio.get_running_loop().create_future()
        bus._pending_rpc[(reply_channel, corr_id)] = fut

        async with bus._rpc_lock:
            await bus._rpc_subscribe(reply_channel)

        envelope = BaseEnvelope(
            kind="test.reply.v1",
            source=ServiceRef(name="test", version="0"),
            correlation_id=corr_id,
            payload={"ok": True},
        )
        codec = OrionCodec()
        fake_redis.current.push(
            {
                "type": "message",
                "channel": reply_channel.encode("utf-8"),
                "data": codec.encode(envelope),
            }
        )

        try:
            msg = await asyncio.wait_for(fut, timeout=2.0)
            assert msg is not None
        finally:
            await bus.close()


@pytest.mark.asyncio
async def test_run_rpc_only_reconnects_after_transport_error() -> None:
    """A genuine read failure (network blip) must not kill the worker permanently —
    it should reconnect/re-subscribe and keep serving future RPC replies.
    """
    bus, fake_redis = _make_bus()
    with patch.object(bus, "_create_pubsub_redis", return_value=fake_redis):
        bus.start_rpc_worker()

        corr_id = "00000000-0000-4000-8000-000000000002"
        reply_channel = f"orion:test:result:{corr_id}"
        async with bus._rpc_lock:
            await bus._rpc_subscribe(reply_channel)
        pubsub_before = fake_redis.current

        # Simulate a dropped connection on the next read.
        pubsub_before.get_message_override = ConnectionError("simulated transport drop")
        await asyncio.sleep(0.3)

        try:
            assert bus._rpc_worker_task is not None
            assert not bus._rpc_worker_task.done(), "worker died instead of reconnecting"
            assert fake_redis.current is not pubsub_before, "worker never reconnected to a fresh pubsub"
            assert reply_channel in fake_redis.current.subscribed, (
                "worker reconnected but did not re-subscribe the live reply channel"
            )

            fut = asyncio.get_running_loop().create_future()
            bus._pending_rpc[(reply_channel, corr_id)] = fut
            envelope = BaseEnvelope(
                kind="test.reply.v1",
                source=ServiceRef(name="test", version="0"),
                correlation_id=corr_id,
                payload={"ok": True},
            )
            codec = OrionCodec()
            fake_redis.current.push(
                {
                    "type": "message",
                    "channel": reply_channel.encode("utf-8"),
                    "data": codec.encode(envelope),
                }
            )
            msg = await asyncio.wait_for(fut, timeout=2.0)
            assert msg is not None
        finally:
            await bus.close()


@pytest.mark.asyncio
async def test_run_rpc_only_survives_bad_payload_in_handle_rpc_result() -> None:
    """A malformed/unexpected message must degrade like a transport error (reconnect
    and keep going), not kill the worker — otherwise a single bad payload reintroduces
    the exact "every future RPC call silently falls back to ad-hoc subscribe forever"
    failure this fix exists to close, just triggered by bad data instead of the
    pre-subscribe crash.
    """
    bus, fake_redis = _make_bus()
    with patch.object(bus, "_create_pubsub_redis", return_value=fake_redis):
        bus.start_rpc_worker()

        corr_id = "00000000-0000-4000-8000-000000000003"
        reply_channel = f"orion:test:result:{corr_id}"
        async with bus._rpc_lock:
            await bus._rpc_subscribe(reply_channel)

        real_handle_rpc_result = bus._handle_rpc_result
        calls = {"n": 0}

        async def _flaky_handle_rpc_result(msg: dict) -> None:
            calls["n"] += 1
            if calls["n"] == 1:
                raise ValueError("simulated malformed payload")
            await real_handle_rpc_result(msg)

        bus._handle_rpc_result = _flaky_handle_rpc_result

        fake_redis.current.push(
            {"type": "message", "channel": reply_channel.encode("utf-8"), "data": b"garbage"}
        )
        await asyncio.sleep(0.3)

        try:
            assert calls["n"] >= 1, "handler was never invoked"
            assert bus._rpc_worker_task is not None
            assert not bus._rpc_worker_task.done(), (
                "worker died from a bad payload instead of reconnecting/continuing"
            )

            fut = asyncio.get_running_loop().create_future()
            bus._pending_rpc[(reply_channel, corr_id)] = fut
            envelope = BaseEnvelope(
                kind="test.reply.v1",
                source=ServiceRef(name="test", version="0"),
                correlation_id=corr_id,
                payload={"ok": True},
            )
            codec = OrionCodec()
            fake_redis.current.push(
                {
                    "type": "message",
                    "channel": reply_channel.encode("utf-8"),
                    "data": codec.encode(envelope),
                }
            )
            msg = await asyncio.wait_for(fut, timeout=2.0)
            assert msg is not None
        finally:
            await bus.close()
