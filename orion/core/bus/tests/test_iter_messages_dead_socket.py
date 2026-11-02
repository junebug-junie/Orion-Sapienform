"""iter_messages must notice a pub/sub socket the server already dropped.

Regression for 2026-11-02: circe's gpu-lane controller blocked in ``pubsub.listen()`` for 13 h on
a socket that still read ESTABLISHED locally while the bus had no subscriber for it.
"""
from __future__ import annotations

import asyncio

import pytest
from redis import asyncio as aioredis

from orion.core.bus.async_service import OrionBusAsync

FAST = {"idle_ping_sec": 0.05, "pong_deadline_sec": 0.05, "poll_sec": 0.01}


class FakePubSub:
    """Stand-in for redis.asyncio PubSub: get_message(timeout) returns None on silence."""

    def __init__(self, *, answers_ping: bool, messages: list[dict] | None = None) -> None:
        self.subscribed = True
        self.answers_ping = answers_ping
        self.queue: list[dict] = list(messages or [])
        self.pings = 0

    async def get_message(self, timeout: float | None = 0.0):
        if self.queue:
            return self.queue.pop(0)
        await asyncio.sleep(timeout or 0)
        return None

    async def ping(self) -> None:
        self.pings += 1
        if self.answers_ping:
            self.queue.append({"type": "pong", "pattern": None, "channel": None, "data": b""})


def _bus() -> OrionBusAsync:
    return OrionBusAsync(url="redis://localhost:6379/0", enabled=False)


async def _drain(bus: OrionBusAsync, pubsub: FakePubSub, seconds: float) -> list[dict]:
    got: list[dict] = []

    async def run() -> None:
        async for msg in bus.iter_messages(pubsub, **FAST):
            got.append(msg)

    try:
        await asyncio.wait_for(run(), timeout=seconds)
    except asyncio.TimeoutError:
        pass
    return got


@pytest.mark.asyncio
async def test_dead_socket_raises_connection_error() -> None:
    pubsub = FakePubSub(answers_ping=False)
    with pytest.raises(aioredis.ConnectionError, match="pubsub silent"):
        await asyncio.wait_for(_drain_raise(_bus(), pubsub), timeout=2.0)
    assert pubsub.pings == 1


async def _drain_raise(bus: OrionBusAsync, pubsub: FakePubSub) -> None:
    async for _ in bus.iter_messages(pubsub, **FAST):
        pass


@pytest.mark.asyncio
async def test_quiet_live_socket_keeps_iterating_and_hides_pongs() -> None:
    pubsub = FakePubSub(answers_ping=True)
    got = await _drain(_bus(), pubsub, seconds=0.5)
    assert got == []  # pongs and silence are never yielded
    assert pubsub.pings >= 2  # it kept pinging and stayed alive across several idle windows


@pytest.mark.asyncio
async def test_yields_only_data_messages() -> None:
    msgs = [
        {"type": "subscribe", "channel": b"c", "data": 1},
        {"type": "message", "channel": b"c", "data": b"x"},
        {"type": "pmessage", "pattern": b"c*", "channel": b"c1", "data": b"y"},
    ]
    pubsub = FakePubSub(answers_ping=True, messages=msgs)
    got = await _drain(_bus(), pubsub, seconds=0.2)
    assert [m["data"] for m in got] == [b"x", b"y"]


@pytest.mark.asyncio
async def test_stops_when_unsubscribed() -> None:
    pubsub = FakePubSub(answers_ping=True)
    pubsub.subscribed = False
    got = await _drain(_bus(), pubsub, seconds=0.5)
    assert got == []
