"""The equilibrium watch survives a Redis drop, and /ready tracks it.

2026-10-20: a bare subscribe loop ended on one redis error while the process stayed
up. The watch now resubscribes; equilibrium_subscriber_alive() reads False while it
backs off and True once it has resubscribed."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pytest

from app.equilibrium_watch import equilibrium_subscribed, watch_equilibrium
from orion.schemas.telemetry.system_health import EquilibriumSnapshotV1


class _Session:
    """Hands out queued messages, then reads as a quiet channel."""

    def __init__(self, messages: list[dict]) -> None:
        self._messages = list(messages)

    async def get_message(self, ignore_subscribe_messages=True, timeout=1.0):
        if self._messages:
            return self._messages.pop(0)
        await asyncio.sleep(0.01)
        return None


class _DropsFirstSubscribeBus:
    """The first subscribe raises like a dropped Redis; the second delivers `messages`."""

    def __init__(self, messages: list[dict]) -> None:
        self._messages = messages
        self.subscribe_calls = 0
        self.subscribe_args: list[tuple] = []

    def subscribe(self, *channels: str, patterns: bool = False):
        bus = self

        @asynccontextmanager
        async def _cm():
            bus.subscribe_calls += 1
            bus.subscribe_args.append((channels, patterns))
            if bus.subscribe_calls == 1:
                raise ConnectionError("Error 111 connecting to redis")
            yield _Session(bus._messages)

        return _cm()


async def _until(predicate, timeout=5.0):
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while not predicate():
        assert loop.time() < end, "condition not reached"
        await asyncio.sleep(0.02)


def _snapshot_msg() -> dict:
    snap = EquilibriumSnapshotV1(
        source_service="equilibrium",
        producer_boot_id="boot",
        generated_at=datetime.now(timezone.utc),
        grace_multiplier=3.0,
        windows_sec=[300],
        expected_services=[],
        services=[],
    )
    body = {"kind": "equilibrium.snapshot.v1", "payload": snap.model_dump(mode="json")}
    return {"type": "message", "data": json.dumps(body).encode()}


@pytest.mark.asyncio
async def test_watch_resubscribes_and_alive_follows_the_subscription() -> None:
    name = "mesh-guardian-equilibrium-test"
    queue: asyncio.Queue = asyncio.Queue(maxsize=8)
    stop = asyncio.Event()
    bus = _DropsFirstSubscribeBus([{"type": "message", "data": b"not json"}, _snapshot_msg()])
    task = asyncio.create_task(watch_equilibrium(bus, "orion:equilibrium:snapshot", queue, stop_event=stop, name=name))
    await _until(lambda: bus.subscribe_calls == 1)
    assert equilibrium_subscribed(name) is False  # backing off after the drop
    snapshot = await asyncio.wait_for(queue.get(), 5)
    assert isinstance(snapshot, EquilibriumSnapshotV1)
    assert bus.subscribe_calls == 2
    assert equilibrium_subscribed(name) is True
    stop.set()
    await asyncio.wait_for(task, 5)
    assert equilibrium_subscribed(name) is False
