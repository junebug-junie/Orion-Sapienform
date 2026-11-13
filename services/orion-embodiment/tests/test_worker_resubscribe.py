"""The intent consumer survives a Redis drop (2026-10-20: a bare subscribe loop
ended on one redis error while the process stayed up and looked healthy)."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from app.worker import EmbodimentWorker


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


@pytest.mark.asyncio
async def test_consume_loop_resubscribes_after_a_drop() -> None:
    seen: list[dict] = []
    w = EmbodimentWorker.__new__(EmbodimentWorker)
    w._stop = asyncio.Event()
    w._settings = SimpleNamespace(channel_intent="orion:embodiment:intent")
    w._bus = _DropsFirstSubscribeBus([{"type": "message", "data": "intent"}])

    async def handle(raw_msg: dict) -> None:
        seen.append(raw_msg)

    w._handle = handle
    task = asyncio.create_task(w._consume_loop())
    await _until(lambda: len(seen) == 1)
    assert w._bus.subscribe_calls == 2
    w._stop.set()
    await asyncio.wait_for(task, 5)
