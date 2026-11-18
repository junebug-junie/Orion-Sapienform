"""The PSU command listener survives a Redis drop (2026-10-20: a bare subscribe
loop ended on one redis error while the process stayed up and looked healthy)."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from app import bus as psu_bus


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


class _Decoded:
    def __init__(self, data: dict) -> None:
        self.ok = True
        self.envelope = SimpleNamespace(payload=data)


@pytest.mark.asyncio
async def test_psu_listener_resubscribes_after_a_drop() -> None:
    calls: list[tuple] = []
    bus = _DropsFirstSubscribeBus(
        [
            {"type": "message", "data": {"action": "explode"}},
            {"type": "message", "data": {"action": "cycle", "mode": 2}},
        ]
    )
    bus.codec = SimpleNamespace(decode=_Decoded)
    bus.enabled = True

    async def handle_command(action, mode=None):
        calls.append((action, mode))

    service = SimpleNamespace(bus=bus, bus_enabled=True, handle_command=handle_command)
    task = asyncio.create_task(psu_bus._command_listener_async(service))
    await _until(lambda: len(calls) == 1)
    assert calls == [("cycle", 2)]  # unknown actions are still refused
    assert bus.subscribe_calls == 2
    task.cancel()
    await asyncio.wait_for(task, 5)  # the listener logs the cancel and returns
