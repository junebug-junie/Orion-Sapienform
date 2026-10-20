"""The social-turn consumer survives a Redis drop (2026-10-20: a bare subscribe
loop ended on one redis error while the process stayed up and looked healthy)."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from app.service import SocialMemoryService


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
        self.envelope = SimpleNamespace(kind=data["kind"], payload=data["payload"], correlation_id="c")


@pytest.mark.asyncio
async def test_consume_social_turns_resubscribes_after_a_drop() -> None:
    seen: list[dict] = []
    bus = _DropsFirstSubscribeBus(
        [
            {"type": "message", "data": {"kind": "other.kind", "payload": {"n": 0}}},
            {"type": "message", "data": {"kind": "social.turn.stored.v1", "payload": {"n": 1}}},
        ]
    )
    bus.codec = SimpleNamespace(decode=_Decoded)
    service = SocialMemoryService.__new__(SocialMemoryService)
    service.bus = bus
    service.settings = SimpleNamespace(social_memory_input_channel="orion:social:turn:stored")

    async def process(payload: dict) -> None:
        seen.append(payload)

    service.process_social_turn = process
    task = asyncio.create_task(service._consume_social_turns())
    await _until(lambda: len(seen) == 1)
    assert seen == [{"n": 1}]  # the kind filter still applies
    assert bus.subscribe_calls == 2
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
