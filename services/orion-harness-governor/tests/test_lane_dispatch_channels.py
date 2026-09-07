"""Two independent dispatch loops, one per compute lane, same handler code.

Confirmed live 2026-09-07: `run_bus_worker` was a single loop that processed
`orion:harness:run:request` messages one at a time, and both chat turns and
agent-lane turns (curiosity, Mode=Agent+Compute=Agent) published to that same
channel. A single 40-minute agent-lane run left a real chat turn waiting the
entire time. This split gives each lane its own channel and its own loop;
these tests pin (1) the settings default for the new channel, (2) that a
worker only ever reacts to the channel it was given, and (3) the actual
property the incident needed: one lane's slow turn does not delay the other's.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest

from app import bus_listener
from app.settings import HarnessGovernorSettings


def test_agent_lane_channel_default_and_distinct_from_chat_channel(monkeypatch) -> None:
    monkeypatch.delenv("CHANNEL_HARNESS_RUN_REQUEST_AGENT", raising=False)
    settings = HarnessGovernorSettings()
    assert settings.channel_harness_run_request_agent == "orion:harness:run:request:agent"
    assert settings.channel_harness_run_request_agent != settings.channel_harness_run_request


class _FakePubSub:
    """One message, then silence. `get_message` blocks out `timeout` on every
    later call so `run_bus_worker`'s loop idles realistically instead of
    busy-spinning, matching the real `redis.asyncio` PubSub contract closely
    enough for this test."""

    def __init__(self, channel: str, queue: "asyncio.Queue[dict] | None") -> None:
        self._channel = channel
        self._queue = queue

    async def get_message(self, ignore_subscribe_messages: bool = True, timeout: float = 0.0) -> dict | None:
        if self._queue is None:
            await asyncio.sleep(timeout)
            return None
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None


class _FakeBus:
    """Stands in for OrionBusAsync: one queue per channel, so a worker
    subscribed to channel A never sees a message queued for channel B."""

    def __init__(self, queues: dict[str, "asyncio.Queue[dict]"]) -> None:
        self._queues = queues
        self.codec = _PassthroughCodec()

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    @asynccontextmanager
    async def subscribe(self, channel: str):
        yield _FakePubSub(channel, self._queues.get(channel))


class _DecodedEnvelope:
    def __init__(self, payload: dict) -> None:
        self.ok = True
        self.error = None
        self.envelope = _Envelope(payload)


class _Envelope:
    def __init__(self, payload: dict) -> None:
        self.payload = payload


class _PassthroughCodec:
    """The messages this test enqueues already carry their intended payload
    directly (no real wire encoding needed) -- only `_handle_bus_message`'s
    caller (`run_bus_worker`) is under test here, not the codec."""

    def decode(self, data):  # noqa: ANN001 - test double
        return _DecodedEnvelope(data)


@pytest.mark.asyncio
async def test_worker_only_reacts_to_its_own_channel(monkeypatch) -> None:
    chat_queue: asyncio.Queue = asyncio.Queue()
    agent_queue: asyncio.Queue = asyncio.Queue()
    bus = _FakeBus({"chat:channel": chat_queue, "agent:channel": agent_queue})
    monkeypatch.setattr(bus_listener, "OrionBusAsync", lambda url: bus)
    monkeypatch.setattr(bus_listener.settings, "orion_bus_enabled", True)
    monkeypatch.setattr(bus_listener.settings, "orion_harness_governor_enabled", True)

    seen: list[tuple[str, dict]] = []

    async def _record(_bus, msg) -> None:
        seen.append((msg["lane_marker"], msg))

    monkeypatch.setattr(bus_listener, "_handle_bus_message", _record)

    await agent_queue.put({"type": "message", "lane_marker": "agent"})
    stop_event = asyncio.Event()

    async def _stop_soon() -> None:
        await asyncio.sleep(0.1)
        stop_event.set()

    await asyncio.gather(
        bus_listener.run_bus_worker("chat:channel", stop_event, lane="chat"),
        _stop_soon(),
    )

    assert seen == [], "a message queued for the agent channel must never reach the chat-channel worker"


@pytest.mark.asyncio
async def test_a_slow_agent_lane_turn_does_not_delay_the_chat_lane(monkeypatch) -> None:
    """The actual property the 2026-09-07 incident needed: two independent
    loops, so a long-running agent-lane turn cannot make a chat-lane turn
    wait behind it. Regression test for exactly that failure mode."""
    chat_queue: asyncio.Queue = asyncio.Queue()
    agent_queue: asyncio.Queue = asyncio.Queue()
    bus = _FakeBus({"chat:channel": chat_queue, "agent:channel": agent_queue})
    monkeypatch.setattr(bus_listener, "OrionBusAsync", lambda url: bus)
    monkeypatch.setattr(bus_listener.settings, "orion_bus_enabled", True)
    monkeypatch.setattr(bus_listener.settings, "orion_harness_governor_enabled", True)

    order: list[str] = []

    async def _handle(_bus, msg) -> None:
        lane = msg["lane_marker"]
        if lane == "agent":
            # Stands in for curiosity's real 20-40 minute turn.
            await asyncio.sleep(0.3)
        order.append(lane)

    monkeypatch.setattr(bus_listener, "_handle_bus_message", _handle)

    await agent_queue.put({"type": "message", "lane_marker": "agent"})
    await asyncio.sleep(0.05)  # let the agent-lane worker pick it up first
    await chat_queue.put({"type": "message", "lane_marker": "chat"})

    stop_event = asyncio.Event()

    async def _stop_soon() -> None:
        await asyncio.sleep(0.5)
        stop_event.set()

    await asyncio.gather(
        bus_listener.run_bus_worker("chat:channel", stop_event, lane="chat"),
        bus_listener.run_bus_worker("agent:channel", stop_event, lane="agent"),
        _stop_soon(),
    )

    # Chat finishes BEFORE the still-running agent-lane turn, even though the
    # agent-lane message arrived and started being handled first. On the old
    # single shared loop this would be impossible -- chat could not even be
    # read until the agent-lane turn's 0.3s "run" returned.
    assert order == ["chat", "agent"]
