"""Every pub/sub read loop must notice a socket the server already dropped.

#3033 fixed iter_messages. A silently dead line answers "nothing yet" on time, so the
read-overrun guards in the RPC reply worker and run_subscription never fired on it
(an RPC worker on a dead line times out every call that process makes until restart).
These tests drive each loop with a pub/sub that never answers PING.
"""
from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock, patch

import pytest

from orion.core.bus import async_service
from orion.core.bus.async_service import OrionBusAsync, PubSubSilentError, PubSubWatchdog


@pytest.fixture
def fast_watchdog(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(async_service, "PUBSUB_IDLE_PING_SEC", 0.05)
    monkeypatch.setattr(async_service, "PUBSUB_PONG_DEADLINE_SEC", 0.05)


class SilentPubSub:
    """Half-open line: every read returns None on time; PING gets no pong unless ``alive``."""

    def __init__(self, *, alive: bool = False, messages: list[dict] | None = None) -> None:
        self.alive = alive
        self.queue: list[dict] = list(messages or [])
        self.pings = 0
        self.connection = object()
        self.subscribed: set[str] = set()
        self.closed = False

    async def get_message(self, ignore_subscribe_messages: bool = False, timeout: float | None = 0.0):
        if self.queue:
            return self.queue.pop(0)
        await asyncio.sleep(min(timeout or 0, 0.01))
        return None

    async def ping(self) -> None:
        self.pings += 1
        if self.alive:
            self.queue.append({"type": "pong", "pattern": None, "channel": None, "data": b""})

    async def subscribe(self, *channels: str) -> None:
        self.subscribed.update(channels)

    async def close(self) -> None:
        self.closed = True


# --- the helper itself -------------------------------------------------------------


@pytest.mark.asyncio
async def test_watchdog_pings_after_quiet_then_raises_without_pong() -> None:
    dog = PubSubWatchdog(idle_ping_sec=0.02, pong_deadline_sec=0.02)
    line = SilentPubSub()
    assert await dog.idle(line) is None and line.pings == 0  # not quiet long enough yet
    await asyncio.sleep(0.03)
    assert await dog.idle(line) is None and line.pings == 1  # quiet: PING sent
    await asyncio.sleep(0.03)
    with pytest.raises(PubSubSilentError, match="pubsub silent"):
        await dog.idle(line)


@pytest.mark.asyncio
async def test_watchdog_returns_a_late_reply_instead_of_declaring_dead() -> None:
    dog = PubSubWatchdog(idle_ping_sec=0.0, pong_deadline_sec=0.0)
    line = SilentPubSub(alive=True)
    assert await dog.idle(line) is None  # PING; pong now buffered
    late = await dog.idle(line)  # deadline passed, drain finds the pong
    assert late is not None and late["type"] == "pong"


@pytest.mark.asyncio
async def test_watchdog_heard_cancels_the_pending_ping() -> None:
    dog = PubSubWatchdog(idle_ping_sec=0.0, pong_deadline_sec=0.0)
    line = SilentPubSub()
    await dog.idle(line)  # PING sent
    dog.heard()  # traffic arrived
    assert await dog.idle(line) is None  # new PING, not a death verdict
    assert line.pings == 2


# --- run_subscription (resilience.py) ------------------------------------------------


class _Bus:
    def __init__(self, sessions: list[SilentPubSub]) -> None:
        self.sessions = list(sessions)
        self.subscribe_calls = 0

    def subscribe(self, *channels: str, patterns: bool = False):
        bus = self

        @contextlib.asynccontextmanager
        async def _cm():
            bus.subscribe_calls += 1
            yield bus.sessions.pop(0)

        return _cm()


@pytest.mark.asyncio
async def test_run_subscription_reopens_a_silently_dead_session(fast_watchdog) -> None:
    from orion.core.bus.resilience import run_subscription, subscription_health

    stop = asyncio.Event()
    seen: list[dict] = []
    dead = SilentPubSub()
    live = SilentPubSub(alive=True, messages=[{"type": "message", "data": 1}])
    bus = _Bus([dead, live])

    def handle(msg: dict) -> None:
        seen.append(msg)
        stop.set()

    await asyncio.wait_for(
        run_subscription(bus, "orion:test", handle, stop_event=stop, name="dead-line", max_backoff_sec=0.01),
        timeout=5,
    )
    assert dead.pings == 1
    assert bus.subscribe_calls == 2
    assert seen == [{"type": "message", "data": 1}]
    row = next(r for r in subscription_health()["listeners"] if r["name"] == "dead-line")
    assert row["failures_total"] == 1 and "pubsub silent" in row["last_failure"]


@pytest.mark.asyncio
async def test_run_subscription_keeps_a_quiet_live_session(fast_watchdog) -> None:
    from orion.core.bus.resilience import run_subscription

    stop = asyncio.Event()
    line = SilentPubSub(alive=True)
    bus = _Bus([line])
    task = asyncio.create_task(
        run_subscription(bus, "orion:test", lambda m: None, stop_event=stop, name="quiet-line")
    )
    await asyncio.sleep(0.5)
    stop.set()
    await asyncio.wait_for(task, timeout=3)
    assert line.pings >= 2  # kept pinging across several quiet windows
    assert bus.subscribe_calls == 1  # and never reopened


# --- the shared RPC reply worker (async_service._run_rpc_only) -----------------------


class _Redis:
    def __init__(self) -> None:
        self.made: list[SilentPubSub] = []

    def pubsub(self) -> SilentPubSub:
        # First connection is dead; every reconnect gets a live one.
        line = SilentPubSub(alive=bool(self.made))
        self.made.append(line)
        return line

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_rpc_worker_reconnects_a_silently_dead_reply_line(fast_watchdog) -> None:
    bus = OrionBusAsync("redis://127.0.0.1:6379/0", enabled=True)
    bus.connect = AsyncMock()
    redis = _Redis()
    with patch.object(bus, "_create_pubsub_redis", return_value=redis):
        bus._rpc_subscribed.add("orion:test:reply:a")
        bus.start_rpc_worker()
        try:
            await asyncio.sleep(0.6)
            assert len(redis.made) == 2, "worker never left the dead reply line"
            dead, live = redis.made
            assert dead.pings == 1 and dead.closed
            assert "orion:test:reply:a" in live.subscribed  # replies keep flowing after reconnect
            assert bus._rpc_worker_task is not None and not bus._rpc_worker_task.done()
        finally:
            await bus.close()
