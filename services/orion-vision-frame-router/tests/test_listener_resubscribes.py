"""Regression: orion-vision-frame-router's bus listeners survive a Redis drop.

Live 2026-10-20: a bare ``async with bus.subscribe(...)`` loop died on one redis
ConnectionError/TimeoutError and the process stayed up looking healthy. Each
listener now runs under ``orion.core.bus.resilience.run_subscription``. The fake
bus below fails the first subscribe, drops the second session mid-stream, and
only the third session works; the listener must still see both messages.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
from types import SimpleNamespace

from orion.core.bus.resilience import run_subscription as _real_run_subscription


class _RedisTimeout(Exception):
    """Stands in for redis.exceptions.TimeoutError, which is not asyncio's."""


class _PubSub:
    def __init__(self, script):
        self._script = list(script)

    async def get_message(self, ignore_subscribe_messages=True, timeout=1.0):
        if not self._script:
            await asyncio.sleep(0.01)  # a quiet channel
            return None
        item = self._script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class _Codec:
    def __init__(self):
        self.decoded = []

    def decode(self, data):
        self.decoded.append(data)
        return SimpleNamespace(ok=True, envelope=SimpleNamespace(data=data, payload={}), error=None)


class _FlakyBus:
    """First subscribe fails, the second session drops mid-stream, the third works."""

    def __init__(self, first, second):
        self.codec = _Codec()
        self.subscribe_calls = []
        self._sessions = [
            ConnectionError("Error connecting to redis"),
            _PubSub([{"type": "message", "data": first}, _RedisTimeout("Timeout reading from socket")]),
            _PubSub([{"type": "message", "data": second}]),
        ]

    def subscribe(self, *channels, patterns=False):
        bus = self

        @contextlib.asynccontextmanager
        async def _cm():
            bus.subscribe_calls.append((channels, patterns))
            session = bus._sessions.pop(0)
            if isinstance(session, BaseException):
                raise session
            yield session

        return _cm()


async def _until(cond, timeout=5.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not cond():
        assert loop.time() < deadline, "listener never got the second message"
        await asyncio.sleep(0.01)


def _fast_backoff(monkeypatch, module):
    monkeypatch.setattr(
        module, "run_subscription", functools.partial(_real_run_subscription, max_backoff_sec=0.01)
    )


import app.main as router_main
from app.main import FrameRouterService
from app.settings import Settings


def test_frame_and_reply_listeners_resubscribe_after_redis_errors(monkeypatch):
    _fast_backoff(monkeypatch, router_main)

    async def body():
        settings = Settings()
        for loop_name, handler_name, channel, patterns in (
            ("_frames_loop", "handle_frame_envelope", settings.CHANNEL_FRAMES_IN, False),
            ("_reply_loop", "handle_reply_envelope", f"{settings.CHANNEL_REPLY_PREFIX}:*", True),
        ):
            bus = _FlakyBus("a", "b")
            svc = FrameRouterService(settings=settings, bus=bus)
            seen = []

            async def handle(env, seen=seen):
                seen.append(env.data)

            svc.dispatcher = SimpleNamespace(**{handler_name: handle})
            task = asyncio.create_task(getattr(svc, loop_name)())
            await _until(lambda: seen == ["a", "b"])
            svc._shutdown.set()
            await asyncio.wait_for(task, 5)
            assert len(bus.subscribe_calls) == 3, loop_name
            assert bus.subscribe_calls == [((channel,), patterns)] * 3, loop_name

    asyncio.run(body())
