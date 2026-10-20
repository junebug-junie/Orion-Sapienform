"""Regression: orion-vision-edge's bus listeners survive a Redis drop.

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


import importlib
import sys


def test_detector_loop_resubscribes_after_redis_errors(monkeypatch):
    bus = _FlakyBus("a", "b")
    bus.enabled = True
    context = SimpleNamespace(
        settings=SimpleNamespace(
            EDGE_ACTIVITY_MIN_INTERVAL_S=1.0, CHANNEL_VISION_FRAMES="orion:vision:frames"
        ),
        bus=bus,
        detectors=[],
    )
    # app.utils reads settings at import, and SOURCE has no default.
    monkeypatch.setenv("SOURCE", "rtsp://test-camera")
    edge_settings = importlib.import_module("app.settings")
    edge_settings.get_settings.cache_clear()
    # app.context opens the camera and loads detectors at import; stand it in.
    monkeypatch.setitem(sys.modules, "app.context", context)
    monkeypatch.delitem(sys.modules, "app.detector_worker", raising=False)
    worker = importlib.import_module("app.detector_worker")
    _fast_backoff(monkeypatch, worker)

    async def body():
        # The detector loop has no stop event; it runs until cancelled.
        task = asyncio.create_task(worker.run_detector_loop())
        await _until(lambda: bus.codec.decoded == ["a", "b"])
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert len(bus.subscribe_calls) == 3

    try:
        asyncio.run(body())
    finally:
        sys.modules.pop("app.detector_worker", None)
        edge_settings.get_settings.cache_clear()
