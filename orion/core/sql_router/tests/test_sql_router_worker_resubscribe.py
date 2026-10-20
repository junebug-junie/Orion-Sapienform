from __future__ import annotations

import asyncio
import contextlib
import json
import threading

from orion.core.sql_router import worker as worker_mod


class _ScriptedPubSub:
    """Returns scripted get_message results; exceptions are raised. Idles once empty."""

    def __init__(self, script: list[object]) -> None:
        self._script = list(script)

    async def get_message(self, ignore_subscribe_messages=True, timeout=1.0):
        if not self._script:
            await asyncio.sleep(0.05)
            return None
        item = self._script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class _ScriptedBus:
    def __init__(self, sessions: list[object]) -> None:
        self.sessions = list(sessions)
        self.subscribe_calls = 0
        self.published: list[tuple[str, object]] = []

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def publish(self, channel: str, msg: object) -> None:
        self.published.append((channel, msg))

    def subscribe(self, *channels: str):
        bus = self

        @contextlib.asynccontextmanager
        async def _cm():
            bus.subscribe_calls += 1
            session = bus.sessions.pop(0)
            if isinstance(session, BaseException):
                raise session
            yield session

        return _cm()


class _Model:
    __name__ = "_Model"


def test_worker_resubscribes_after_the_subscription_drops(monkeypatch) -> None:
    """Live 2026-10-20: one redis error ended a bare subscribe loop while the process looked healthy."""
    msg = lambda n: {"type": "message", "data": json.dumps({"n": n}).encode()}  # noqa: E731
    first = _ScriptedPubSub([msg(1), ConnectionError("Connection reset by peer")])
    bus = _ScriptedBus([first, TimeoutError("Timeout connecting to server"), _ScriptedPubSub([msg(2)])])
    monkeypatch.setattr(worker_mod, "OrionBusAsync", lambda url: bus)
    monkeypatch.setattr(worker_mod, "SessionLocal", contextlib.nullcontext)

    rows: list[dict] = []
    two = threading.Event()

    def writer(model_class, payload, db):
        rows.append(payload)
        if len(rows) == 2:
            two.set()
        return None

    real_sleep = asyncio.sleep

    async def fast_sleep(delay, *a, **k):  # keep the 1 s backoff out of the test
        return await real_sleep(min(delay, 0.01), *a, **k)

    monkeypatch.setattr(worker_mod.asyncio, "sleep", fast_sleep)
    worker_mod.start_worker_thread("redis://unused", "orion:test:rows", _Model, writer_func=writer,
                                   publish_channel="orion:test:out")
    assert two.wait(timeout=10)
    assert rows == [{"n": 1}, {"n": 2}]
    assert bus.subscribe_calls == 3
    assert bus.published == [("orion:test:out", {"n": 1}), ("orion:test:out", {"n": 2})]
