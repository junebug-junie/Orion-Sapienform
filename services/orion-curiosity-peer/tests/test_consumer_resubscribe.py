"""Both peer consumers must outlive a Redis drop (live 2026-10-20: one ConnectionError ended a bare loop)."""

from __future__ import annotations

import asyncio
import contextlib

from app import worker
from app.settings import Settings


class _Session:
    """One pubsub session: hands out ``messages`` then sets ``stop``."""

    def __init__(self, messages: list[dict], stop: asyncio.Event) -> None:
        self._messages = list(messages)
        self._stop = stop

    async def get_message(self, ignore_subscribe_messages=True, timeout=1.0):
        if not self._messages:
            self._stop.set()
            return None
        return self._messages.pop(0)


class _DroppingBus:
    """First subscribe raises like a Redis drop; later ones hand out scripted sessions."""

    def __init__(self, sessions: list[object]) -> None:
        self.sessions = list(sessions)
        self.subscribe_calls = 0

    def subscribe(self, *channels: str, patterns: bool = False):
        bus = self

        @contextlib.asynccontextmanager
        async def _cm():
            bus.subscribe_calls += 1
            session = bus.sessions.pop(0)
            if isinstance(session, BaseException):
                raise session
            yield session

        return _cm()


def _run(consumer, monkeypatch, target: str) -> tuple[_DroppingBus, list]:
    handled: list = []
    monkeypatch.setattr(worker, target, lambda data, **kw: handled.append(data))

    async def _go() -> _DroppingBus:
        stop = asyncio.Event()
        msg = {"type": "message", "data": b'{"x": 1}'}
        bus = _DroppingBus([ConnectionError("bus down"), _Session([msg], stop)])
        await asyncio.wait_for(consumer(Settings(), bus, stop), timeout=10)
        return bus

    return asyncio.run(_go()), handled


def test_help_consumer_resubscribes_after_redis_drop(monkeypatch) -> None:
    bus, handled = _run(worker.run_help_consumer, monkeypatch, "handle_help_request")
    assert bus.subscribe_calls == 2
    assert handled == ['{"x": 1}']


def test_consumed_consumer_resubscribes_after_redis_drop(monkeypatch) -> None:
    bus, handled = _run(worker.run_consumed_consumer, monkeypatch, "apply_peer_brief_consumed")
    assert bus.subscribe_calls == 2
    assert handled == ['{"x": 1}']
