"""run_consumer must outlive a Redis drop (live 2026-10-20: one ConnectionError ended a bare loop)."""

from __future__ import annotations

import asyncio
import contextlib
import json

from app import main as room_main
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


def test_run_consumer_resubscribes_after_redis_drop(monkeypatch) -> None:
    handled: list[dict] = []

    async def _handle_message(settings, bus, raw):
        handled.append(raw)

    monkeypatch.setattr(room_main, "handle_message", _handle_message)

    async def _run() -> _DroppingBus:
        stop = asyncio.Event()
        msg = {"type": "message", "data": json.dumps({"request_id": "r1"}).encode()}
        bus = _DroppingBus([ConnectionError("bus down"), _Session([msg], stop)])
        await asyncio.wait_for(room_main.run_consumer(Settings(), bus, stop), timeout=10)
        return bus

    bus = asyncio.run(_run())
    assert bus.subscribe_calls == 2
    assert handled == [{"request_id": "r1"}]
