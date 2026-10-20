"""The VoIP command listener must outlive a Redis drop.

Live 2026-10-20: a bare subscribe loop died on one Redis ConnectionError and the
process stayed up looking healthy. Here that meant the endpoint kept taking calls
while bus commands were silently ignored.
"""

from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace

from app.bus_integration import bus_listener_loop


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


def test_bus_listener_resubscribes_after_redis_drop() -> None:
    calls: list[str] = []
    published: list[tuple] = []

    async def _run() -> _DroppingBus:
        stop = asyncio.Event()
        messages = [
            {"type": "message", "data": b'{"action": "ring"}'},
            {"type": "message", "data": b'{"action": "nope"}'},
        ]
        bus = _DroppingBus([ConnectionError("bus down"), _Session(messages, stop)])
        settings = SimpleNamespace(bus_command_channel="orion:voip:cmd", bus_redis_url="redis://x")
        await asyncio.wait_for(
            bus_listener_loop(
                bus,
                settings,
                lambda event, **extra: published.append((event, extra.get("action"))),
                {"ring": lambda: calls.append("ring")},
                stop=stop,
            ),
            timeout=10,
        )
        return bus

    bus = asyncio.run(_run())
    assert bus.subscribe_calls == 2
    assert calls == ["ring"]
    assert published == [("unknown_command", "nope")]
