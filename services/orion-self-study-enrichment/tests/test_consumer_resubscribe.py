"""run_consumer must outlive a Redis drop (live 2026-10-20: one ConnectionError ended a bare loop)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app import main as enrichment_main  # noqa: E402
from app.settings import Settings  # noqa: E402


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
    monkeypatch.setattr(
        enrichment_main, "handle_request_payload", lambda settings, payload: handled.append(payload)
    )

    async def _run() -> _DroppingBus:
        stop = asyncio.Event()
        msg = {"type": "message", "data": json.dumps({"touched_paths": ["a.py"]}).encode()}
        bus = _DroppingBus([ConnectionError("bus down"), _Session([msg], stop)])
        await asyncio.wait_for(enrichment_main.run_consumer(Settings(), bus, stop), timeout=10)
        return bus

    bus = asyncio.run(_run())
    assert bus.subscribe_calls == 2
    assert handled == [{"touched_paths": ["a.py"]}]
