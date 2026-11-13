"""The RAG request listener must outlive a Redis drop.

Live 2026-10-20: a bare subscribe loop died on one Redis ConnectionError and the
process stayed up looking healthy. This service is the first test in its own
tests/ directory; heavy deps (chromadb, sentence-transformers) are kept out by
stubbing app.processor, which is all the listener imports from them.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import threading
import types
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

# app.settings builds Settings() at import and every field below is required.
for _key, _value in {
    "SERVICE_NAME": "orion-rag",
    "SERVICE_VERSION": "test",
    "PORT": "0",
    "ORION_BUS_ENABLED": "true",
    "ORION_BUS_URL": "redis://unused:6379/0",
    "SUBSCRIBE_CHANNEL_RAG_REQUEST": "orion:rag:request",
    "PUBLISH_CHANNEL_BRAIN_INTAKE": "orion:brain:intake",
    "VECTOR_DB_URL": "http://unused:8000",
    "VECTOR_DB_COLLECTION": "test",
    "EMBEDDING_MODEL": "test",
}.items():
    os.environ.setdefault(_key, _value)

_processor = types.ModuleType("app.processor")
_processor.process_rag_request = lambda data: None
sys.modules.setdefault("app.processor", _processor)

from app import listener  # noqa: E402


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

    async def close(self) -> None:
        self.closed = True


def test_rag_listener_resubscribes_after_redis_drop(monkeypatch) -> None:
    handled: list[dict] = []
    done = threading.Event()

    def _process(data: dict) -> None:
        handled.append(data)
        done.set()

    monkeypatch.setattr(listener, "process_rag_request", _process)

    async def _run() -> _DroppingBus:
        stop = asyncio.Event()
        msg = {"type": "message", "data": b'{"query": "what is orion"}'}
        bus = _DroppingBus([ConnectionError("bus down"), _Session([msg], stop)])
        monkeypatch.setattr(listener, "OrionBusAsync", lambda **kw: bus)
        await asyncio.wait_for(listener._listener_worker_async(stop), timeout=10)
        return bus

    bus = asyncio.run(_run())
    assert done.wait(5)
    assert bus.subscribe_calls == 2
    assert handled == [{"query": "what is orion"}]
    assert bus.closed
