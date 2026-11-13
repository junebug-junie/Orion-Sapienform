"""STT and TTS intake listeners must outlive a Redis drop (live 2026-10-20: one ConnectionError ended a bare loop)."""

from __future__ import annotations

import asyncio
import contextlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVICE_ROOT.parents[1]
for p in (str(REPO_ROOT), str(SERVICE_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

# stt_worker imports app.stt, which imports torch/whisper at module load (same stubs as test_stt_engine).
sys.modules.setdefault("whisper", MagicMock())
sys.modules.setdefault("torch", MagicMock())

from app import stt_worker, tts_worker  # noqa: E402


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


class _RecordingCodec:
    def __init__(self, ok: bool) -> None:
        self.ok = ok
        self.seen: list[object] = []

    def decode(self, data):
        self.seen.append(data)
        return SimpleNamespace(ok=self.ok, error="bad", envelope=SimpleNamespace(), raw=data)


def _bus(messages: list[dict], stop: asyncio.Event, codec: _RecordingCodec) -> _DroppingBus:
    bus = _DroppingBus([ConnectionError("bus down"), _Session(messages, stop)])
    bus.enabled = True
    bus.codec = codec
    return bus


def test_stt_listener_resubscribes_after_redis_drop() -> None:
    async def _run():
        stop = asyncio.Event()
        codec = _RecordingCodec(ok=False)
        bus = _bus([{"type": "message", "data": b"m1"}], stop, codec)
        await asyncio.wait_for(stt_worker.stt_listener_worker(bus, stop=stop), timeout=10)
        return bus, codec

    bus, codec = asyncio.run(_run())
    assert bus.subscribe_calls == 2
    assert codec.seen == [b"m1"]


def test_tts_listener_resubscribes_and_does_not_wait_on_synthesis(monkeypatch) -> None:
    started: list[object] = []
    release = asyncio.Event()

    async def _slow_process(bus, envelope, raw):
        started.append(raw)
        await release.wait()

    monkeypatch.setattr(tts_worker, "process_tts_request", _slow_process)

    async def _run():
        stop = asyncio.Event()
        codec = _RecordingCodec(ok=True)
        messages = [{"type": "message", "data": b"m1"}, {"type": "message", "data": b"m2"}]
        bus = _bus(messages, stop, codec)
        await asyncio.wait_for(tts_worker.listener_worker(bus, stop=stop), timeout=10)
        await asyncio.sleep(0)  # let the spawned tasks start
        release.set()
        return bus

    bus = asyncio.run(_run())
    assert bus.subscribe_calls == 2
    # Both syntheses started although the first never finished: the listener did not wait on it.
    assert started == [b"m1", b"m2"]
