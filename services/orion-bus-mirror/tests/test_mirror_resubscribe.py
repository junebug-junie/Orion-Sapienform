"""The mirror survives a Redis drop in-process instead of exiting.

2026-10-20: a bare subscribe loop ends on one redis error. The mirror used to exit
and rely on a container restart (the 2026-10-02 crash loop restarted it 581 times);
it now resubscribes through run_subscription and keeps writing rows."""

from __future__ import annotations

import asyncio
import sqlite3
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest

from app.main import mirror_bus
from app.settings import settings
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.codec import OrionCodec
from orion.core.bus.bus_service_chassis import HeartbeatOnly


class _Session:
    """Hands out queued messages, then reads as a quiet channel."""

    def __init__(self, messages: list[dict]) -> None:
        self._messages = list(messages)

    async def get_message(self, ignore_subscribe_messages=True, timeout=1.0):
        if self._messages:
            return self._messages.pop(0)
        await asyncio.sleep(0.01)
        return None


class _DropsFirstSubscribeBus:
    """The first subscribe raises like a dropped Redis; the second delivers `messages`."""

    def __init__(self, messages: list[dict]) -> None:
        self._messages = messages
        self.subscribe_calls = 0
        self.subscribe_args: list[tuple] = []

    def subscribe(self, *channels: str, patterns: bool = False):
        bus = self

        @asynccontextmanager
        async def _cm():
            bus.subscribe_calls += 1
            bus.subscribe_args.append((channels, patterns))
            if bus.subscribe_calls == 1:
                raise ConnectionError("Error 111 connecting to redis")
            yield _Session(bus._messages)

        return _cm()


async def _until(predicate, timeout=5.0):
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while not predicate():
        assert loop.time() < end, "condition not reached"
        await asyncio.sleep(0.02)


@pytest.mark.asyncio
async def test_mirror_resubscribes_after_a_drop_and_keeps_writing(tmp_path, monkeypatch) -> None:
    codec = OrionCodec()
    env = BaseEnvelope(kind="test.event.v1", source=ServiceRef(name="t"), payload={"n": 1})
    bus = _DropsFirstSubscribeBus(
        [{"type": "pmessage", "pattern": b"orion:*", "channel": b"orion:test", "data": codec.encode(env)}]
    )
    bus.codec = codec
    bus.connect = AsyncMock()
    bus.close = AsyncMock()
    db = tmp_path / "mirror.sqlite"
    monkeypatch.setattr(settings, "MIRROR_PARQUET_DIR", str(tmp_path / "parquet"))
    monkeypatch.setattr(settings, "MIRROR_SQLITE_PATH", str(db))
    monkeypatch.setattr(settings, "MIRROR_GRAPH_ENABLED", False)

    def rows() -> list[tuple]:
        with sqlite3.connect(db) as conn:
            try:
                return conn.execute("SELECT channel FROM bus_events").fetchall()
            except sqlite3.OperationalError:
                return []

    with patch("app.main.OrionBusAsync", return_value=bus), patch(
        "app.main.build_heartbeat_chassis", return_value=AsyncMock(spec=HeartbeatOnly)
    ):
        task = asyncio.create_task(mirror_bus())
        await _until(lambda: rows() == [("orion:test",)])
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert bus.subscribe_calls == 2
    assert bus.subscribe_args[-1] == ((settings.MIRROR_PATTERN,), True)
    bus.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_mirror_exits_for_a_restart_when_writes_keep_failing(tmp_path, monkeypatch) -> None:
    """A broken store must not leave a mirror that heartbeats while writing nothing."""

    class _BrokenCodec:
        def decode(self, data):
            raise OSError("database or disk is full")

    bus = _DropsFirstSubscribeBus([{"type": "pmessage", "channel": b"orion:x", "data": b"{}"}] * 5)
    bus.codec = _BrokenCodec()
    bus.connect = AsyncMock()
    bus.close = AsyncMock()
    monkeypatch.setattr("app.main.MIRROR_MAX_CONSECUTIVE_WRITE_FAILURES", 3)
    monkeypatch.setattr(settings, "MIRROR_PARQUET_DIR", str(tmp_path / "parquet"))
    monkeypatch.setattr(settings, "MIRROR_SQLITE_PATH", str(tmp_path / "mirror.sqlite"))
    monkeypatch.setattr(settings, "MIRROR_GRAPH_ENABLED", False)

    with patch("app.main.OrionBusAsync", return_value=bus), patch(
        "app.main.build_heartbeat_chassis", return_value=AsyncMock(spec=HeartbeatOnly)
    ):
        with pytest.raises(SystemExit):
            await asyncio.wait_for(mirror_bus(), 10)
    bus.close.assert_awaited_once()
