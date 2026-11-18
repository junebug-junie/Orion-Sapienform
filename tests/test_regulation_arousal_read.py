"""The decision-point arousal fetch: anything doubtful is unknown, and it never raises."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from orion.regulation.arousal_read import read_arousal_async, read_arousal_sync
from orion.schemas.regulation import REGULATION_STATE_REDIS_KEY, ArousalReadingV1, RegulationStateV1

NOW = datetime(2026, 10, 13, 15, 0, tzinfo=timezone.utc)


def state_json(level: str, *, age_sec: float = 30.0) -> str:
    at = NOW - timedelta(seconds=age_sec)
    reading = ArousalReadingV1(arousal_level=level, since=at, observed_at=at, reasons=[f"t_{level}"])
    return RegulationStateV1(generated_at=at, arousal=reading).model_dump_json()


class SyncClient:
    def __init__(self, value=None, exc=None):
        self.value, self.exc, self.keys = value, exc, []

    def get(self, key):
        self.keys.append(key)
        if self.exc:
            raise self.exc
        return self.value


def test_sync_reads_the_regulation_key_and_judges_it():
    c = SyncClient(state_json("strained"))
    v = read_arousal_sync(None, enabled=True, reader="t", now=NOW, client=c)
    assert v.level == "strained" and c.keys == [REGULATION_STATE_REDIS_KEY]


def test_sync_disabled_never_touches_redis():
    c = SyncClient(state_json("strained"))
    v = read_arousal_sync("redis://x", enabled=False, reader="t", now=NOW, client=c)
    assert v.level == "unknown" and v.reason == "disabled" and c.keys == []


def test_sync_error_stale_absent_and_no_url_are_unknown():
    assert read_arousal_sync(None, enabled=True, reader="t", now=NOW,
                             client=SyncClient(exc=ConnectionError("down"))).level == "unknown"
    assert read_arousal_sync(None, enabled=True, reader="t", now=NOW,
                             client=SyncClient(state_json("strained", age_sec=400))).level == "unknown"
    assert read_arousal_sync(None, enabled=True, reader="t", now=NOW, client=SyncClient(None)).level == "unknown"
    assert read_arousal_sync("", enabled=True, reader="t", now=NOW).reason == "no_redis_url"


def test_sync_unreachable_real_url_is_unknown_not_raise():
    v = read_arousal_sync("redis://127.0.0.1:1/0", enabled=True, reader="t", now=NOW, timeout_sec=0.2)
    assert v.level == "unknown" and v.reason == "read_failed"


class AsyncRedis:
    def __init__(self, value=None, hang=False):
        self.value, self.hang = value, hang

    async def get(self, key):
        if self.hang:
            await asyncio.sleep(10)
        return self.value


class Bus:
    def __init__(self, redis):
        self._r = redis

    @property
    def redis(self):
        if self._r is None:
            raise RuntimeError("OrionBusAsync not connected")
        return self._r


def test_async_reads_level():
    v = asyncio.run(read_arousal_async(Bus(AsyncRedis(state_json("idle"))), enabled=True, reader="t", now=NOW))
    assert v.level == "idle"


def test_async_unconnected_hang_and_disabled_are_unknown():
    assert asyncio.run(read_arousal_async(Bus(None), enabled=True, reader="t", now=NOW)).level == "unknown"
    hung = asyncio.run(read_arousal_async(Bus(AsyncRedis(hang=True)), enabled=True, reader="t",
                                          now=NOW, timeout_sec=0.05))
    assert hung.level == "unknown" and hung.reason == "read_failed"
    off = asyncio.run(read_arousal_async(Bus(AsyncRedis(state_json("strained"))), enabled=False,
                                         reader="t", now=NOW))
    assert off.reason == "disabled"
