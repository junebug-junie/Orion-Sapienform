"""Expectation steers attention (walkway spec idea 8).

docs/superpowers/specs/2026-09-22-walkway-camera-busy-world-design.md.

Contract: one Redis key per camera, ``orion:vision:expect:<stream_id>``, set by
orion-sql-writer's rhythm loop with a TTL equal to the open expectation window
(value is informational; only presence matters). While the key exists, the
router selects the ``triggered`` tier for that stream, so Orion looks harder
exactly when it expects something to happen.

``decide()`` is synchronous and runs under the dispatcher's state lock, so it
must never touch the network. A background task refreshes a cached set of
streams whose key exists every ``refresh_sec``; ``decide()`` reads the set.
Any Redis failure clears the set: an unreachable Redis means "no open
expectation", never "stuck triggered".
"""

from __future__ import annotations

import asyncio
from typing import Any, Iterable

from loguru import logger

EXPECT_KEY_PREFIX = "orion:vision:expect:"


def expect_key(stream_id: str) -> str:
    return f"{EXPECT_KEY_PREFIX}{stream_id}"


class ExpectationCache:
    def __init__(self, *, refresh_sec: float = 5.0) -> None:
        self.refresh_sec = max(0.5, float(refresh_sec))
        self._known_streams: set[str] = set()
        self._open: frozenset[str] = frozenset()
        self.refresh_failures = 0
        self.last_error: str | None = None

    # -- sync side (decide) -------------------------------------------------
    def note_stream(self, stream_id: str | None) -> None:
        if stream_id:
            self._known_streams.add(str(stream_id))

    def is_open(self, stream_id: str | None) -> bool:
        return bool(stream_id) and stream_id in self._open

    def open_streams(self) -> list[str]:
        return sorted(self._open)

    # -- async side (refresh loop) ------------------------------------------
    async def refresh_once(self, redis: Any, streams: Iterable[str] | None = None) -> frozenset[str]:
        names = sorted(set(streams) if streams is not None else self._known_streams)
        if not names:
            self._open = frozenset()
            return self._open
        try:
            pipe = redis.pipeline()
            for s in names:
                pipe.exists(expect_key(s))
            results = await pipe.execute()
            self._open = frozenset(s for s, r in zip(names, results) if int(r or 0) > 0)
            self.last_error = None
        except Exception as exc:
            self.refresh_failures += 1
            self.last_error = str(exc)
            self._open = frozenset()
            logger.warning(f"[ROUTER] expectation refresh failed (treated as no expectation): {exc}")
        return self._open

    async def run(self, redis: Any, stop: asyncio.Event) -> None:
        while not stop.is_set():
            await self.refresh_once(redis)
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.refresh_sec)
            except asyncio.TimeoutError:
                pass
