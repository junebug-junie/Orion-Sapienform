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

Experiment override (2026-10-29, vision focus experiment): the same refresh
also reads ``orion:vision:experiment:<stream_id>``, written by
orion-prediction's switchback runner (services/orion-prediction/app/
experiment_runner.py). Value ``on`` with a TTL forces the triggered tier
(``triggered_by="experiment"``). Absent, any other value, a key with no TTL
(a hand-set key that would never expire), or a Redis failure all mean "no
override": the experiment can only add focus, never remove perception, and
a dead runner cannot leave it stuck on.
"""

from __future__ import annotations

import asyncio
from typing import Any, Iterable

from loguru import logger

EXPECT_KEY_PREFIX = "orion:vision:expect:"
EXPERIMENT_KEY_PREFIX = "orion:vision:experiment:"


def expect_key(stream_id: str) -> str:
    return f"{EXPECT_KEY_PREFIX}{stream_id}"


def experiment_key(stream_id: str) -> str:
    return f"{EXPERIMENT_KEY_PREFIX}{stream_id}"


def _override_on(value: Any, pttl: Any) -> bool:
    """``on`` and an expiry set. pttl -1 = no expiry (refused), -2 = gone."""
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    try:
        ttl_ok = int(pttl) > 0
    except (TypeError, ValueError):
        ttl_ok = False
    return ttl_ok and isinstance(value, str) and value.strip().lower() == "on"


class ExpectationCache:
    def __init__(
        self,
        *,
        refresh_sec: float = 5.0,
        expectation_enabled: bool = True,
        experiment_enabled: bool = True,
    ) -> None:
        self.refresh_sec = max(0.5, float(refresh_sec))
        self.expectation_enabled = bool(expectation_enabled)
        self.experiment_enabled = bool(experiment_enabled)
        self._known_streams: set[str] = set()
        self._open: frozenset[str] = frozenset()
        self._experiment: frozenset[str] = frozenset()
        # "on" keys ignored because they carried no TTL (would never expire).
        self.experiment_no_ttl_refusals = 0
        self.refresh_failures = 0
        self.last_error: str | None = None

    # -- sync side (decide) -------------------------------------------------
    def note_stream(self, stream_id: str | None) -> None:
        if stream_id:
            self._known_streams.add(str(stream_id))

    def is_open(self, stream_id: str | None) -> bool:
        return self.expectation_enabled and bool(stream_id) and stream_id in self._open

    def open_streams(self) -> list[str]:
        return sorted(self._open) if self.expectation_enabled else []

    def experiment_on(self, stream_id: str | None) -> bool:
        return self.experiment_enabled and bool(stream_id) and stream_id in self._experiment

    def experiment_streams(self) -> list[str]:
        return sorted(self._experiment) if self.experiment_enabled else []

    # -- async side (refresh loop) ------------------------------------------
    async def refresh_once(self, redis: Any, streams: Iterable[str] | None = None) -> frozenset[str]:
        names = sorted(set(streams) if streams is not None else self._known_streams)
        if not names:
            self._open = frozenset()
            self._experiment = frozenset()
            return self._open
        try:
            pipe = redis.pipeline()
            for s in names:
                pipe.exists(expect_key(s))
                pipe.get(experiment_key(s))
                pipe.pttl(experiment_key(s))
            # A hung Redis must not freeze the last answer (a stream stuck on
            # the triggered tier): bound the round trip by the refresh period.
            results = await asyncio.wait_for(pipe.execute(), timeout=self.refresh_sec)
            opened, forced = set(), set()
            for i, s in enumerate(names):
                exists, value, pttl = results[3 * i : 3 * i + 3]
                if int(exists or 0) > 0:
                    opened.add(s)
                if _override_on(value, pttl):
                    forced.add(s)
                elif self.experiment_enabled and pttl == -1 and _override_on(value, 1):
                    self.experiment_no_ttl_refusals += 1
                    logger.warning(f"[ROUTER] experiment override for {s} has no TTL; ignored")
            self._open = frozenset(opened)
            self._experiment = frozenset(forced)
            self.last_error = None
        except Exception as exc:
            self.refresh_failures += 1
            self.last_error = str(exc)
            self._open = frozenset()
            self._experiment = frozenset()
            logger.warning(f"[ROUTER] expectation refresh failed (treated as no expectation): {exc}")
        return self._open

    async def run(self, redis: Any, stop: asyncio.Event) -> None:
        while not stop.is_set():
            await self.refresh_once(redis)
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.refresh_sec)
            except asyncio.TimeoutError:
                pass
