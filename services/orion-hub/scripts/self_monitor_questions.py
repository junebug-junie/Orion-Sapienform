"""Self-monitor E2 in Hub: a faculty newly held failing mints ONE self-question (rung 6).

Spec: docs/superpowers/specs/2026-10-20-self-monitor-organ-design.md, "Effects" E2. Every tick this
reads the organ's published state (Redis ``orion:self_monitor:latest``, written hourly by
orion-durable-runs), compares it with the statuses it saw last (Redis ``SEEN_KEY``, no TTL, so a
Hub restart remembers), and inserts one ``curiosity_self_questions`` row per new failing episode
(``minted_by='orion'``). The pure decision is ``orion.self_monitor.questions.self_questions_for``.

* Unknown, stale or absent state: nothing is minted and the memory is not touched.
* The insert ignores conflicts (``UPSERT_SEED_SQL``: ON CONFLICT DO NOTHING) on a question id that
  is deterministic per episode, so a lost memory never asks twice or re-opens an answered question.
* A faculty whose insert failed is not remembered, so the next tick retries it.

Never writes a node prediction_error and never touches endogenous curiosity's concept_graph zone.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from orion.curiosity.self_question_pool import UPSERT_SEED_SQL
from orion.self_monitor.questions import self_questions_for
from orion.self_monitor.view import SELF_MONITOR_REDIS_KEY, self_monitor_view

logger = logging.getLogger("orion-hub.self_monitor_questions")

SEEN_KEY = "orion:self_monitor:questions:seen"


def _provided(provider: Callable[[], Any]) -> Any:
    """The provider's value, or None when it is not ready (a bus not yet connected raises)."""
    try:
        return provider()
    except Exception:  # noqa: BLE001
        return None


class SelfMonitorQuestionMinter:
    def __init__(
        self,
        *,
        enabled: bool,
        redis_provider: Callable[[], Any],
        pool_provider: Callable[[], Any],
        tick_interval_sec: float = 900.0,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self._redis_provider = redis_provider
        self._pool_provider = pool_provider
        self.tick_interval_sec = max(30.0, float(tick_interval_sec))
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._task: Optional[asyncio.Task] = None
        self.last_tick: dict[str, Any] = {}

    async def start(self) -> None:
        if not self.enabled:
            logger.info("self_monitor_questions disabled")
            return
        self._task = asyncio.create_task(self._run())
        logger.info("self_monitor_questions started tick=%ss", self.tick_interval_sec)

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    async def _run(self) -> None:
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 -- one bad tick must not stop the loop
                logger.exception("self_monitor_questions_tick_failed")
            await asyncio.sleep(self.tick_interval_sec)

    async def tick(self) -> dict[str, Any]:
        now = self._clock()
        redis = _provided(self._redis_provider)
        pool = _provided(self._pool_provider)
        if redis is None or pool is None:
            self.last_tick = {"at": now.isoformat(), "skipped": "no_redis" if redis is None else "no_pool"}
            return self.last_tick
        raw, seen_raw = await redis.get(SELF_MONITOR_REDIS_KEY), await redis.get(SEEN_KEY)
        view = self_monitor_view(raw, now=now)
        if not view.known:
            self.last_tick = {"at": now.isoformat(), "view": view.reason, "minted": 0}
            return self.last_tick
        try:
            seen = json.loads(seen_raw) if seen_raw else {}
            if not isinstance(seen, dict):
                raise ValueError("seen is not an object")
        except Exception:  # noqa: BLE001 -- corrupt memory: the deterministic ids still dedupe
            logger.warning("self_monitor_questions_seen_unreadable; starting from empty")
            seen = {}
        mints, next_seen = self_questions_for(seen, view, now=now)
        minted: list[str] = []
        failed: set[str] = set()
        if mints:
            async with pool.acquire() as conn:
                for mint in mints:
                    try:
                        status = await conn.execute(UPSERT_SEED_SQL, mint.question_id, mint.text, mint.family,
                                                    False, "orion")
                    except Exception as exc:  # noqa: BLE001 -- retried next tick
                        failed.add(mint.faculty)
                        logger.warning("self_monitor_question_insert_failed faculty=%s err=%s", mint.faculty, exc)
                        continue
                    if str(status or "").endswith(" 1"):  # "INSERT 0 0" = already asked
                        minted.append(mint.question_id)
                    logger.info("self_monitor_question_minted faculty=%s status=%s question_id=%s inserted=%s text=%r",
                                mint.faculty, mint.status, mint.question_id, status, mint.text)
        for faculty in failed:
            if faculty in seen:
                next_seen[faculty] = seen[faculty]
            else:
                next_seen.pop(faculty, None)
        if next_seen != seen:
            await redis.set(SEEN_KEY, json.dumps(next_seen, sort_keys=True))
        self.last_tick = {"at": now.isoformat(), "view": view.reason, "minted": len(minted),
                          "question_ids": minted, "failed": sorted(failed)}
        return self.last_tick
