"""Reconciler for memory.episode_distill: no closed episode is silently left undistilled.

A closed shadow episode can miss its distill run two ways: the ``memory.episode.closed.v1``
publish was lost (pub/sub, durable-runs down at that moment), or the run ended ``failed``.
Every RECONCILE interval this finds closed direct-conversation episodes (older than a grace
period, inside a lookback window) that have no ``episode_distill_run`` row, and resubmits them
as ordinary admitted durable runs -- the retry lives on the durable run and the GPU pool, never on
a side queue:

* no attempt at all          -> the base run id (``memdistill-<episode>``), i.e. the lost event;
* every attempt terminal, none completed, the latest older than RETRY_AFTER, fewer than
  max attempts               -> a NEW attempt id (``memdistill-<episode>-a<N>``);
* an attempt still in flight, or a completed one, or max attempts reached -> nothing.

Bounded (LIMIT per pass, max attempts per episode) and idempotent (deterministic run ids: a
re-submit of an existing id is refused by the admission store).
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

from app.episode_distill_graph import base_run_id, request_from_closed_event

logger = logging.getLogger("orion-durable-runs.episode_distill_reconcile")

GRACE_SEC = 900              # let the live event path go first
LOOKBACK_HOURS = 72
RETRY_AFTER_SEC = 3600
BATCH_LIMIT = 10

CANDIDATES_SQL = """
SELECT e.episode_id, e.source_platform, e.started_at, e.last_turn_at, e.closed_at, e.close_reason,
       e.phase_at_close, e.boundary_score_at_close, e.close_lag_sec, e.closing_correlation_id,
       e.juniper_turn_count, e.command_turn_count, e.turns, e.episode_status, e.skip_reason
FROM memory_episode_shadow e
WHERE e.status = 'closed' AND e.episode_status = 'closed' AND e.source_platform IS NULL
  AND e.closed_at < %s - make_interval(secs => %s)
  AND e.closed_at > %s - make_interval(hours => %s)
  AND NOT EXISTS (SELECT 1 FROM episode_distill_run d WHERE d.episode_id = e.episode_id)
ORDER BY e.closed_at
LIMIT %s
"""
ATTEMPTS_SQL = """
SELECT run_id, terminal, updated_at FROM durable_admission_runs
WHERE run_id = %s OR run_id LIKE %s
"""


def next_attempt_run_id(episode_id: str, attempts: list[dict[str, Any]], *, now: datetime,
                        max_attempts: int, retry_after_sec: float = RETRY_AFTER_SEC) -> str | None:
    base = base_run_id(episode_id)
    if not attempts:
        return base
    if any(a.get("terminal") is None for a in attempts):
        return None                                   # in flight
    if any(a.get("terminal") == "completed" for a in attempts):
        return None                                   # done (its persist owns the run row)
    if len(attempts) >= max_attempts:
        return None
    latest = max(a["updated_at"] for a in attempts)
    if now - latest < timedelta(seconds=retry_after_sec):
        return None
    taken = {a["run_id"] for a in attempts}
    n = len(attempts) + 1
    while f"{base}-a{n}" in taken:
        n += 1
    return f"{base}-a{n}"


def closed_event_from_row(row: dict[str, Any]) -> dict[str, Any]:
    turns = row.get("turns")
    turns = json.loads(turns) if isinstance(turns, str) else (turns or [])
    return {
        "episode_id": row["episode_id"],
        "source_platform": row.get("source_platform"),
        "started_at": row["started_at"],
        "ended_at": row["last_turn_at"],
        "closed_at": row["closed_at"],
        "turn_ids": [str(t.get("correlation_id")) for t in turns if isinstance(t, dict)],
        "juniper_turn_count": int(row.get("juniper_turn_count") or 0),
        "command_turn_count": int(row.get("command_turn_count") or 0),
        "close_reason": str(row.get("close_reason") or ""),
        "phase_at_close": row.get("phase_at_close"),
        "boundary_score_at_close": row.get("boundary_score_at_close"),
        "close_lag_sec": row.get("close_lag_sec"),
        "closing_turn_id": row.get("closing_correlation_id"),
        "episode_status": row.get("episode_status") or "closed",
        "skip_reason": row.get("skip_reason"),
    }


async def reconcile_once(pool: Any, submit: Callable[[Any], Awaitable[Any]], settings: Any, *,
                         now: datetime | None = None) -> list[str]:
    """One pass. Returns the run ids submitted."""
    now = now or datetime.now(timezone.utc)
    async with pool.connection() as conn:
        rows = await (await conn.execute(CANDIDATES_SQL, (now, GRACE_SEC, now, LOOKBACK_HOURS, BATCH_LIMIT))).fetchall()
        plans = []
        for row in rows:
            row = dict(row)
            base = base_run_id(row["episode_id"])
            attempts = [dict(a) for a in await (await conn.execute(ATTEMPTS_SQL, (base, f"{base}-a%"))).fetchall()]
            run_id = next_attempt_run_id(row["episode_id"], attempts, now=now,
                                         max_attempts=int(settings.memory_episode_distill_max_attempts))
            if run_id:
                plans.append((row, run_id))
    submitted = []
    for row, run_id in plans:
        request = request_from_closed_event(closed_event_from_row(row), settings=settings, now=now, run_id=run_id)
        if request is None:
            continue
        try:
            await submit(request)
        except ValueError as exc:   # duplicate run id: another pass or the event got there first
            logger.info("memory_episode_reconcile_duplicate run=%s err=%s", run_id, exc)
            continue
        submitted.append(run_id)
        logger.info("memory_episode_reconcile_submitted run=%s episode=%s", run_id, row["episode_id"])
    return submitted


async def run_reconcile_loop(pool: Any, submit: Callable[[Any], Awaitable[Any]], settings: Any,
                             stop: asyncio.Event) -> None:
    interval = max(60.0, float(settings.memory_episode_reconcile_interval_sec))
    while not stop.is_set():
        try:
            await reconcile_once(pool, submit, settings)
        except Exception:  # noqa: BLE001 -- e.g. migration not applied yet: log, try again later
            logger.exception("memory_episode_reconcile_failed")
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass
