"""The day dream, Patch 2: the night tick that starts one ``dream.day`` run for the Denver day that just closed.

Design: docs/superpowers/specs/2026-11-17-nightly-day-dream-design.md, Patch 2 "Start". Every POLL_SEC, inside the
Denver window [DREAM_DAY_START_HOUR_LOCAL, DREAM_DAY_END_HOUR_LOCAL) of night N (01:00-03:00 by default: after the
day closes, before the reminiscence's 03:00 window), for the day N-1:

1. A ``dream_day`` row for N-1 exists: nothing (``complete``).
2. Runs of N-1 (``dream-day-<day>``, ``-a<k>``): none -> submit the base run. One in flight or completed, or
   MAX_RUNS_PER_DAY spent -> nothing. All failed, the newest older than RETRY_AFTER_SEC, and one whole turn
   attempt still fits before the window ends -> one more run.

The run's admission deadline is the end of the window: a night with no run is a missed night (the spec), shown by
the missing dream_day row. Idempotent: deterministic run ids, and the store refuses a duplicate.

Why this service ticks, not orion-dream (the spec's "orion-dream submits it through carry_submit"): the run lives
here, the reminiscence and field notes already tick here with exactly this window/poll/run_request shape, and a
submit from here needs no cortex-orch hop or receipt round-trip. orion-dream is unchanged.

``POST /dream-day/run`` (``submit_for(day, forced=True)``) is the hand trigger: any closed Denver day of the last
week, outside the window, with a 3 h deadline. Same idempotency rules.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

from orion.memory.dream_day import ZONE, night_window
from orion.schemas.dream_day import (
    DREAM_DAY_MIN_ATTEMPT_SEC, DREAM_DAY_RUN_PREFIX, DREAM_DAY_WORKFLOW, DreamDayBriefV1, base_run_id,
)
from orion.schemas.durable_run import DurableRunRequestV1
from orion.schemas.resource_admission import ResourceRequirementV1

logger = logging.getLogger("orion-durable-runs.dream_day_poll")

POLL_SEC = 600.0
MAX_RUNS_PER_DAY = 2
RETRY_AFTER_SEC = 1200.0
# A hand trigger: how long its run may wait for the hold, and how far back a day may be.
FORCED_DEADLINE_SEC = 3 * 3600.0
FORCED_MAX_DAYS_BACK = 7
FORCED_MAX_RUNS = 4

RUNS_SQL = "SELECT run_id, terminal, created_at, updated_at FROM durable_admission_runs WHERE run_id LIKE %s"
IN_FLIGHT_SQL = "SELECT count(*) AS n FROM durable_admission_runs WHERE run_id LIKE %s AND terminal IS NULL"
DONE_SQL = "SELECT run_id FROM dream_day WHERE day = %s"


def dreamt_day(now: datetime) -> date:
    """The day a night's run reads: the Denver day before ``now``'s Denver date."""
    return now.astimezone(ZONE).date() - timedelta(days=1)


def next_run_id(day: date, attempts: list[dict[str, Any]], *, now: datetime, deadline: datetime, min_left_sec: float,
                max_runs: int = MAX_RUNS_PER_DAY, retry_after_sec: float = RETRY_AFTER_SEC) -> str | None:
    """The run to submit for ``day``, or None (the reminiscence tick's rule, app/reminiscence_poll.py)."""
    base = base_run_id(day)
    if (deadline - now).total_seconds() < min_left_sec:
        return None
    if not attempts:
        return base
    if any(a.get("terminal") is None for a in attempts):
        return None
    if any(a.get("terminal") == "completed" for a in attempts):
        return None
    if len(attempts) >= max_runs:
        return None
    if now - max(a["updated_at"] for a in attempts) < timedelta(seconds=retry_after_sec):
        return None
    taken = {a["run_id"] for a in attempts}
    k = len(attempts) + 1
    while f"{base}-a{k}" in taken:
        k += 1
    return f"{base}-a{k}"


def run_request(run_id: str, brief: DreamDayBriefV1, *, now: datetime, deadline: datetime) -> DurableRunRequestV1:
    return DurableRunRequestV1(
        run_id=run_id,
        workflow=DREAM_DAY_WORKFLOW,
        # Deterministic: a re-submit of the same run id is the same request (the store's dedup).
        correlation_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"orion:dream.day:{run_id}")),
        requested_at=now,
        brief=brief,
        admission=ResourceRequirementV1(resource=f"llm.route.{brief.llm_route}", preferred_lane=brief.llm_route,
                                        priority="background", deadline_at=deadline),
    )


@dataclass
class TickResult:
    at: str
    day: str
    decision: str          # submitted | outside_window | complete | nothing_due | waiting_in_flight | ...
    run_id: str | None = None
    deadline_at: str | None = None
    forced: bool = False
    dry_run: bool = False
    error: str | None = None
    request: DurableRunRequestV1 | None = field(default=None, repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != "request"}


class DreamDayTicker:
    def __init__(self, *, settings: Any, pool: Any, submit: Callable[[DurableRunRequestV1], Awaitable[Any]] | None,
                 now: Callable[[], datetime] | None = None):
        self.settings = settings
        self.pool = pool
        self.submit = submit
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.last: TickResult | None = None
        self._lock = asyncio.Lock()

    async def _state(self, day: date) -> tuple[str | None, list[dict[str, Any]], int]:
        async with self.pool.connection() as conn:
            done = await (await conn.execute(DONE_SQL, (day,))).fetchone()
            rows = await (await conn.execute(RUNS_SQL, (base_run_id(day) + "%",))).fetchall()
            row = await (await conn.execute(IN_FLIGHT_SQL, (DREAM_DAY_RUN_PREFIX + "%",))).fetchone()
        return (done or {}).get("run_id"), [dict(r) for r in rows], int(row["n"]) if row else 0

    async def tick_once(self, *, dry_run: bool = False) -> TickResult:
        """One decision for tonight. ``dry_run``: decide and build the request, submit nothing."""
        async with self._lock:
            now = self.now()
            night = now.astimezone(ZONE).date()
            lo, hi = night_window(night, int(self.settings.dream_day_start_hour_local),
                                  int(self.settings.dream_day_end_hour_local))
            result = await self._decide(dreamt_day(now), now, deadline=hi, in_window=lo <= now < hi,
                                        forced=False, dry_run=dry_run)
            if not result.dry_run:
                self.last = result
            return result

    async def submit_for(self, day: date, *, dry_run: bool = False) -> TickResult:
        """The hand trigger (POST /dream-day/run): ``day`` must be a closed Denver day of the last week."""
        async with self._lock:
            now = self.now()
            today = now.astimezone(ZONE).date()
            result = TickResult(at=now.isoformat(), day=day.isoformat(), decision="", forced=True, dry_run=dry_run)
            if not (today - timedelta(days=FORCED_MAX_DAYS_BACK) <= day < today):
                result.decision = "day_out_of_range"
                result.error = f"a closed Denver day from the last {FORCED_MAX_DAYS_BACK} days, before {today}"
                return result
            return await self._decide(day, now, deadline=now + timedelta(seconds=FORCED_DEADLINE_SEC),
                                      in_window=True, forced=True, dry_run=dry_run)

    async def _decide(self, day: date, now: datetime, *, deadline: datetime, in_window: bool, forced: bool,
                      dry_run: bool) -> TickResult:
        result = TickResult(at=now.isoformat(), day=day.isoformat(), decision="", deadline_at=deadline.isoformat(),
                            forced=forced, dry_run=dry_run)
        if not in_window:
            result.decision = "outside_window"
            return result
        try:
            done, runs, in_flight = await self._state(day)
        except Exception as exc:  # noqa: BLE001 -- e.g. the dream_day table is missing: say so, try next tick
            result.decision, result.error = "state_unreadable", f"{type(exc).__name__}: {exc}"[:300]
            logger.warning("dream_day_tick_state_unreadable %s", result.error)
            return result
        if done:
            result.decision, result.run_id = "complete", done
            return result
        timeout = float(self.settings.dream_day_turn_timeout_sec)
        # A new run needs the least night an attempt can use, not a whole (longer) brief attempt: its window is
        # clamped to the deadline (orion/schemas/dream_day.py DREAM_DAY_MIN_ATTEMPT_SEC).
        run_id = next_run_id(day, runs, now=now, deadline=deadline, min_left_sec=min(timeout, DREAM_DAY_MIN_ATTEMPT_SEC),
                             **({"max_runs": FORCED_MAX_RUNS, "retry_after_sec": 0.0} if forced else {}))
        if run_id is None:
            result.decision = "nothing_due"
            return result
        if in_flight:
            result.decision = "waiting_in_flight"
            return result
        brief = DreamDayBriefV1(dream_day="dream.day", day=day, timeout_sec=timeout,
                                llm_route=str(self.settings.dream_day_route), operator_forced=forced)
        request = run_request(run_id, brief, now=now, deadline=deadline)
        result.request, result.run_id = request, run_id
        if dry_run:
            result.decision = "would_submit"
            return result
        try:
            await self.submit(request)
        except ValueError as exc:   # duplicate run id (a concurrent tick): nothing to do
            result.decision = "duplicate"
            logger.info("dream_day_tick_duplicate run=%s err=%s", run_id, exc)
            return result
        result.decision = "submitted"
        logger.info("dream_day_tick_submitted %s", json.dumps(result.as_dict()))
        return result

    async def run_forever(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await self.tick_once()
            except Exception:  # noqa: BLE001 -- e.g. DB away: log, try next tick (the window allows several)
                logger.exception("dream_day_tick_failed")
            try:
                await asyncio.wait_for(stop.wait(), timeout=POLL_SEC)
            except asyncio.TimeoutError:
                pass

    def health(self) -> dict[str, Any] | None:
        return self.last.as_dict() if self.last else None
