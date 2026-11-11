"""Daily reminiscence, Patch 2: the overnight tick that starts one ``reminiscence.episode`` run a day.

Design: docs/superpowers/specs/2026-11-07-daily-reminiscence-design.md, section 6. Every
REMINISCENCE_POLL_SEC (600 s), inside the Denver window [REMINISCENCE_HOUR_LOCAL, REMINISCENCE_END_HOUR_LOCAL)
(03:00-09:00 by default: the agent lanes are contended by day, and lane time is this run's real cost):

1. Today's episode is the Denver date. If its five hops are already live and no run of today failed
   (Juniper asked in chat, or today's run completed), nothing. If they are live but today's run failed (its
   cross-read ran out of attempts), the retry below runs: its hops are all ``already_written``, so it is the
   one cross-read turn.
2. The period: the one today's first hop fixed, else the oldest week that ended before today with no
   complete episode, with a run of weeks that have no commit on main joined onto the next week that has one
   (``app/reminiscence_ledger.py``, the same rule ``dig reminisce`` uses). None left: nothing
   (``no_week``); only quiet weeks left, none with a commit after them yet: nothing (``waiting_for_commit``);
   main's history unreadable: nothing (``history_unreadable``, never a guessed week); a period that would
   overlap a visited week (an out-of-order ledger): nothing (``period_refused``).
3. Runs of today (``reminiscence-<episode>``, ``-a<k>``): none -> submit the base run. One in flight or
   completed, or MAX_RUNS_PER_DAY spent -> nothing. All failed, the newest older than RETRY_AFTER_SEC, and a
   whole step attempt still fits before the window ends -> one more run (it skips the hops already written).

The run's admission deadline is the end of the window: a week left unfinished is picked again the next day
(design: an unfinished week stays open; hops do not carry over). Idempotent: deterministic run ids, and the
store refuses a duplicate. Reads only ids, dates and booleans from the ledger.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Awaitable, Callable
from zoneinfo import ZoneInfo

from app.reminiscence_ledger import HistoryUnreadable, PeriodRefused
from orion.schemas.durable_run import DurableRunRequestV1
from orion.schemas.reminiscence_run import (
    REMINISCENCE_RUN_PREFIX, REMINISCENCE_WORKFLOW, ReminiscenceRunBriefV1, base_run_id,
)
from orion.schemas.resource_admission import ResourceRequirementV1

logger = logging.getLogger("orion-durable-runs.reminiscence_poll")

ZONE = ZoneInfo("America/Denver")
MAX_RUNS_PER_DAY = 2
RETRY_AFTER_SEC = 1800.0
LANE = "agent"

RUNS_SQL = "SELECT run_id, terminal, created_at, updated_at FROM durable_admission_runs WHERE run_id LIKE %s"
# Any unfinished reminiscence run, of any day (a run from yesterday still running blocks a second one).
IN_FLIGHT_SQL = "SELECT count(*) AS n FROM durable_admission_runs WHERE run_id LIKE %s AND terminal IS NULL"


def window(day: date, start_hour: int, end_hour: int) -> tuple[datetime, datetime]:
    """[start, end) of ``day``'s window as aware UTC instants (DST-correct: built in Denver time)."""
    lo = datetime.combine(day, time(start_hour), tzinfo=ZONE)
    hi = datetime.combine(day, time(end_hour), tzinfo=ZONE)
    return lo.astimezone(timezone.utc), hi.astimezone(timezone.utc)


def next_run_id(episode: date, attempts: list[dict[str, Any]], *, now: datetime, deadline: datetime,
                min_left_sec: float, max_runs: int = MAX_RUNS_PER_DAY,
                retry_after_sec: float = RETRY_AFTER_SEC) -> str | None:
    """The run to submit for ``episode``, or None. Same rule as the field-note poller, plus: a retry that
    could not get one whole step attempt before the window ends is not started."""
    base = base_run_id(episode)
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


def run_request(run_id: str, brief: ReminiscenceRunBriefV1, *, now: datetime, deadline: datetime) -> DurableRunRequestV1:
    return DurableRunRequestV1(
        run_id=run_id,
        workflow=REMINISCENCE_WORKFLOW,
        # Deterministic: a re-submit of the same run id is the same request (the store's dedup).
        correlation_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"orion:reminiscence:{run_id}")),
        requested_at=now,
        brief=brief,
        admission=ResourceRequirementV1(resource=f"llm.route.{LANE}", preferred_lane=LANE, priority="background",
                                        deadline_at=deadline),
    )


@dataclass
class TickResult:
    at: str
    episode: str
    decision: str                     # submitted | outside_window | complete | no_week | waiting_for_commit | ...
    run_id: str | None = None
    period_start: str | None = None
    period_end: str | None = None
    week_fixed_by_hop: bool | None = None
    quiet_weeks: int | None = None
    hops_written: list[str] = field(default_factory=list)
    weeks_complete: int | None = None
    weeks_total: int | None = None
    deadline_at: str | None = None
    dry_run: bool = False
    error: str | None = None
    # The request built for a submit or a would-submit (not in as_dict: /health keeps the decision only).
    request: DurableRunRequestV1 | None = field(default=None, repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != "request"}


class ReminiscenceTicker:
    def __init__(self, *, settings: Any, pool: Any, submit: Callable[[DurableRunRequestV1], Awaitable[Any]] | None,
                 ledger: Any, now: Callable[[], datetime] | None = None):
        self.settings = settings
        self.pool = pool
        self.submit = submit
        self.ledger = ledger
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.last: TickResult | None = None
        self._lock = asyncio.Lock()

    async def _runs(self, episode: date) -> tuple[list[dict[str, Any]], int]:
        async with self.pool.connection() as conn:
            rows = await (await conn.execute(RUNS_SQL, (base_run_id(episode) + "%",))).fetchall()
            row = await (await conn.execute(IN_FLIGHT_SQL, (REMINISCENCE_RUN_PREFIX + "%",))).fetchone()
        # LIKE 'reminiscence-2026-11-08%' also matches nothing else: ids are the date plus an optional -a<k>.
        return [dict(r) for r in rows], int(row["n"]) if row else 0

    async def tick_once(self, *, dry_run: bool = False, ignore_window: bool = False) -> TickResult:
        """One decision. ``dry_run``: decide and build the request (``result.request``), submit nothing.
        ``ignore_window``: decide as if inside the window (a read-only check by day; never with a submit)."""
        async with self._lock:
            result = await self._tick(dry_run=dry_run or ignore_window, ignore_window=ignore_window)
            if not result.dry_run:
                self.last = result
            return result

    async def _tick(self, *, dry_run: bool, ignore_window: bool) -> TickResult:
        now = self.now()
        episode = now.astimezone(ZONE).date()
        start_h, end_h = int(self.settings.reminiscence_hour_local), int(self.settings.reminiscence_end_hour_local)
        lo, hi = window(episode, start_h, end_h)
        result = TickResult(at=now.isoformat(), episode=episode.isoformat(), decision="", deadline_at=hi.isoformat(),
                            dry_run=dry_run)
        if not ignore_window and not (lo <= now < hi):
            result.decision = "outside_window"
            return result
        try:
            written = await asyncio.to_thread(self.ledger.hop_ids, episode)
            result.hops_written = sorted(written)
            pick = await asyncio.to_thread(self.ledger.week_for, episode)
        except OSError as exc:
            result.decision, result.error = "ledger_unreadable", str(exc)[:300]
            logger.warning("reminiscence_tick_ledger_unreadable %s", exc)
            return result
        except HistoryUnreadable as exc:
            result.decision, result.error = "history_unreadable", str(exc)[:300]
            logger.warning("reminiscence_tick_history_unreadable %s", exc)
            return result
        except PeriodRefused as exc:
            result.decision, result.error = "period_refused", str(exc)[:300]
            logger.warning("reminiscence_tick_period_refused %s", exc)
            return result
        if pick is None:
            wait = await asyncio.to_thread(self.ledger.quiet_wait_from, episode)
            if wait is not None:
                result.decision = "waiting_for_commit"
                result.error = f"the weeks from {wait.isoformat()} have no commit on main yet"
            else:
                result.decision = "no_week"
            return result
        result.period_start, result.period_end = pick.period_start.isoformat(), pick.period_end.isoformat()
        result.week_fixed_by_hop = pick.fixed_by_hop
        result.quiet_weeks = pick.quiet_weeks
        result.weeks_complete, result.weeks_total = pick.weeks_complete, pick.weeks_total
        runs, in_flight = await self._runs(episode)
        if len(written) == 5 and (not runs or any(r.get("terminal") == "completed" for r in runs)):
            result.decision = "complete"
            return result
        timeout = float(self.settings.reminiscence_turn_timeout_sec)
        # A by-day check outside the window decides as of the window's opening.
        at = lo if ignore_window and not (lo <= now < hi) else now
        run_id = next_run_id(episode, runs, now=at, deadline=hi, min_left_sec=2 * timeout)
        if run_id is None:
            result.decision = "nothing_due"
            return result
        if in_flight:
            result.decision = "waiting_in_flight"
            return result
        brief = ReminiscenceRunBriefV1(
            reminiscence="reminiscence.episode", episode=episode, period_start=pick.period_start,
            period_end=pick.period_end, weeks_complete=pick.weeks_complete, weeks_total=pick.weeks_total,
            timeout_sec=timeout)
        request = run_request(run_id, brief, now=now, deadline=hi)
        result.request = request
        result.run_id = run_id
        if dry_run:
            result.decision = "would_submit"
            return result
        try:
            await self.submit(request)
        except ValueError as exc:   # duplicate run id (a concurrent tick): nothing to do
            result.decision = "duplicate"
            logger.info("reminiscence_tick_duplicate run=%s err=%s", run_id, exc)
            return result
        result.decision = "submitted"
        logger.info("reminiscence_tick_submitted %s", json.dumps(result.as_dict()))
        return result

    async def run_forever(self, stop: asyncio.Event) -> None:
        interval = max(60.0, float(self.settings.reminiscence_poll_sec))
        while not stop.is_set():
            try:
                await self.tick_once()
            except Exception:  # noqa: BLE001 -- e.g. DB away: log, try next tick (the window allows several)
                logger.exception("reminiscence_tick_failed")
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass

    def health(self) -> dict[str, Any] | None:
        return self.last.as_dict() if self.last else None
