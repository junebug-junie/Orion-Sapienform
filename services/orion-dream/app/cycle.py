"""Dream cycle v2: one sleep, end to end.

    pressure --(>= threshold AND idle)--> replay --> REM compaction (staged)
                                            |
                                            +--> recombination --> hypotheses
                                                 (dream arm + control arm)

Nothing here mutates canonical memory. The only writes are the v2 tables
(cycle_store.CYCLE_WRITE_TABLES) and, when ORION_DREAM_REM_ENABLED, the
existing staged dream_compaction_delta table.

All IO is injected (`CycleDeps`) so the whole cycle runs in tests with fakes.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Optional
from uuid import uuid4

from orion.schemas.dream_cycle import DreamCycleV1, SleepPressureV1

from app.recombine import Complete, control_pairs, dream_pairs, recombine
from app.replay import build_candidates, compute_pressure, select_replay
from app.settings import settings

logger = logging.getLogger("orion-dream.cycle")


@dataclass
class CycleDeps:
    load_source_rows: Callable[[datetime, int], dict[str, list[dict[str, Any]]]]
    load_idle_minutes: Callable[[], Optional[float]]
    # started_at of the last non-failed cycle -> the replay window start
    load_last_window_start: Callable[[], Optional[datetime]]
    # ended_at of the last cycle of any status -> the min-interval gate
    load_last_attempt_end: Callable[[], Optional[datetime]]
    persist_cycle: Callable[[DreamCycleV1], bool]
    complete: Complete
    # (cycle_id, window since) -> staged delta id or None
    rem_compaction: Optional[Callable[[str, datetime], Awaitable[Optional[str]]]] = None


def _utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def window_start(now: datetime, last_start: Optional[datetime]) -> datetime:
    """Since the last good cycle began, but never further back than the lookback cap."""
    floor = now - timedelta(hours=settings.DREAM_LOOKBACK_HOURS)
    last = _utc(last_start)
    return max(floor, last) if last else floor


def read_pressure(deps: CycleDeps, now: datetime, last_start: Optional[datetime]):
    """(SleepPressureV1, candidates). Blocking (sync DB); call via to_thread."""
    since = window_start(now, last_start)
    candidates = build_candidates(deps.load_source_rows(since, settings.DREAM_CANDIDATES_PER_SOURCE))
    total, counts = compute_pressure(candidates)
    pressure = SleepPressureV1(
        since=since,
        computed_at=now,
        pressure=total,
        counts=counts,
        idle_minutes=deps.load_idle_minutes(),
        threshold=settings.DREAM_SLEEP_PRESSURE_THRESHOLD,
        idle_required_minutes=settings.DREAM_IDLE_MINUTES,
    )
    return pressure, candidates


def too_soon(now: datetime, last_end: Optional[datetime]) -> bool:
    last = _utc(last_end)
    return last is not None and now - last < timedelta(hours=settings.DREAM_MIN_INTERVAL_HOURS)


async def run_cycle_once(deps: CycleDeps, *, trigger: str = "pressure", force: bool = False) -> Optional[DreamCycleV1]:
    """Run one sleep if due (or forced). None when not due. Never raises."""
    started = datetime.now(timezone.utc)
    try:
        last_start = await asyncio.to_thread(deps.load_last_window_start)
        last_end = await asyncio.to_thread(deps.load_last_attempt_end)
        pressure, candidates = await asyncio.to_thread(read_pressure, deps, started, last_start)
    except Exception as exc:
        logger.warning("dream_cycle pressure read failed err=%s", exc)
        return None

    if not force:
        if too_soon(started, last_end):
            return None
        if not pressure.should_sleep:
            logger.info(
                "dream_cycle not due pressure=%.2f/%.2f idle=%s/%s counts=%s",
                pressure.pressure, pressure.threshold, pressure.idle_minutes,
                pressure.idle_required_minutes, pressure.counts,
            )
            return None

    cycle_id = f"dc-{uuid4().hex[:12]}"
    if not candidates:
        cycle = DreamCycleV1(
            cycle_id=cycle_id, trigger=trigger, status="empty",  # type: ignore[arg-type]
            started_at=started, ended_at=datetime.now(timezone.utc), pressure=pressure,
            note="nothing unprocessed since last sleep",
        )
        await asyncio.to_thread(deps.persist_cycle, cycle)
        return cycle

    replay = select_replay(candidates, settings.DREAM_REPLAY_MAX)

    compaction_delta_id = None
    if deps.rem_compaction is not None:
        try:
            compaction_delta_id = await deps.rem_compaction(cycle_id, pressure.since)
        except Exception as exc:
            logger.warning("dream_cycle rem compaction failed err=%s", exc)

    d_pairs = dream_pairs(replay, settings.DREAM_HYPOTHESES_PER_CYCLE)
    c_pairs = control_pairs(
        candidates, settings.DREAM_CONTROL_PER_CYCLE, seed=cycle_id, exclude=d_pairs
    )
    rem = await recombine(
        d_pairs + c_pairs, deps.complete, cycle_id=cycle_id,
        ttl_hours=settings.DREAM_HYPOTHESIS_TTL_HOURS,
    )

    pairs = d_pairs + c_pairs
    # Every LLM call failed: nothing was recombined, so this sleep must not
    # close the window (the backlog stays for the next attempt, which the
    # min-interval gate still spaces out).
    failed = bool(pairs) and rem.failures == len(pairs)
    cycle = DreamCycleV1(
        cycle_id=cycle_id,
        trigger=trigger,  # type: ignore[arg-type]
        status="failed" if failed else "completed",
        started_at=started,
        ended_at=datetime.now(timezone.utc),
        pressure=pressure,
        replay=replay,
        hypotheses=rem.hypotheses,
        no_link_count=rem.no_link,
        unparseable_count=rem.unparseable,
        llm_failures=rem.failures,
        compaction_delta_id=compaction_delta_id,
        note=f"pairs dream={len(d_pairs)} control={len(c_pairs)}",
    )
    if not await asyncio.to_thread(deps.persist_cycle, cycle):
        logger.warning("dream_cycle %s ran but was not persisted", cycle_id)
    logger.info(
        "dream_cycle %s id=%s pressure=%.2f replay=%d hypotheses=%d (dream=%d control=%d) "
        "no_link=%d unparseable=%d llm_failures=%d",
        cycle.status, cycle_id, pressure.pressure, len(replay), len(rem.hypotheses),
        sum(1 for h in rem.hypotheses if h.arm == "dream"),
        sum(1 for h in rem.hypotheses if h.arm == "control"),
        rem.no_link, rem.unparseable, rem.failures,
    )
    return cycle


async def sleep_loop(deps: CycleDeps, stop: asyncio.Event) -> None:
    """Check pressure every interval; sleep when due. Exits on `stop`."""
    while not stop.is_set():
        await run_cycle_once(deps)
        try:
            await asyncio.wait_for(stop.wait(), timeout=settings.DREAM_CYCLE_CHECK_INTERVAL_SEC)
        except asyncio.TimeoutError:
            pass
