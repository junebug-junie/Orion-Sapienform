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
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Optional
from uuid import uuid4

from orion.schemas.dream_cycle import DreamCycleV1, DreamPressureObservationV1, SleepPressureV1

from app.recombine import Complete, control_pairs, dream_pairs, recombine
from app.replay import compute_pressure, keyed_candidates, prior_keys, select_replay
from app.settings import settings

logger = logging.getLogger("orion-dream.cycle")


@dataclass
class CycleDeps:
    # (since, limit, until=None) -> one row per thing per source in [since, until)
    load_source_rows: Callable[..., dict[str, list[dict[str, Any]]]]
    load_idle_minutes: Callable[[], Optional[float]]
    # started_at of the last non-failed cycle -> the replay window start
    load_last_window_start: Callable[[], Optional[datetime]]
    # ended_at of the last cycle of any status -> the min-interval gate
    load_last_attempt_end: Callable[[], Optional[datetime]]
    persist_cycle: Callable[[DreamCycleV1], bool]
    complete: Complete
    # (cycle_id, window since) -> staged delta id or None
    rem_compaction: Optional[Callable[[str, datetime], Awaitable[Optional[str]]]] = None
    persist_pressure_observation: Optional[Callable[[DreamPressureObservationV1], bool]] = None
    # One deps instance per loop/request. The real clock loaders append failures;
    # each serialized check clears it before reading. Scheduling still sees None.
    read_errors: list[str] = field(default_factory=list)


# Rows are one per thing (cycle_store), ~72 metacog kinds per 48 h live (10-09).
# A cap near that drops keys silently and makes old ones look new; replay's own
# DREAM_REPLAY_MAX is what bounds a sleep.
KEYS_PER_SOURCE = 5000


def _utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def window_start(now: datetime, last_start: Optional[datetime]) -> datetime:
    """Since the last good cycle began, but never further back than the lookback cap."""
    floor = now - timedelta(hours=settings.DREAM_LOOKBACK_HOURS)
    last = _utc(last_start)
    return max(floor, last) if last else floor


def read_pressure(deps: CycleDeps, now: datetime, last_start: Optional[datetime], *, read_errors=None):
    """(SleepPressureV1, candidates). Blocking (sync DB); call via to_thread."""
    since = window_start(now, last_start)
    current_rows = deps.load_source_rows(since, KEYS_PER_SOURCE)
    keyed = keyed_candidates(current_rows)
    lookback = since - timedelta(hours=settings.DREAM_LOOKBACK_HOURS)
    previous_rows = deps.load_source_rows(lookback, KEYS_PER_SOURCE, until=since)
    seen = prior_keys(previous_rows)
    if read_errors is not None:
        read_errors.extend(f"current:{kind}" for kind in getattr(current_rows, "read_errors", ()))
        read_errors.extend(f"prior:{kind}" for kind in getattr(previous_rows, "read_errors", ()))
    total, counts, new_counts = compute_pressure(keyed, seen)
    candidates = list(keyed.values())
    pressure = SleepPressureV1(
        since=since,
        computed_at=now,
        pressure=total,
        counts=counts,
        new_counts=new_counts,
        idle_minutes=deps.load_idle_minutes(),
        threshold=settings.DREAM_SLEEP_PRESSURE_THRESHOLD,
        idle_required_minutes=settings.DREAM_IDLE_MINUTES,
    )
    return pressure, candidates


def overdue(now: datetime, last_start: Optional[datetime]) -> bool:
    """The window has reached DREAM_LOOKBACK_HOURS, the furthest it reaches back.
    Past this, unreplayed material starts falling out of view. Pressure counts
    only NEW things, so a repetitive stretch can hold it under threshold
    indefinitely; this is the backstop that still lets Orion sleep."""
    last = _utc(last_start)
    return last is None or now - last >= timedelta(hours=settings.DREAM_LOOKBACK_HOURS)


def too_soon(now: datetime, last_end: Optional[datetime]) -> bool:
    last = _utc(last_end)
    return last is not None and now - last < timedelta(hours=settings.DREAM_MIN_INTERVAL_HOURS)


async def run_cycle_once(deps: CycleDeps, *, trigger: str = "pressure", force: bool = False) -> Optional[DreamCycleV1]:
    """Run one sleep if due (or forced). None when not due. Never raises."""
    started = datetime.now(timezone.utc)
    deps.read_errors.clear()
    read_errors = []
    try:
        last_start = await asyncio.to_thread(deps.load_last_window_start)
        last_end = await asyncio.to_thread(deps.load_last_attempt_end)
        pressure, candidates = await asyncio.to_thread(read_pressure, deps, started, last_start, read_errors=read_errors)
        read_errors.extend(deps.read_errors)
    except Exception as exc:
        logger.warning("dream_cycle pressure read failed err=%s", exc)
        return None

    if deps.persist_pressure_observation is not None:
        # Observe every successful read, including refractory, busy and low-pressure
        # checks. A recording failure must not authorize, suppress or fail a dream.
        try:
            observation = DreamPressureObservationV1(
                check_id=f"dp-{uuid4().hex}", observed_at=started, reading=pressure,
                trigger=trigger, forced=force, last_window_start=_utc(last_start),
                last_attempt_end=_utc(last_end), source_errors=read_errors,
                min_interval_hours=settings.DREAM_MIN_INTERVAL_HOURS,
                check_interval_sec=settings.DREAM_CYCLE_CHECK_INTERVAL_SEC,
                lookback_hours=settings.DREAM_LOOKBACK_HOURS,
            )
            if not await asyncio.to_thread(deps.persist_pressure_observation, observation):
                logger.warning("dream_pressure_history_failed check_id=%s observed_at=%s", observation.check_id, started)
        except Exception:
            logger.exception("dream_pressure_history_failed observed_at=%s", started)

    backstop = False
    if not force:
        if too_soon(started, last_end):
            return None
        backstop = not pressure.should_sleep and pressure.is_idle and bool(candidates) \
            and overdue(started, last_start)
        if not pressure.should_sleep and not backstop:
            logger.info(
                "dream_cycle not due pressure=%.2f/%.2f idle=%s/%s new=%s counts=%s",
                pressure.pressure, pressure.threshold, pressure.idle_minutes,
                pressure.idle_required_minutes, pressure.new_counts, pressure.counts,
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
        note=f"pairs dream={len(d_pairs)} control={len(c_pairs)}"
        + (f" | overdue: {settings.DREAM_LOOKBACK_HOURS:g} h without crossing threshold" if backstop else ""),
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
