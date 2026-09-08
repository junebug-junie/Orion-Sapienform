"""claude_limit producer: publish what Claude's own rate-limit messages say.

`orion/dev_economics/rate_limit_events.py` has been correct and completely
unconsumed since it shipped -- `grep` for it across `orion/`, `services/` and
`scripts/` finds only its own module. This loop is the consumer that gives it
a wire. It scans the same read-only `~/.claude/projects` mount
`dev_economics_loop` and `affective_state_loop` already use.

ONE EVENT PER (KIND, WINDOW), NOT PER WINDOW -- AND THE KIND IS THE FIX.
Claude announces two independent limits, `session_limit` and `weekly_limit`,
drawn from two different pools. `LimitObservation.state` keys off the
chronologically LAST event in the window **regardless of kind**, so a weekly
limit still in force until Friday reads `clear` the moment a session limit
bound an hour ago and already reset -- the newer, spent event masks the older,
binding one. Caught in review before this shipped, and it defeated the exact
reason the wide window was added.

So each spec names its kind, and `_to_event` filters the observation's events
to that kind before reading `state`, `resets_at` and `event_count` off it. The
kind rides on the wire too, because a consumer that cannot tell the two pools
apart would re-merge them downstream.

STATELESS, UNLIKE ITS NEIGHBOURS. `dev_economics_loop` and `git_delta_loop`
publish a *delta* against an in-process baseline and therefore need a cold
start and a "don't advance the baseline on a failed publish" contract. This
one does not: every tick is an independent scan of a trailing window, so a
failed publish loses one observation and the next tick is complete on its own.
There is deliberately no cold-start tick to suppress.

ONE TASK PER SPEC, EACH WITH ITS OWN CADENCE. Measured on the real tree, a 5h
window reads 42 files / 110MB while a 168h window reads 429 files / 462MB and
2.4s of wall time, because `scan_window` does `read_text()` + `splitlines()`
per file. Sharing one 300s interval would re-read ~460MB every five minutes --
~165GB/day -- for an answer that moves on a day scale. The tight cadence is
justified by the session window's expiring reset time, which simply does not
apply to the weekly one, so each spec carries its own interval.

ABSENCE IS PUBLISHED, NOT SWALLOWED. A window with no observable transcript
activity publishes `state="unknown", observed=False` rather than nothing. A
consumer must be able to tell "the pool is full" from "the mount went away",
and a producer that goes quiet makes those identical -- the failure mode
CLAUDE.md 0A's prediction-error incidents exist to force out.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.async_service import OrionBusAsync
from orion.dev_economics.rate_limit_events import EventKind, LimitObservation, observe
from orion.schemas.claude_limit import ClaudeLimitObservationV1

logger = logging.getLogger("orion-cocreation-signals.claude_limit")

KIND = "substrate.claude_limit.v1"


class WindowSpec(NamedTuple):
    """One published series: which limit, over how long, how often."""

    kind: EventKind
    window_hours: float
    interval_sec: float


def _for_kind(obs: LimitObservation, kind: EventKind) -> LimitObservation:
    """The same observation with its events narrowed to one limit kind.

    `dataclasses.replace` rather than a hand-built copy so a new field on
    `LimitObservation` is carried automatically instead of being silently
    dropped here. Only `events` changes -- the activity timestamps are about
    the transcript, not about either pool, and `state`'s "real activity after
    the event" recovery path needs them intact.
    """
    return dataclasses.replace(obs, events=tuple(e for e in obs.events if e.kind == kind))


def _to_event(
    obs: LimitObservation, *, kind: EventKind, observed_at: datetime
) -> ClaudeLimitObservationV1:
    """Map the kind-narrowed observation field-for-field. Every derived value
    comes from the observation's own property, never recomputed here -- `state`
    encodes two independent ways a limit lifts (a stated reset time passing, or
    real activity after the event) and a second implementation of that would
    drift from the one that was debugged live."""
    scoped = _for_kind(obs, kind)
    return ClaudeLimitObservationV1(
        observed_at=observed_at,
        kind=kind,
        window_hours=scoped.window_hours,
        window_start=scoped.window_start,
        window_end=scoped.window_end,
        state=scoped.state,
        observed=scoped.observed,
        event_count=scoped.event_count,
        resets_at=scoped.resets_at,
        seconds_until_reset=scoped.seconds_until_reset,
        staleness_sec=scoped.staleness_sec,
        observed_message_count=scoped.observed_message_count,
        scanned_file_count=scoped.scanned_file_count,
    )


async def _publish(
    bus: OrionBusAsync, channel: str, source: ServiceRef, event: ClaudeLimitObservationV1
) -> None:
    if not getattr(bus, "enabled", False):
        logger.info("cocreation_claude_limit_publish_skipped_bus_disabled")
        return
    envelope = BaseEnvelope(kind=KIND, source=source, payload=event.model_dump(mode="json"))
    try:
        await bus.publish(channel, envelope)
        logger.info(
            "cocreation_claude_limit_published kind=%s window_hours=%s state=%s observed=%s "
            "event_count=%d resets_at=%s staleness_sec=%s",
            event.kind, event.window_hours, event.state, event.observed,
            event.event_count, event.resets_at, event.staleness_sec,
        )
    except Exception:
        logger.exception(
            "cocreation_claude_limit_publish_failed kind=%s window_hours=%s",
            event.kind, event.window_hours,
        )


async def _spec_loop(
    *,
    bus: OrionBusAsync,
    channel: str,
    source: ServiceRef,
    claude_projects_path: str,
    spec: WindowSpec,
    stop: asyncio.Event,
) -> None:
    """One series, on its own cadence. Isolated per spec so a scan that raises
    costs only its own series, not the tick's other observation."""
    while not stop.is_set():
        try:
            observed_at = datetime.now(timezone.utc)
            obs = await asyncio.to_thread(
                observe, window_hours=spec.window_hours, root=claude_projects_path
            )
            await _publish(bus, channel, source, _to_event(obs, kind=spec.kind, observed_at=observed_at))
        except Exception:
            logger.exception(
                "cocreation_claude_limit_tick_failed kind=%s window_hours=%s",
                spec.kind, spec.window_hours,
            )
        try:
            await asyncio.wait_for(stop.wait(), timeout=spec.interval_sec)
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            break


async def claude_limit_loop(
    *,
    bus: OrionBusAsync,
    channel: str,
    source: ServiceRef,
    claude_projects_path: str,
    specs: tuple[WindowSpec, ...],
    stop: asyncio.Event,
) -> None:
    if not Path(claude_projects_path).exists():
        # Fail loud once at startup rather than publishing a stream of empty
        # ticks that would misreport a broken mount as a quiet window. Note
        # this is the one case the loop cannot report as `unknown` on the bus,
        # because it never starts -- hence the error log rather than a silent
        # return.
        logger.error(
            "cocreation_claude_limit_claude_projects_path_missing path=%s -- "
            "producer will not start; check COCREATION_SIGNALS_CLAUDE_PROJECTS_HOST_PATH",
            claude_projects_path,
        )
        return
    if not specs:
        logger.error("cocreation_claude_limit_no_windows_configured -- producer will not start")
        return

    await asyncio.gather(*[
        _spec_loop(
            bus=bus, channel=channel, source=source,
            claude_projects_path=claude_projects_path, spec=spec, stop=stop,
        )
        for spec in specs
    ])
