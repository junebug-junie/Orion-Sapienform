"""claude_limit producer: publish what Claude's own rate-limit messages say.

`orion/dev_economics/rate_limit_events.py` has been correct and completely
unconsumed since it shipped -- `grep` for it across `orion/`, `services/` and
`scripts/` finds only its own module. This loop is the consumer that gives it
a wire. It scans the same read-only `~/.claude/projects` mount
`dev_economics_loop` and `affective_state_loop` already use, and publishes one
`ClaudeLimitObservationV1` per configured trailing window per tick.

STATELESS, UNLIKE ITS NEIGHBOURS. `dev_economics_loop` and `git_delta_loop`
publish a *delta* against an in-process baseline and therefore need a cold
start and a "don't advance the baseline on a failed publish" contract. This
one does not: every tick is an independent scan of a trailing window, so a
failed publish loses one observation and the next tick is complete on its own.
There is deliberately no cold-start tick to suppress.

TWO WINDOWS, BECAUSE THE CONSTRAINT HAS TWO SHAPES. The real messages are
`session limit` and `weekly limit`, so a single window cannot represent both:
a 5h scan cannot see a weekly limit that bound on Tuesday, and a 168h scan
reports `event_count` for a week when the question is whether Orion may speak
in the next minute. Both are published, distinguished by `window_hours`, and
a consumer picks the one its question needs.

ABSENCE IS PUBLISHED, NOT SWALLOWED. A window with no observable transcript
activity publishes `state="unknown", observed=False` rather than nothing. A
consumer must be able to tell "the pool is full" from "the mount went away",
and a producer that goes quiet makes those identical -- the failure mode
CLAUDE.md §0A's prediction-error incidents exist to force out.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.async_service import OrionBusAsync
from orion.dev_economics.rate_limit_events import LimitObservation, observe
from orion.schemas.claude_limit import ClaudeLimitObservationV1

logger = logging.getLogger("orion-cocreation-signals.claude_limit")

KIND = "substrate.claude_limit.v1"


def _to_event(obs: LimitObservation, *, observed_at: datetime) -> ClaudeLimitObservationV1:
    """Map the observation field-for-field. Every derived value comes from the
    observation's own property, never recomputed here -- `state` in particular
    encodes two independent ways a limit lifts (stated reset passed, or real
    activity after the event) and a second implementation of that would drift.
    """
    return ClaudeLimitObservationV1(
        observed_at=observed_at,
        window_hours=obs.window_hours,
        window_start=obs.window_start,
        window_end=obs.window_end,
        state=obs.state,
        observed=obs.observed,
        event_count=obs.event_count,
        resets_at=obs.resets_at,
        seconds_until_reset=obs.seconds_until_reset,
        staleness_sec=obs.staleness_sec,
        observed_message_count=obs.observed_message_count,
        scanned_file_count=obs.scanned_file_count,
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
            "cocreation_claude_limit_published window_hours=%s state=%s observed=%s "
            "event_count=%d resets_at=%s staleness_sec=%s",
            event.window_hours, event.state, event.observed,
            event.event_count, event.resets_at, event.staleness_sec,
        )
    except Exception:
        logger.exception("cocreation_claude_limit_publish_failed window_hours=%s", event.window_hours)


async def claude_limit_loop(
    *,
    bus: OrionBusAsync,
    channel: str,
    source: ServiceRef,
    claude_projects_path: str,
    window_hours: tuple[float, ...],
    poll_interval_sec: float,
    stop: asyncio.Event,
) -> None:
    from pathlib import Path

    if not Path(claude_projects_path).exists():
        # Same fail-loud-once posture as dev_economics_loop: a missing mount
        # must not masquerade as a window with nothing in it. Note this is the
        # one case the loop cannot report as `unknown` on the bus, because it
        # never starts -- hence the error log rather than a silent return.
        logger.error(
            "cocreation_claude_limit_claude_projects_path_missing path=%s -- "
            "producer will not start; check COCREATION_SIGNALS_CLAUDE_PROJECTS_HOST_PATH",
            claude_projects_path,
        )
        return
    if not window_hours:
        logger.error("cocreation_claude_limit_no_windows_configured -- producer will not start")
        return

    while not stop.is_set():
        for hours in window_hours:
            try:
                observed_at = datetime.now(timezone.utc)
                obs = await asyncio.to_thread(
                    observe, window_hours=hours, root=claude_projects_path
                )
                await _publish(bus, channel, source, _to_event(obs, observed_at=observed_at))
            except Exception:
                # Per-window, so one unparseable window does not cost the tick
                # its other observation.
                logger.exception("cocreation_claude_limit_tick_failed window_hours=%s", hours)
        try:
            await asyncio.wait_for(stop.wait(), timeout=poll_interval_sec)
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            break
