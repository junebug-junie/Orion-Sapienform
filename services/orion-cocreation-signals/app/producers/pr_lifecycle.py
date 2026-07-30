"""pr_lifecycle producer: polls this repo's real GitHub PR history on an
interval, publishes a ``CodebaseDeltaV1(domain="pr_lifecycle")`` event every
tick -- unlike git_delta/graph_delta, a time window always has a real,
well-defined answer (even an all-zero one), so "checked and found nothing"
is a genuine observation this domain should report, not a no-op to skip.

Half-open ``[since, until)`` windows tile contiguously tick-to-tick (this
tick's ``until`` becomes the next tick's ``since``) -- the dedup mechanism
named in ``orion/structural_mass/pr_lifecycle.py::pr_lifecycle_delta``'s own
docstring, not a per-event store.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.codebase_delta import CodebaseDeltaV1, PrLifecycleDeltaPayloadV1
from orion.structural_mass.pr_lifecycle import fetch_recent_prs, pr_lifecycle_delta

logger = logging.getLogger("orion.cocreation_signals.pr_lifecycle")


async def _publish(bus: OrionBusAsync, channel: str, source: ServiceRef, delta) -> bool:
    """Returns True if communicated (real publish or bus intentionally
    disabled), False only on a real transient publish failure -- see
    ``git_delta._publish()``'s docstring for why the caller must not advance
    ``last_until`` on False (a failed publish's window would otherwise be
    silently skipped rather than folded into the next tick's window)."""
    if not getattr(bus, "enabled", False):
        logger.info("cocreation_pr_lifecycle_publish_skipped_bus_disabled")
        return True
    event = CodebaseDeltaV1(
        domain="pr_lifecycle",
        observed_at=datetime.now(timezone.utc),
        pr_lifecycle=PrLifecycleDeltaPayloadV1(
            since=delta.since,
            until=delta.until,
            submitted_count=delta.submitted_count,
            merged_count=delta.merged_count,
            closed_without_merge_count=delta.closed_without_merge_count,
            submitted_numbers=list(delta.submitted_numbers),
            merged_numbers=list(delta.merged_numbers),
            closed_without_merge_numbers=list(delta.closed_without_merge_numbers),
            possibly_truncated=delta.possibly_truncated,
        ),
    )
    envelope = BaseEnvelope(
        kind="substrate.codebase_delta.v1", source=source, payload=event.model_dump(mode="json")
    )
    try:
        await bus.publish(channel, envelope)
        logger.info(
            "cocreation_pr_lifecycle_published submitted=%d merged=%d closed_without_merge=%d "
            "possibly_truncated=%s",
            delta.submitted_count, delta.merged_count, delta.closed_without_merge_count,
            delta.possibly_truncated,
        )
        if delta.possibly_truncated:
            logger.warning(
                "cocreation_pr_lifecycle_possibly_truncated -- fetch_limit may not reach "
                "far enough back for this window; counts may under-report"
            )
        return True
    except Exception:
        logger.exception("cocreation_pr_lifecycle_publish_failed")
        return False


async def pr_lifecycle_loop(
    *,
    bus: OrionBusAsync,
    channel: str,
    source: ServiceRef,
    owner: str,
    repo: str,
    fetch_limit: int,
    poll_interval_sec: float,
    cold_start_lookback_sec: float,
    stop: asyncio.Event,
) -> None:
    """**Real, acknowledged data-loss gap on restart, unlike git_delta/
    graph_delta (code review 2026-07-30):** a PR event entirely contained
    within a downtime window longer than ``cold_start_lookback_sec`` is gone
    forever -- not recovered in a bigger diff next time, since a time
    *window* (not a diff between two saved states) has no way to know how
    far back "since the container last ran" actually was. Kept as a
    dedicated, generous setting (independent of the steady-state
    ``poll_interval_sec``) specifically so a restart doesn't silently narrow
    the real recovery window to whatever the steady-state poll cadence
    happens to be -- but it is still a fixed cap, not a persisted "last
    known window," and callers should not assume parity with the other two
    producers' true self-healing story.
    """
    last_until = datetime.now(timezone.utc) - timedelta(seconds=cold_start_lookback_sec)
    while not stop.is_set():
        try:
            until = datetime.now(timezone.utc)
            prs = await asyncio.to_thread(fetch_recent_prs, owner, repo, limit=fetch_limit)
            delta = pr_lifecycle_delta(prs, since=last_until, until=until, fetch_limit=fetch_limit)
            published = await _publish(bus, channel, source, delta)
            if published:
                last_until = until
            # else: leave last_until unchanged so the next tick's window
            # extends to cover this failed range too, instead of silently
            # dropping it (same reasoning as git_delta_loop/graph_delta_loop).
        except Exception:
            logger.exception("cocreation_pr_lifecycle_tick_failed")
        try:
            await asyncio.wait_for(stop.wait(), timeout=poll_interval_sec)
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            break
