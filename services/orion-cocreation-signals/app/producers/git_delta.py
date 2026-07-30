"""git_delta producer: polls this repo's own HEAD on an interval, publishes a
``CodebaseDeltaV1(domain="git")`` event on real change only -- self-healing by
construction (a commit made on another node, via Cursor, or without a
post-commit hook firing just produces a bigger diff on the next check instead
of being lost; see the design spec's "Trigger shape" note).

State (``last_sha``) is caller-owned, kept in-process only -- a container
restart loses it, which just means the very next tick re-seeds the baseline
(cold start) instead of publishing a delta for whatever changed while the
container was down. Accepted simplification: real, self-correcting behavior
already exists for "missed a window" (the next real change just diffs a
bigger range), so losing one window's worth of granularity on a redeploy is
not a new failure mode, just a coarser instance of one already designed for.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from datetime import datetime, timezone

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.codebase_delta import CodebaseDeltaV1, GitDeltaPayloadV1
from orion.structural_mass.git_delta import git_churn_delta

logger = logging.getLogger("orion.cocreation_signals.git_delta")


def _current_head_sha(repo_path: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_path, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


async def _publish(bus: OrionBusAsync, channel: str, source: ServiceRef, delta) -> bool:
    """Returns True if the delta was successfully communicated (real publish,
    or the bus is intentionally disabled -- an operator choice, not data
    loss) and False only on a real transient publish failure. The caller
    uses this to decide whether ``last_sha`` may advance: advancing past a
    delta that failed to publish would permanently lose it (the next tick
    would diff from the new baseline, not retry the failed range) --
    contradicts the "self-healing by construction" story this domain is
    built around. Confirmed live via code review 2026-07-30: the first draft
    advanced state unconditionally after calling this function, silently
    undermining that guarantee for the one failure mode it's supposed to
    cover (a transient bus hiccup, not a restart)."""
    if not getattr(bus, "enabled", False):
        logger.info("cocreation_git_delta_publish_skipped_bus_disabled head_sha=%s", delta.head_sha)
        return True
    event = CodebaseDeltaV1(
        domain="git",
        observed_at=datetime.now(timezone.utc),
        git=GitDeltaPayloadV1(
            prev_sha=delta.prev_sha,
            head_sha=delta.head_sha,
            commit_count=delta.commit_count,
            files_added=delta.files_added,
            files_deleted=delta.files_deleted,
            files_modified=delta.files_modified,
            lines_added=delta.lines_added,
            lines_removed=delta.lines_removed,
        ),
    )
    envelope = BaseEnvelope(
        kind="substrate.codebase_delta.v1", source=source, payload=event.model_dump(mode="json")
    )
    try:
        await bus.publish(channel, envelope)
        logger.info(
            "cocreation_git_delta_published prev_sha=%s head_sha=%s lines_changed=%d commit_count=%d",
            delta.prev_sha, delta.head_sha, delta.lines_changed, delta.commit_count,
        )
        return True
    except Exception:
        logger.exception("cocreation_git_delta_publish_failed head_sha=%s", delta.head_sha)
        return False


async def git_delta_loop(
    *,
    bus: OrionBusAsync,
    channel: str,
    source: ServiceRef,
    repo_path: str,
    poll_interval_sec: float,
    stop: asyncio.Event,
) -> None:
    """One tick: cheap `git rev-parse HEAD` check against the last-seen SHA.
    No change -> explicit no-op (not a spurious delta, not decay). A change ->
    diff since last-seen SHA, however large, then publish."""
    last_sha: str | None = None
    while not stop.is_set():
        try:
            head_sha = await asyncio.to_thread(_current_head_sha, repo_path)
            if last_sha is None:
                last_sha = head_sha
                logger.info("cocreation_git_delta_cold_start head_sha=%s", head_sha)
            elif head_sha != last_sha:
                delta = await asyncio.to_thread(git_churn_delta, last_sha, head_sha, repo_path)
                published = await _publish(bus, channel, source, delta)
                if published:
                    last_sha = head_sha
                # else: leave last_sha unchanged -- the next tick's diff will
                # naturally cover this failed range too, instead of silently
                # losing it (see _publish()'s own docstring).
        except Exception:
            logger.exception("cocreation_git_delta_tick_failed")
        try:
            await asyncio.wait_for(stop.wait(), timeout=poll_interval_sec)
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            break
