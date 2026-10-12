"""Fetch Orion's current day frame and render the stance cue (Temporal Self patch 4).

The unified turn's stance step (``stance_react``) is the only chat path that gets this cue
(Juniper, 2026-10-11: the production chat path). Read over HTTP from orion-durable-runs'
existing ``GET /temporal-self/frame`` -- the chronicle's own read route, the same seam the Hub
already uses for durable runs -- rather than a direct DB read: orion-thought holds no DSN for the
chronicle tables, and the route returns exactly the validated ``TemporalSelfFrameV1``.

Fail-open and bounded: a short timeout (``TEMPORAL_SELF_CUE_FETCH_TIMEOUT_SEC``); any failure,
a frame for another day, or one older than ``TEMPORAL_SELF_CUE_MAX_AGE_SEC`` renders nothing
(``text == ""``) and logs ``temporal_self_cue_skipped reason=...`` with the turn's correlation id.
The fetch runs concurrently with Mind enrichment, so it adds no wait to the turn unless Mind is off.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from orion.schemas.temporal_self import TemporalSelfStanceCueV1
from orion.temporal_self.stance_cue import project_stance_cue

from .settings import settings

logger = logging.getLogger("orion-thought.temporal_self_cue")


def _transport() -> httpx.AsyncBaseTransport | None:
    """Seam for tests to inject an httpx.MockTransport. None in production."""
    return None


async def fetch_frame(base_url: str, timeout_sec: float) -> dict[str, Any] | None:
    try:
        async with httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=httpx.Timeout(timeout_sec), transport=_transport(),
        ) as client:
            resp = await client.get("/temporal-self/frame")
    except httpx.HTTPError as exc:
        logger.debug("temporal_self_frame_fetch_failed err=%s: %s", type(exc).__name__, exc)
        return None
    if resp.status_code != 200:
        return None
    try:
        body = resp.json()
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


async def build_stance_cue(correlation_id: str, *, now: datetime | None = None) -> TemporalSelfStanceCueV1:
    """Always returns a cue; ``text`` is "" when nothing may be shown. Never raises."""
    now = now or datetime.now(timezone.utc)
    if not settings.temporal_self_stance_cue_enabled:
        return TemporalSelfStanceCueV1(rendered_at=now, skip_reason="disabled")
    started = time.perf_counter()
    timeout = settings.temporal_self_cue_fetch_timeout_sec
    try:
        # httpx's timeout is per phase (connect, each read; DNS not covered): bound the whole fetch.
        frame = await asyncio.wait_for(fetch_frame(settings.temporal_self_frame_url, timeout), timeout)
    except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001 -- the cue must never fail a turn
        logger.debug("temporal_self_frame_fetch_failed corr=%s err=%s", correlation_id, type(exc).__name__)
        frame = None
    try:
        cue = project_stance_cue(frame, now, max_age_sec=settings.temporal_self_cue_max_age_sec)
    except Exception as exc:  # noqa: BLE001 -- a render bug is named as one, never as a network problem
        logger.warning("temporal_self_cue_render_failed corr=%s err=%s: %s", correlation_id, type(exc).__name__, exc)
        cue = TemporalSelfStanceCueV1(rendered_at=now, skip_reason="render_failed")
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 1)
    if cue.text:
        logger.info("temporal_self_cue corr=%s day=%s chars=%s arcs=%s dropped=%s fetch_ms=%s",
                    correlation_id, cue.day_id, len(cue.text), len(cue.arc_ids), cue.clauses_dropped, elapsed_ms)
    else:
        logger.info("temporal_self_cue_skipped corr=%s reason=%s day=%s fetch_ms=%s",
                    correlation_id, cue.skip_reason, cue.day_id, elapsed_ms)
    return cue
