"""Fetch Orion's personality frame and render the read-back cue (personality graph, spec rev 2).

What Orion keeps coming back to, in Orion's own words, read back as a RECORD, not an instruction:
``orion.personality.cue.project_personality_cue`` is a fixed template (no LLM, no tone/style
directives). Read over HTTP from orion-durable-runs ``GET /personality/frame`` (same service and
base URL as the Temporal Self frame), mirroring ``app/temporal_self_cue.py``.

Fail-open and bounded: any failure, an invalid frame, a frame older than
``PERSONALITY_CUE_MAX_AGE_SEC``, or no active node renders nothing (``text == ""``). Shown to the
unified turn's stance step (fetched alongside Mind, so no added serial wait) and to reverie (one
fetch per reverie step, alongside its other reads). One log line per build:
``personality_cue corr=<id> rendered=<bool> ids=<...> skip=<reason>``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from orion.personality.cue import project_personality_cue
from orion.schemas.personality import PersonalityCueV1

from .settings import settings

logger = logging.getLogger("orion-thought.personality_cue")


def _transport() -> httpx.AsyncBaseTransport | None:
    """Seam for tests to inject an httpx.MockTransport. None in production."""
    return None


async def fetch_personality_frame(base_url: str, timeout_sec: float) -> dict[str, Any] | None:
    try:
        async with httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=httpx.Timeout(timeout_sec), transport=_transport(),
        ) as client:
            resp = await client.get("/personality/frame")
    except httpx.HTTPError as exc:
        logger.debug("personality_frame_fetch_failed err=%s: %s", type(exc).__name__, exc)
        return None
    if resp.status_code != 200:
        return None
    try:
        body = resp.json()
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


async def fetch_personality_frame_bounded(timeout_sec: float, correlation_id: str = "") -> dict[str, Any] | None:
    """The current frame, or None. Never raises, never takes longer than ``timeout_sec``."""
    try:
        # httpx's timeout is per phase: bound the whole fetch (DNS included).
        return await asyncio.wait_for(
            fetch_personality_frame(settings.personality_frame_url, timeout_sec), timeout_sec
        )
    except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001 -- a reader must never fail its caller
        logger.debug("personality_frame_fetch_failed corr=%s err=%s", correlation_id, type(exc).__name__)
        return None


def _log(correlation_id: str, cue: PersonalityCueV1, elapsed_ms: float, skip: str | None = None) -> None:
    logger.info(
        "personality_cue corr=%s rendered=%s ids=%s skip=%s chars=%s fetch_ms=%s",
        correlation_id, bool(cue.text), ",".join(cue.disposition_ids) or "-",
        skip or cue.skip_reason or "-", len(cue.text), elapsed_ms,
    )


async def build_personality_cue(
    correlation_id: str, *, now: datetime | None = None, timeout_sec: float | None = None,
) -> PersonalityCueV1:
    """Always returns a cue; ``text`` is "" when nothing may be shown. Never raises."""
    now = now or datetime.now(timezone.utc)
    started = time.perf_counter()
    if not settings.personality_cue_enabled:
        cue = PersonalityCueV1(rendered_at=now, skip_reason="disabled")
        _log(correlation_id, cue, 0.0)
        return cue
    frame = await fetch_personality_frame_bounded(
        settings.personality_cue_timeout_sec if timeout_sec is None else timeout_sec, correlation_id
    )
    try:
        cue = project_personality_cue(
            frame, now=now, max_age_sec=settings.personality_cue_max_age_sec,
            top_k=settings.personality_cue_top_k,
        )
    except Exception as exc:  # noqa: BLE001 -- the renderer promises never to raise; named if it does
        logger.warning("personality_cue_render_failed corr=%s err=%s: %s", correlation_id, type(exc).__name__, exc)
        cue = PersonalityCueV1(rendered_at=now, skip_reason="render_failed")
    _log(correlation_id, cue, round((time.perf_counter() - started) * 1000.0, 1))
    return cue
