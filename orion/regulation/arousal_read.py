"""Fetch Orion's arousal reading for one decision (Temporal Self rev 4, R3 readers 4-6).

One Redis GET of ``orion:regulation:latest``, judged by ``arousal_view`` at the caller's
``now``. Never raises: a disabled flag, no URL, a Redis error, a timeout, or a stale /
malformed value all come back as ``unknown``, and ``unknown`` changes nothing.

Readers call this at the moment they would act (not at loop start), so a strain that began
during a long cycle is seen, and a strain that ended is not acted on (#2628's lesson).

What a level *does* stays in each reader's own gate; this module only fetches.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from orion.regulation.arousal_view import (
    DEFAULT_MAX_AGE_SEC,
    UNKNOWN_DISABLED,
    ArousalView,
    arousal_view,
)
from orion.schemas.regulation import REGULATION_STATE_REDIS_KEY

logger = logging.getLogger("orion.regulation.arousal_read")

# One GET must never stall a reader's loop.
READ_TIMEOUT_SEC = 2.0


def _judge(raw: Any, *, now: Optional[datetime], max_age_sec: float, reader: str) -> ArousalView:
    view = arousal_view(raw, now=now or datetime.now(timezone.utc), max_age_sec=max_age_sec)
    logger.info(
        "arousal_view reader=%s level=%s reason=%s since=%s age_sec=%s",
        reader, view.level, view.reason, view.since, view.age_sec,
    )
    return view


def read_arousal_sync(
    redis_url: Optional[str],
    *,
    enabled: bool,
    reader: str,
    now: Optional[datetime] = None,
    max_age_sec: float = DEFAULT_MAX_AGE_SEC,
    timeout_sec: float = READ_TIMEOUT_SEC,
    client: Any = None,
) -> ArousalView:
    """Blocking read for code that already runs off the event loop. ``client`` is for tests."""
    if not enabled:
        return UNKNOWN_DISABLED
    raw = None
    try:
        if client is None:
            if not redis_url:
                return ArousalView(level="unknown", reason="no_redis_url")
            import redis  # local: only readers that are switched on need it

            client = redis.Redis.from_url(
                redis_url, socket_timeout=timeout_sec, socket_connect_timeout=timeout_sec
            )
            try:
                raw = client.get(REGULATION_STATE_REDIS_KEY)
            finally:
                try:
                    client.close()
                except Exception:  # noqa: BLE001
                    pass
        else:
            raw = client.get(REGULATION_STATE_REDIS_KEY)
    except Exception:  # noqa: BLE001 -- unreadable is unknown, never strained
        logger.warning("arousal_read_failed reader=%s -- treating as unknown", reader, exc_info=True)
        return ArousalView(level="unknown", reason="read_failed")
    return _judge(raw, now=now, max_age_sec=max_age_sec, reader=reader)


async def read_arousal_async(
    bus: Any,
    *,
    enabled: bool,
    reader: str,
    now: Optional[datetime] = None,
    max_age_sec: float = DEFAULT_MAX_AGE_SEC,
    timeout_sec: float = READ_TIMEOUT_SEC,
) -> ArousalView:
    """Read through an already-connected ``OrionBusAsync`` (its ``redis`` property)."""
    if not enabled:
        return UNKNOWN_DISABLED
    try:
        # Inside the try: OrionBusAsync.redis RAISES on a bus that is not connected.
        redis = getattr(bus, "redis", None)
        if redis is None:
            return ArousalView(level="unknown", reason="no_redis")
        raw = await asyncio.wait_for(redis.get(REGULATION_STATE_REDIS_KEY), timeout=timeout_sec)
    except Exception:  # noqa: BLE001 -- includes TimeoutError; unknown, never strained
        logger.warning("arousal_read_failed reader=%s -- treating as unknown", reader, exc_info=True)
        return ArousalView(level="unknown", reason="read_failed")
    return _judge(raw, now=now, max_age_sec=max_age_sec, reader=reader)


__all__ = ["READ_TIMEOUT_SEC", "read_arousal_async", "read_arousal_sync"]
