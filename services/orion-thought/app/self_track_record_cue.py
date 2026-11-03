"""Orion's track record for stance_react and reverie (self-monitor organ, spec step 3).

What Orion is failing at or misjudging right now, from their own settled bets: a few plain sentences
rendered by ``orion.self_monitor.cue.project_track_record_cue`` (fixed templates, no LLM) from the
self-monitor state orion-durable-runs publishes hourly to Redis ``orion:self_monitor:latest``
(``orion.self_monitor.view.SELF_MONITOR_REDIS_KEY``) on ``ORION_BUS_URL``. Read-only.

Never on the caller's path: the Redis value is read at most every ``REFRESH_SEC`` in a background
thread (and once at startup warm); a turn or reverie tick only re-judges the cached value at its own
``now``, so staleness is always judged by the state's own ``computed_at`` (the reader contract: an
absent, unparseable, stale or future-stamped state renders nothing). A failed read keeps the last
value it got; the age gate still blanks it once it is too old.

Fail-open: every miss renders "" and logs ``self_track_record_skipped reason=...`` with the caller's
correlation id. Kill switch ``SELF_TRACK_RECORD_CUE_ENABLED=false``: both prompts exactly as before.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Optional

from orion.schemas.self_monitor import SelfTrackRecordCueV1

from .settings import settings

logger = logging.getLogger("orion-thought.self_track_record")

# The producer writes hourly; re-reading every 5 min keeps a fresh state at most ~5 min late.
REFRESH_SEC = 300.0
_SOCKET_TIMEOUT_SEC = 2.0

_lock = threading.Lock()
# (monotonic fetched_at, last raw value or None, last read error or None)
_cache: Optional[tuple[float, object, Optional[str]]] = None
_refreshing = False


def fetch_raw() -> object:
    """Seam for tests. The raw Redis value (None when the key is absent). Raises on a read error."""
    import redis

    from orion.self_monitor.view import SELF_MONITOR_REDIS_KEY

    client = redis.Redis.from_url(
        settings.orion_bus_url, socket_timeout=_SOCKET_TIMEOUT_SEC, socket_connect_timeout=_SOCKET_TIMEOUT_SEC,
    )
    try:
        return client.get(SELF_MONITOR_REDIS_KEY)
    finally:
        client.close()


def refresh() -> None:
    """Blocking read into the cache. Never raises. A failed read keeps the previous value."""
    global _cache, _refreshing
    try:
        raw, error = fetch_raw(), None
    except Exception as exc:  # noqa: BLE001 -- unread state is "nothing to say", never a guess
        logger.debug("self_track_record_fetch_failed err=%s: %s", type(exc).__name__, exc)
        raw, error = None, type(exc).__name__
    with _lock:
        if error is not None and _cache is not None and _cache[1] is not None:
            raw = _cache[1]
        _cache = (time.monotonic(), raw, error)
        _refreshing = False


def _cached() -> Optional[tuple[float, object, Optional[str]]]:
    """The cached read, starting one background refresh when it is missing or older than REFRESH_SEC."""
    global _refreshing
    with _lock:
        cached = _cache
        stale = cached is None or time.monotonic() - cached[0] >= REFRESH_SEC
        start = stale and not _refreshing
        if start:
            _refreshing = True
    if start:
        threading.Thread(target=refresh, name="self-track-record", daemon=True).start()
    return cached


def render(now: datetime) -> SelfTrackRecordCueV1:
    """Cached state -> cue at ``now``. Never reads Redis on the caller's thread."""
    from orion.self_monitor.cue import project_track_record_cue
    from orion.self_monitor.view import self_monitor_view

    cached = _cached()
    if cached is None:
        return SelfTrackRecordCueV1(rendered_at=now, skip_reason="not_loaded")
    _, raw, error = cached
    if raw is None and error is not None:
        return SelfTrackRecordCueV1(rendered_at=now, skip_reason="fetch_failed")
    return project_track_record_cue(self_monitor_view(raw, now=now), now=now)


def warm() -> None:
    """First read at startup (in a thread), so the first turn can already show the cue. Never raises."""
    try:
        refresh()
        cue = render(datetime.now(timezone.utc))
        logger.info("self_track_record_warm reason=%s chars=%s", cue.skip_reason, len(cue.text))
    except Exception as exc:  # noqa: BLE001
        logger.warning("self_track_record_warm_failed err=%s: %s", type(exc).__name__, exc)


async def build_self_track_record_cue(
    correlation_id: str, *, purpose: str, now: datetime | None = None,
) -> SelfTrackRecordCueV1:
    """Always returns a cue; ``text`` is "" when nothing may be shown. Never raises, never waits on I/O."""
    now = now or datetime.now(timezone.utc)
    if not settings.self_track_record_cue_enabled:
        cue = SelfTrackRecordCueV1(rendered_at=now, skip_reason="disabled")
    else:
        try:
            cue = render(now)
        except Exception as exc:  # noqa: BLE001 -- the cue must never fail a turn or a reverie tick
            logger.warning("self_track_record_render_failed corr=%s err=%s: %s",
                           correlation_id, type(exc).__name__, exc)
            cue = SelfTrackRecordCueV1(rendered_at=now, skip_reason="render_failed")
    if cue.text:
        logger.info("self_track_record corr=%s purpose=%s chars=%s shown=%s dropped=%s age_sec=%s",
                    correlation_id, purpose, len(cue.text), ",".join(cue.shown), cue.lines_dropped,
                    cue.state_age_sec)
    else:
        logger.info("self_track_record_skipped corr=%s purpose=%s reason=%s age_sec=%s",
                    correlation_id, purpose, cue.skip_reason, cue.state_age_sec)
    return cue
