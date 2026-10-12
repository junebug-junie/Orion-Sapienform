"""Hub's reader for Orion's arousal reading (Temporal Self rev 4, R3 readers).

orion-durable-runs writes one `RegulationStateV1` to Redis `orion:regulation:latest` every
regulate step (120 s and on every Juniper turn, 360 s TTL). Curiosity and outreach each own
one reader with their own flag, so either can be switched off alone.

What each owner does with a level is decided in that owner's own gate function, never here
(`scheduling_block_reason`, `outreach_block_reason`). This class only fetches and judges
freshness: absent, expired, unparseable, stale, producer-unknown, a Redis failure, or the
flag off is UNKNOWN, and UNKNOWN changes nothing.

Same shape as `scripts/rest_drive_reader.py`: `refresh()` is async (one Redis GET per tick,
2 s timeout); `view()` is sync and re-judges staleness at the caller's `now`, so a reader
that stops being refreshed decays to UNKNOWN on its own instead of staying strained.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from orion.regulation.arousal_view import UNKNOWN_DISABLED, ArousalView, arousal_view
from orion.schemas.regulation import REGULATION_STATE_REDIS_KEY

logger = logging.getLogger("orion-hub.arousal")

# One Redis GET must not stall a tick: the bus socket timeout is 60 s.
READ_TIMEOUT_SEC = 2.0


class ArousalReader:
    def __init__(self, *, enabled: bool, max_age_sec: float, name: str) -> None:
        self.enabled = bool(enabled)
        self.max_age_sec = float(max_age_sec)
        self.name = name
        self._raw: Any = None
        self._last_reason: Optional[str] = None

    async def refresh(self, bus: Any, now: Optional[datetime] = None) -> ArousalView:
        if not self.enabled:
            self._raw = None
            return UNKNOWN_DISABLED
        raw = None
        try:
            # Inside the try: OrionBusAsync.redis is a property that RAISES on a bus that is
            # not connected, which getattr's default does not catch.
            redis = getattr(bus, "redis", None)
            if redis is not None:
                raw = await asyncio.wait_for(redis.get(REGULATION_STATE_REDIS_KEY), timeout=READ_TIMEOUT_SEC)
        except Exception:  # noqa: BLE001 -- unreadable is unknown, never strained
            logger.warning("arousal_read_failed reader=%s -- treating as unknown", self.name, exc_info=True)
            raw = None
        self._raw = raw
        return self.view(now)

    def view(self, now: Optional[datetime] = None) -> ArousalView:
        if not self.enabled:
            return UNKNOWN_DISABLED
        now = now or datetime.now(timezone.utc)
        v = arousal_view(self._raw, now=now, max_age_sec=self.max_age_sec)
        if v.reason != self._last_reason:
            # One line per change, not per tick: the trace of when this loop's gain moved.
            logger.info(
                "arousal_view reader=%s level=%s reason=%s since=%s age_sec=%s reasons=%s",
                self.name, v.level, v.reason, v.since, v.age_sec, ",".join(v.reasons),
            )
            self._last_reason = v.reason
        return v

    def level(self, now: Optional[datetime] = None) -> Optional[str]:
        """The level a gate may act on, or None for unknown (None = today's behaviour)."""
        v = self.view(now)
        return v.level if v.known else None


__all__ = ["ArousalReader"]
