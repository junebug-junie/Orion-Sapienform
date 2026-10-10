"""Hub's reader for Orion's rest drive (Temporal Self rev 4, R2).

orion-dream writes its latest sleep-pressure reading to Redis
`orion:drive:rest:latest` at every 600 s check (orion/schemas/drive_reading.py).
Curiosity and outreach each own one of these readers, with their own flag and
multiplier, so either can be switched off alone.

The only effect: while the drive reads `due` (Orion is tired and waiting to
sleep), the owning loop's cooldown is multiplied (default 2x). Absent, expired,
unparseable, `no_reading`, or flag off is UNKNOWN, and UNKNOWN changes nothing.
A Redis failure is UNKNOWN too -- tiredness must never freeze a loop.

`refresh()` is async (one Redis GET per tick); `view()` / `cooldown_sec()` are
sync and re-judge staleness at the caller's `now`, so a reader that stops being
refreshed decays to UNKNOWN on its own instead of staying tired.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from orion.regulation.rest_drive import (
    UNKNOWN_DISABLED,
    RestDriveView,
    eased_cooldown_sec,
    rest_drive_view,
)
from orion.schemas.drive_reading import REST_DRIVE_REDIS_KEY

logger = logging.getLogger("orion-hub.rest-drive")


class RestDriveReader:
    def __init__(self, *, enabled: bool, multiplier: float, max_age_sec: float, name: str) -> None:
        self.enabled = bool(enabled)
        self.multiplier = float(multiplier)
        self.max_age_sec = float(max_age_sec)
        self.name = name
        self._raw: Any = None
        self._last_reason: Optional[str] = None

    async def refresh(self, bus: Any, now: Optional[datetime] = None) -> RestDriveView:
        if not self.enabled:
            self._raw = None
            return UNKNOWN_DISABLED
        redis = getattr(bus, "redis", None)
        raw = None
        if redis is not None:
            try:
                raw = await redis.get(REST_DRIVE_REDIS_KEY)
            except Exception:  # noqa: BLE001 -- unreadable is unknown, never tired
                logger.warning("rest_drive_read_failed reader=%s -- treating as unknown", self.name, exc_info=True)
                raw = None
        self._raw = raw
        return self.view(now)

    def view(self, now: Optional[datetime] = None) -> RestDriveView:
        if not self.enabled:
            return UNKNOWN_DISABLED
        now = now or datetime.now(timezone.utc)
        v = rest_drive_view(self._raw, now=now, max_age_sec=self.max_age_sec)
        if v.reason != self._last_reason:
            # One line per change, not per tick: the trace of when this loop
            # started and stopped easing off.
            logger.info(
                "rest_drive_view reader=%s verdict=%s reason=%s state=%s level=%s age_sec=%s source_ref=%s",
                self.name, v.verdict, v.reason, v.state, v.level, v.age_sec, v.source_ref,
            )
            self._last_reason = v.reason
        return v

    def cooldown_sec(self, base_sec: float, now: Optional[datetime] = None) -> Optional[float]:
        """The stretched cooldown while tired, else None (no change)."""
        if not self.enabled:
            return None
        return eased_cooldown_sec(base_sec, self.view(now), self.multiplier)


__all__ = ["RestDriveReader"]
