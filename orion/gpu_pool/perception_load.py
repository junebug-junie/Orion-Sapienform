"""Is Orion's sight using far more GPU than it normally does? (body-economy design Patch A0)

``perception`` leases (vision-council's ``metacog_perception`` route) are never shed by heat --
that is the point: Orion keeps seeing while the cabinet runs hot. The price of that exemption is
that perception's own load must not quietly grow into a heat source. This tracker sums the GPU
slot-seconds perception leases held over the last hour; the runtime raises an attention card when
the sum passes a ceiling and an info card when it falls back under.

Ceiling default (``GPU_POOL_PERCEPTION_ALERT_HELD_SEC_PER_HOUR=3360``): 2x the 14-day p90 of
vision-council's hourly held slot-seconds on 2026-10-23 (p90 1,679; p99 2,120; max 2,811 --
``gpu_pool_events`` released rows, 337 hours). Pure: no clock, no I/O. Memory only: a restart
forgets the window (the alert re-arms within an hour of a still-high load).
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta

WINDOW = timedelta(hours=1)


class PerceptionLoad:
    def __init__(self, ceiling_sec: float) -> None:
        self.ceiling_sec = float(ceiling_sec)
        self._held: deque[tuple[datetime, float]] = deque()
        self.alerted_at: datetime | None = None

    def record(self, at: datetime, held_sec: float) -> None:
        if held_sec > 0:
            self._held.append((at, float(held_sec)))

    def held_last_hour(self, now: datetime) -> float:
        while self._held and now - self._held[0][0] > WINDOW:
            self._held.popleft()
        return sum(s for _, s in self._held)

    def check(self, now: datetime, realert: timedelta) -> str | None:
        """``"high"`` when the hour is over the ceiling and no card was sent within ``realert``;
        ``"recovered"`` the first check back under after a ``"high"``; else ``None``."""
        held = self.held_last_hour(now)
        if held > self.ceiling_sec:
            if self.alerted_at is None or now - self.alerted_at >= realert:
                self.alerted_at = now
                return "high"
            return None
        if self.alerted_at is not None:
            self.alerted_at = None
            return "recovered"
        return None
