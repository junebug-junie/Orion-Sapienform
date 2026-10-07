"""Swap-load guards (stage 4 spec, "Guards (pool side)"): the physical check durable-runs'
elastic runtime made before borrowing gpu2 (``environment()`` in
services/orion-durable-runs/app/elastic_runtime.py, deleted in 4.5), read here so the pool is the
one decider.

- ``thermal``: the cabinet sensor through ``orion.autonomy.cabinet_heat.read_cabinet_heat`` -- the one
  owner of "how hot is the cabinet" and of what a missing reading means (thermal controller v2, D1/D10,
  docs/superpowers/specs/2026-10-06-thermal-controller-redesign-design.md). Blocks at ``hot`` (the
  32 C line, hysteresis to 30.5 C, unchanged by the 34 C reflex decision) and on ``unknown``: loading
  an extra model into a room nobody can measure is refused. Unlike before:
  - it starts ``unknown``, not ``hot`` (C13): the reason says there is no reading yet;
  - one failed or degraded read inside the grace window (300 s) holds the last state instead of
    blocking (C10): the guard keeps the readings it saw and re-judges them at ``now``.

``visual_baseline`` (thought ``/visual-chain/activity``: refuse a 27B load while the image baseline
was overdue) was deleted in stage 5.4. A reverie-visual run takes a diffusion hold on its own cadence
and reclaims gpu2 through the queue (owner reclaim), so the pool needs no side read of the chain.

Each guard is None when clear, else a short reason. A read that fails past grace is a reason
("unavailable: ..."), never a silent pass. Runs OUTSIDE the runtime lock: HTTP must not stall lease RPCs.
"""
from __future__ import annotations

import math
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from orion.autonomy.cabinet_heat import DEFAULT_READING_GRACE_SEC, CabinetHeatReading, read_cabinet_heat
from orion.hardware_watch.rules import TempPoint

# Readings kept: enough for the 15-min rise window at the guard's refresh cadence.
_KEEP = 240


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class GuardReader:
    def __init__(self, *, cabinet_url: str, grace_sec: float = DEFAULT_READING_GRACE_SEC,
                 clock: Callable[[], datetime] = _utcnow):
        self.cabinet_url = cabinet_url
        self.grace_sec = grace_sec
        self.clock = clock
        self._points: deque[TempPoint] = deque(maxlen=_KEEP)
        self._reading: CabinetHeatReading | None = None   # None = unknown: no reading yet (D10)

    @property
    def thermal_state(self) -> str:
        return self._reading.thermal_state if self._reading is not None else "unknown"

    async def read(self, client: Any) -> dict[str, str | None]:
        return {"thermal": await self._read_thermal(client)}

    async def _read_thermal(self, client: Any) -> str | None:
        now = self.clock()
        failure: str | None = None
        try:
            r = await client.get(self.cabinet_url)
            r.raise_for_status()
            raw = r.json()
            temp = float(raw["snapshot"]["frame"]["environment"]["temp_c"])
            age = float(raw["age_sec"])
            if not math.isfinite(temp) or not math.isfinite(age) or age < 0:
                failure = "unavailable:invalid_reading"
            else:
                ts = now - timedelta(seconds=age)
                if not self._points or ts > self._points[-1].ts:
                    self._points.append(TempPoint(ts, temp))
        except Exception as exc:  # noqa: BLE001
            failure = f"unavailable:{type(exc).__name__}"
        self._reading = read_cabinet_heat(list(self._points), now, grace_sec=self.grace_sec, previous=self._reading)
        state = self._reading.thermal_state
        if state == "unknown":
            age = self._reading.age_sec
            why = failure or (f"degraded:reading_stale_{age:.0f}s" if age is not None else "degraded:no_reading")
            return why[:120]
        if state == "hot":
            return f"hot:room_at_{self._reading.temp_c:.1f}c"[:120]
        return None
