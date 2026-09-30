"""Swap-load guards (stage 4 spec, "Guards (pool side)"): the physical check durable-runs'
elastic runtime made before borrowing gpu2 (``environment()`` in
services/orion-durable-runs/app/elastic_runtime.py, deleted in 4.5), read here so the pool is the
one decider.

- ``thermal``: the cabinet sensor through ``orion.autonomy.thermal_gate.thermal_state`` (same
  hysteresis). Blocks when the verdict does not allow GPU work OR is degraded (no/stale reading):
  loading an extra model into a room nobody can measure is refused, like the elastic path does.

``visual_baseline`` (thought ``/visual-chain/activity``: refuse a 27B load while the image baseline
was overdue) was deleted in stage 5.4. A reverie-visual run takes a diffusion hold on its own cadence
and reclaims gpu2 through the queue (owner reclaim), so the pool needs no side read of the chain.

Each guard is None when clear, else a short reason. A read that fails is a reason ("unavailable:
..."), never a silent pass. Runs OUTSIDE the runtime lock: HTTP must not stall lease RPCs.
"""
from __future__ import annotations

import math
from typing import Any

from orion.autonomy.thermal_gate import thermal_state


class GuardReader:
    def __init__(self, *, cabinet_url: str):
        self.cabinet_url = cabinet_url
        self._thermal = "hot"   # conservative until the first real reading (as elastic_runtime)

    async def read(self, client: Any) -> dict[str, str | None]:
        return {"thermal": await self._read_thermal(client)}

    async def _read_thermal(self, client: Any) -> str | None:
        try:
            r = await client.get(self.cabinet_url)
            r.raise_for_status()
            raw = r.json()
            temp = float(raw["snapshot"]["frame"]["environment"]["temp_c"])
            age = float(raw["age_sec"])
            if not math.isfinite(temp) or not math.isfinite(age) or age < 0:
                return "unavailable:invalid_reading"
        except Exception as exc:  # noqa: BLE001
            return f"unavailable:{type(exc).__name__}"[:120]
        verdict = thermal_state(temp_c=temp, age_sec=age, previous_state=self._thermal)
        if not verdict.degraded:
            self._thermal = verdict.state
        if verdict.degraded:
            return f"degraded:{verdict.reason}"[:120]
        if not verdict.allows_gpu_work:
            return f"{verdict.state}:{verdict.reason}"[:120]
        return None
