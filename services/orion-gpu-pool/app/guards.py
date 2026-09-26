"""Swap-load guards (stage 4 spec, "Guards (pool side)"): the two physical checks durable-runs'
elastic runtime makes today before borrowing gpu2 (services/orion-durable-runs/app/elastic_runtime.py
``environment()``), read here so the pool is the one decider.

- ``thermal``: the cabinet sensor through ``orion.autonomy.thermal_gate.thermal_state`` (same
  hysteresis). Blocks when the verdict does not allow GPU work OR is degraded (no/stale reading):
  loading an extra model into a room nobody can measure is refused, like the elastic path does.
- ``visual_baseline``: thought ``/visual-chain/activity``. Blocks while the visual chain's baseline
  is overdue or an attempt is running (it needs diffusion on gpu2). Stage-4-only; deleted when the
  visual chain moves onto diffusion leases (stage 5).

Each guard is None when clear, else a short reason. A read that fails is a reason ("unavailable:
..."), never a silent pass. Runs OUTSIDE the runtime lock: HTTP must not stall lease RPCs.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any

from orion.autonomy.thermal_gate import thermal_state


class GuardReader:
    def __init__(self, *, cabinet_url: str, visual_activity_url: str):
        self.cabinet_url = cabinet_url
        self.visual_activity_url = visual_activity_url
        self._thermal = "hot"   # conservative until the first real reading (as elastic_runtime)

    async def read(self, client: Any, now: datetime) -> dict[str, str | None]:
        return {"thermal": await self._read_thermal(client),
                "visual_baseline": await self._read_visual(client, now)}

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

    async def _read_visual(self, client: Any, now: datetime) -> str | None:
        from orion.reverie.baseline import load_baseline_policy
        from orion.schemas.reverie_visual import VisualActivityV1

        policy = load_baseline_policy()
        if not policy.enabled:
            return None
        try:
            r = await client.get(self.visual_activity_url)
            r.raise_for_status()
            activity = VisualActivityV1.model_validate(r.json())
        except Exception as exc:  # noqa: BLE001
            return f"unavailable:{type(exc).__name__}"[:120]
        age = (now - activity.observed_at).total_seconds()
        if activity.history_status != "ok" or not 0 <= age <= policy.freshness_sec:
            return "visual_activity_unavailable"
        if activity.active_attempt_id:
            return "visual_attempt_running"
        if not activity.last_success_at or \
                activity.last_success_at + timedelta(seconds=policy.interval_sec) <= now:
            return "visual_baseline_urgent"
        return None
