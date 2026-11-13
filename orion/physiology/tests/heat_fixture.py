"""Synthetic hourly data reproducing the 2026-10-14 circe step, with known ground truth.

14 days of hours. At STEP_AT a crash loop starts (load_15m ~1.5 -> ~14, +CRASH_W wall watts)
and vision-council's busy-hours rise (~0.05 -> ~0.25 per hour). harness and cortex-exec keep
their usual random load throughout; a durable-run hold reserves the pool all along but draws
nothing (its compute is the child leases). athena stays flat. Seeded, so every run is identical.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from orion.physiology.heat_archaeology import HourReading, PoolBusy

START = datetime(2026, 10, 7, 3, tzinfo=timezone.utc)
STEP_AT = datetime(2026, 10, 14, 3, tzinfo=timezone.utc)
END = datetime(2026, 10, 21, 3, tzinfo=timezone.utc)

FLOOR_W = 616.0
HARNESS_W = 285.0
VISION_W = 186.0
CORTEX_W = 120.0
CRASH_W = 144.0
NOISE_W = 12.0


def hours() -> list[datetime]:
    n = int((END - START).total_seconds() // 3600)
    return [START + timedelta(hours=i) for i in range(n)]


def build(seed: int = 7, *, step: bool = True) -> tuple[list[HourReading], list[PoolBusy], dict[str, float]]:
    """(readings, busy, truth). truth holds the expected step and per-workload step watts.

    ``step=False`` is the negative control: same workloads and noise, nothing changes at STEP_AT.
    """
    rng = random.Random(seed)
    readings: list[HourReading] = []
    busy: list[PoolBusy] = []
    vc_before, vc_after = [], []
    harness_before, harness_after = [], []
    cortex_before, cortex_after = [], []
    for h in hours():
        after = step and h >= STEP_AT
        harness = rng.uniform(0.0, 1.0)
        cortex = rng.uniform(0.0, 0.5)
        vision = rng.uniform(0.15, 0.35) if after else rng.uniform(0.0, 0.1)
        crash = 1.0 if after else 0.0
        load = (14.0 if after else 1.5) + rng.uniform(-0.8, 0.8)
        wall = FLOOR_W + HARNESS_W * harness + VISION_W * vision + CORTEX_W * cortex + CRASH_W * crash
        wall += rng.gauss(0.0, NOISE_W)
        gpu = 150.0 + 250.0 * harness + 160.0 * vision + 100.0 * cortex + rng.gauss(0.0, 5.0)
        cpu = 60.0 + 70.0 * crash + 10.0 * harness + rng.gauss(0.0, 3.0)
        readings.append(HourReading(hour=h, node="circe", wall_w=wall, gpu_w=gpu, cpu_w=cpu, load_15m=load))
        readings.append(HourReading(
            hour=h, node="athena", wall_w=400.0 + rng.gauss(0.0, 8.0), gpu_w=80.0, cpu_w=90.0,
            load_15m=12.0 + rng.uniform(-1.0, 1.0),
        ))
        busy += [
            PoolBusy(hour=h, group="harness", busy_h=harness),
            PoolBusy(hour=h, group="cortex-exec", busy_h=cortex),
            PoolBusy(hour=h, group="vision-council", busy_h=vision),
            PoolBusy(hour=h, group="run:curiosity.investigate", busy_h=rng.uniform(0.3, 1.0), reservation=True),
        ]
        (vc_after if after else vc_before).append(vision)
        (harness_after if after else harness_before).append(harness)
        (cortex_after if after else cortex_before).append(cortex)

    def d(a, b):
        return sum(b) / len(b) - sum(a) / len(a) if a and b else 0.0

    truth = {
        "crash_w": CRASH_W if step else 0.0,
        "vision_w": VISION_W * d(vc_before, vc_after),
        "harness_w": HARNESS_W * d(harness_before, harness_after),
        "cortex_w": CORTEX_W * d(cortex_before, cortex_after),
    }
    truth["step_w"] = truth["crash_w"] + truth["vision_w"] + truth["harness_w"] + truth["cortex_w"]
    return readings, busy, truth


def db_rows(seed: int = 7) -> dict[str, list[dict]]:
    """The same data as the three SQL queries would return it (asyncpg-record-like dicts)."""
    readings, busy, _ = build(seed)
    energy = [
        {"hour_start": r.hour, "source": r.node, "kwh": r.wall_w / 1000.0, "covered_sec": 3600.0}
        for r in readings
    ]
    bio = [
        {"hour": r.hour, "node": r.node, "gpu_w": r.gpu_w, "cpu_w": r.cpu_w, "load_15m": r.load_15m}
        for r in readings
    ]
    holders = {"harness": ("http:anthropic", None), "cortex-exec": ("cortex-exec", None),
               "vision-council": ("vision-council", None),
               "run:curiosity.investigate": ("durable-runs:abc123", "curiosity.investigate")}
    pool = [
        {"hour": b.hour, "holder": holders[b.group][0], "workflow": holders[b.group][1], "busy_h": b.busy_h}
        for b in busy
    ]
    return {"energy_hourly": energy, "biometrics_hourly": bio, "pool_busy_hours": pool}
