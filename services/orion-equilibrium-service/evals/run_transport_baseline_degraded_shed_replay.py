#!/usr/bin/env python3
"""Eval: replay the 2026-10-08 GPU shed on cortex-orch's background lane, with refusals
counted as degraded (this fix) and as successes (before it).

What happened, from stored rows: on 2026-10-08 the cabinet overheated and the GPU pool
shed. Every ``journal.compose`` call on ``cortex-orch -> orion:cortex:exec:request:
background`` in 15:00-19:00 UTC (263 of 263) and 41 of 46 at 06:00 were
``gpu_pool_unavailable`` refusals (``cognition_traces``). ``rpc_request()`` recorded them
as successes; the hop's stored floor fell 13.9 s -> 6.2 s at 06:00 and -> 1.26 s by
18:00 (``transport_baseline_hourly``), and real calls then read ~6x "saturated" for days.

Fixture ``fixtures/background_lane_gpu_shed_2026-10-08.jsonl`` (read-only export, 60
hours, 2026-10-07 12:00 -> 2026-10-09 23:00 UTC): per hour the hop's stored call count
(``transport_baseline_hourly.success_count``, which then included the refusals) and the
refusal count + median exec-side duration from ``cognition_traces``.

Reconstruction (approximate, stated):
- real calls per hour = stored count - refusals; refusals as stored;
- a refusal's round trip = its exec-side median + 800 ms. Calibrated on the four
  all-refusal hours (15:00-18:00): 440-460 ms + 800 ms ~= 1.26 s, the stored floor/level
  those hours converged to. (Some refusals were 115 s ``deadline`` waits -- slow
  refusals, also not a speed reading.)
- a real call's round trip: 14.0 s before 2026-10-08 19:00 (stored evaluated level on
  2026-10-07: 13.9-14.7 s), 9.4 s after (the window mean #2629's replay of the stored
  rows stated at its regime_shift); log-normal noise sigma 0.15, fixed seed;
- calls spread evenly over the hour's 120 windows of 30 s; warmed first on 15 pools
  of 5 real calls at 14 s.

Checks:
- control (refusals as successes, the pre-fix producer) reproduces the live collapse:
  floor < 2.5 s by 2026-10-08 19:00;
- fixed (refusals as degraded): the floor never falls below 0.8 x the slower real level
  it can legitimately reach (9.4 s), no saturation or regime_shift opens, and every
  refusal is counted in degraded_count;
- after the shed, normal traffic reads saturation_ratio ~1 (median 0.8-1.3).

Run: ``python services/orion-equilibrium-service/evals/run_transport_baseline_degraded_shed_replay.py``
Exit 0 = all checks pass. Prints a JSON report.
"""

from __future__ import annotations

import json
import math
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVICE_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.metacog.transport_baseline import (  # noqa: E402
    TransportBaselineConfig,
    fold_snapshot,
    new_state,
    state_key,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "background_lane_gpu_shed_2026-10-08.jsonl"
SERVICE, INSTANCE, HOP = "cortex-orch", "main", "orion:cortex:exec:request:background"
WIN = 30.0
REAL_BEFORE_MS, REAL_AFTER_MS = 14000.0, 9400.0
SWITCH = datetime(2026, 10, 8, 19, tzinfo=timezone.utc)
REFUSAL_OVERHEAD_MS = 800.0
CFG = TransportBaselineConfig()
KEY = state_key(SERVICE, INSTANCE, HOP)


def _stats(real: list[float], refused: list[float], *, refused_as_success: bool) -> dict:
    ok = real + (refused if refused_as_success else [])
    logs = [math.log(x) for x in ok]
    out = {
        "success_count": len(ok),
        "timeout_count": 0,
        "log_ms_sum": sum(logs),
        "log_ms_sumsq": sum(v * v for v in logs),
        "max_ms": max(ok) if ok else None,
    }
    if not refused_as_success and refused:
        out["degraded_count"] = len(refused)
    return out


def _snap(end: datetime, stats: dict) -> dict:
    return {
        "service": SERVICE,
        "instance": INSTANCE,
        "window_start": (end - timedelta(seconds=WIN)).isoformat(),
        "window_end": end.isoformat(),
        "success_count": 0,
        "timeout_count": 0,
        "channel_latency": {HOP: stats},
    }


def _spread(n: int, slots: int = 120) -> list[int]:
    """Which of the hour's windows each of n calls lands in (evenly spread)."""
    return [int(i * slots / n) for i in range(n)] if n else []


def load() -> list[dict]:
    return [json.loads(line) for line in FIXTURE.read_text().splitlines() if line.strip()]


def replay(*, refused_as_success: bool) -> dict:
    rng = random.Random(20261008)
    state = new_state(CFG)
    rows = load()
    t0 = datetime.fromisoformat(rows[0]["hour_start"])
    # warm-up: 15 pools of 5 real calls at the pre-shed level, one window each
    start = t0 - timedelta(hours=2)
    for k in range(15):
        lat = [REAL_BEFORE_MS * math.exp(rng.gauss(0, 0.15)) for _ in range(5)]
        fold_snapshot(state, _snap(start + timedelta(seconds=WIN * (k + 1)), _stats(lat, [], refused_as_success=True)),
                      config=CFG)
    ks = state.keys[KEY]
    floor_pre = math.exp(ks.floor)
    events, floor_track, ratios_after, degraded_total, refused_total = [], [], [], 0, 0
    floor_at_switch = None
    for row in rows:
        hour = datetime.fromisoformat(row["hour_start"])
        refused_n = int(row["refused_gpu_pool_unavailable"] or 0)
        real_n = max(int(row["hop_success_count"] or 0) - refused_n, 0)
        refused_total += refused_n
        refused_ms = float(row["refused_median_exec_ms"] or 0.0) + REFUSAL_OVERHEAD_MS
        real_slots, ref_slots = _spread(real_n), _spread(refused_n)
        for w in range(120):
            end = hour + timedelta(seconds=WIN * (w + 1))
            real_ms = REAL_BEFORE_MS if end <= SWITCH else REAL_AFTER_MS
            real = [real_ms * math.exp(rng.gauss(0, 0.15)) for s in real_slots if s == w]
            refused = [refused_ms * math.exp(rng.gauss(0, 0.1)) for s in ref_slots if s == w]
            res = fold_snapshot(state, _snap(end, _stats(real, refused, refused_as_success=refused_as_success)),
                                config=CFG)
            events.extend(res.events)
            for ob in res.observations:
                degraded_total += ob.degraded_count
                if ob.evaluated:
                    floor_track.append((end, ob.floor_ms))
                    if end > SWITCH + timedelta(hours=6):
                        ratios_after.append(ob.saturation_ratio)
            if floor_at_switch is None and end >= SWITCH:
                floor_at_switch = math.exp(ks.floor)
    floors = [f for _, f in floor_track]
    ratios_after.sort()
    return {
        "refused_as_success": refused_as_success,
        "floor_pre_ms": round(floor_pre),
        "floor_at_2026-10-08T19:00_ms": round(floor_at_switch) if floor_at_switch else None,
        "floor_min_ms": round(min(floors)) if floors else None,
        "floor_end_ms": round(math.exp(ks.floor)),
        "ratio_median_after_shed": round(ratios_after[len(ratios_after) // 2], 3) if ratios_after else None,
        "events": sorted({f"{e.condition}:{e.phase}" for e in events}),
        "event_count": len(events),
        "degraded_counted": degraded_total,
        "refusals_in_fixture": refused_total,
    }


def main() -> int:
    control = replay(refused_as_success=True)
    fixed = replay(refused_as_success=False)
    checks = {
        "control reproduces the live collapse (floor < 2.5 s by 10-08 19:00)":
            (control["floor_at_2026-10-08T19:00_ms"] or 1e9) < 2500,
        "fixed: floor never below 0.8 x the 9.4 s real level":
            (fixed["floor_min_ms"] or 0) >= 0.8 * REAL_AFTER_MS,
        "fixed: no saturation or regime_shift opened":
            not any(e.startswith(("saturation", "regime_shift")) for e in fixed["events"]),
        "fixed: every refusal counted as degraded":
            fixed["degraded_counted"] == fixed["refusals_in_fixture"] > 0,
        "fixed: normal traffic after the shed reads ratio ~1 (0.8-1.3)":
            fixed["ratio_median_after_shed"] is not None and 0.8 <= fixed["ratio_median_after_shed"] <= 1.3,
    }
    print(json.dumps({"control": control, "fixed": fixed, "checks": checks}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
