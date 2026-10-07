"""D11: does shedding GPU work lower the cabinet temperature at all? (C14)

Spec: docs/superpowers/specs/2026-10-06-thermal-controller-redesign-design.md, D11 and Acceptance
check 4. Pure functions, no I/O: the hardware-watch eval feeds them recorded shed episodes plus the
biometrics readings around them.

Per episode: the cabinet's minute mean at shed start, +15 min and +30 min (``cabinet_temp_c``), and the
GPU draw before vs during (``gpu_watts_total``). A raw before/after delta is not an effect: the cabinet
drifts on its own (AC cycling, the day's heat, regression to the mean after a trigger fired on a high
reading). So every episode is compared with a CONTROL: the same delta measured at every 5-minute grid
time outside any shed whose start temperature is within ``band_c`` of the episode's AND whose prior
15-minute rise (``cabinet_rise_c``, the trigger's own quantity) is within ``rise_band_c`` of the
episode's -- a shed that fired on a rise is compared with other rises, not with flat stretches
(found on the 10-06 02:12 episode: -0.8 C vs an unmatched control, with GPU draw down only 6 W). The
reported effect is ``delta - control median``. If that stays ~0 over >= 3 episodes, the shed tier protects
nothing and the follow-up is to drop it, not tune it (spec D11).

Metric gate (CLAUDE.md), recorded in the PR: provenance = orion/telemetry/cabinet_sensors.py
``cabinet_temp_c`` and orion/telemetry/biometrics_pipeline.py ``gpu_watts_total`` via
orion_biometrics_summary; independence = reads only the OUTCOME window [start, start+30 min], never the
trigger window, and is differenced against a matched control; anchor = GPU electrical power is
dissipated as heat inside the cabinet, so withheld GPU work must lower heat input.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Sequence

from orion.autonomy.cabinet_heat import minute_mean
from orion.hardware_watch.rules import TempPoint, cabinet_rise_c

HORIZONS_MIN = (15, 30)
CONTROL_STEP_SEC = 300.0


@dataclass(frozen=True)
class ShedEpisode:
    source: str            # "hardware_watch_incident" (v1) | "gpu_pool_orion_shed" | "reflex_replay" ...
    episode_id: str
    reason: str
    start: datetime
    end: datetime | None


def _mean_between(points: Sequence[TempPoint], lo: datetime, hi: datetime) -> float | None:
    vals = [p.value for p in points if lo <= p.ts <= hi and math.isfinite(p.value)]
    return sum(vals) / len(vals) if vals else None


def _delta(cabinet: Sequence[TempPoint], t: datetime, minutes: int) -> tuple[float | None, float | None]:
    t0 = minute_mean(cabinet, t)
    th = minute_mean(cabinet, t + timedelta(minutes=minutes))
    return t0, (None if t0 is None or th is None else round(th - t0, 3))


def _rise_at(cabinet: Sequence[TempPoint], t: datetime) -> float | None:
    return cabinet_rise_c([p for p in cabinet if t - timedelta(minutes=16) <= p.ts <= t], t)


def control_deltas(cabinet: Sequence[TempPoint], episodes: Sequence[ShedEpisode], *, near_c: float,
                   band_c: float, minutes: int, step_sec: float = CONTROL_STEP_SEC,
                   near_rise_c: float | None = None, rise_band_c: float = 0.3) -> list[float]:
    """Delta(+minutes) at every grid time with no shed in [t - 30 min, t + minutes], a start
    temperature within ``band_c`` of ``near_c`` and (when given) a prior 15-min rise within
    ``rise_band_c`` of ``near_rise_c``."""
    if not cabinet:
        return []
    pad = timedelta(minutes=30)
    busy = [(e.start - pad, (e.end or e.start) + timedelta(minutes=minutes)) for e in episodes]
    out: list[float] = []
    t, last = cabinet[0].ts + pad, cabinet[-1].ts - timedelta(minutes=minutes)
    while t <= last:
        if not any(a <= t <= b for a, b in busy):
            t0, d = _delta(cabinet, t, minutes)
            if d is not None and abs(t0 - near_c) <= band_c:
                r = _rise_at(cabinet, t) if near_rise_c is not None else None
                if near_rise_c is None or (r is not None and abs(r - near_rise_c) <= rise_band_c):
                    out.append(d)
        t += timedelta(seconds=step_sec)
    return out


def shed_effect(ep: ShedEpisode, cabinet: Sequence[TempPoint], gpu_watts: Sequence[TempPoint],
                all_episodes: Sequence[ShedEpisode] = (), *, band_c: float = 0.5) -> dict:
    """One episode's row for the D11 report."""
    row: dict = {"source": ep.source, "episode_id": ep.episode_id, "reason": ep.reason,
                 "start": ep.start.isoformat(), "end": ep.end.isoformat() if ep.end else None,
                 "minutes": None if ep.end is None else round((ep.end - ep.start).total_seconds() / 60, 1)}
    t0 = minute_mean(cabinet, ep.start)
    rise = _rise_at(cabinet, ep.start)
    row["cabinet_c_start"] = None if t0 is None else round(t0, 2)
    row["rise_c_before_start"] = rise
    for h in HORIZONS_MIN:
        _, d = _delta(cabinet, ep.start, h)
        ctrl = control_deltas(cabinet, list(all_episodes) or [ep], near_c=t0, band_c=band_c, minutes=h,
                              near_rise_c=rise) if t0 is not None else []
        med = statistics.median(ctrl) if ctrl else None
        row[f"cabinet_c_plus{h}"] = None if d is None or t0 is None else round(t0 + d, 2)
        row[f"delta_c_{h}m"] = d
        row[f"control_median_delta_c_{h}m"] = None if med is None else round(med, 3)
        row[f"control_n_{h}m"] = len(ctrl)
        row[f"effect_c_{h}m"] = None if d is None or med is None else round(d - med, 3)
    before = _mean_between(gpu_watts, ep.start - timedelta(minutes=10), ep.start)
    during = _mean_between(gpu_watts, ep.start, ep.start + timedelta(minutes=15))
    row["gpu_w_before"] = None if before is None else round(before, 1)
    row["gpu_w_first15m"] = None if during is None else round(during, 1)
    row["gpu_w_drop"] = None if before is None or during is None else round(before - during, 1)
    return row


def summarize(rows: Sequence[dict]) -> dict:
    """Acceptance check 4: needs >= 3 episodes with a measured 15-min effect before anyone concludes."""
    eff = [r["effect_c_15m"] for r in rows if r.get("effect_c_15m") is not None]
    return {"episodes": len(rows), "measured_15m": len(eff),
            "mean_effect_c_15m": round(sum(eff) / len(eff), 3) if eff else None,
            "enough_to_judge": len(eff) >= 3}


__all__ = ["HORIZONS_MIN", "ShedEpisode", "control_deltas", "shed_effect", "summarize"]
