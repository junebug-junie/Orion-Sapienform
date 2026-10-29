"""Measured energy, money and heat for one act: the pure arithmetic.

The act ledger (orion/grammar/act_energy_fill.py) reads the inputs; this module only computes.

ENERGY (``ActCostV2.wh_by_host``). Watts ABOVE IDLE, never total: an idle card draws its floor
whether or not Orion acts, so charging total watts would land the always-on floor on Orion's
choices. Same rule as the power-intent loop's ``incremental_over_baseline``
(orion/energy/run_cost.py), and the same function (``incremental_joules``) does the subtraction.

* Per card: the card's watts (``gpu{index}_watts`` in orion_biometrics_summary, ~30 s cadence),
  linearly interpolated over each second of the lease, minus the card's idle floor.
* Idle floor: the 5th percentile of the card's own watts over the trailing 24 h (``idle_floor``).
  The power-intent loop uses one sample just before the window; on a shared card at 30 s cadence
  that sample is often another lease's draw, so the floor is taken from the card's own history.
* Lease share: at each second the card's above-idle watts are split equally among the GPU-pool
  leases holding it then (``max_holds: 2`` cards are shared). The shares of every lease on a card
  sum to the card's above-idle energy, so no watt is charged twice.

MONEY (``usd``): Wh / 1000 x the house tariff's marginal $/kWh (energy_cost_accrued, the energy
service's own ledger). HEAT (``heat_c_by_host``): ``heat_c``, below.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Sequence

JOULES_PER_WH = 3600.0
JOULES_PER_KWH = 3_600_000.0
# A sample further than this from the instant it stands for is not that instant's reading.
# 3 biometrics ticks (TELEMETRY_INTERVAL=30).
MAX_SAMPLE_GAP_S = 90.0
# p05 of the card's own 24 h. Measured bias (eval act_energy_vs_power_intent.py, 90 reverie images,
# 2026-10-29): this floor sits ~5.5 W under the power-intent loop's pre-window sample, so a reverie
# family's Wh reads ~1.2x the 1 Hz meter. Kept deliberately: "above idle" means above the card at
# rest, and a card warmed for an image is not at rest. With the meter's floor the ratio is ~0.93.
IDLE_QUANTILE = 0.05
IDLE_MIN_SAMPLES = 60  # 30 min of 30 s samples


def incremental_joules(mean_w: float, baseline_w: float, elapsed_s: float) -> Optional[float]:
    """Joules above a baseline: max(0, mean - baseline) x elapsed. None when not a measurement.

    Shared with the power-intent run cost (``incremental_over_baseline``)."""
    if not (math.isfinite(mean_w) and math.isfinite(baseline_w) and math.isfinite(elapsed_s)):
        return None
    if mean_w < 0.0 or elapsed_s <= 0.0:
        return None
    return max(0.0, mean_w - baseline_w) * elapsed_s


def idle_floor(values: Sequence[float], *, q: float = IDLE_QUANTILE, min_n: int = IDLE_MIN_SAMPLES) -> Optional[float]:
    """The card's idle floor: a low quantile of its own watts. None with too little history."""
    clean = sorted(v for v in values if isinstance(v, (int, float)) and math.isfinite(v) and v >= 0.0)
    if len(clean) < min_n:
        return None
    pos = q * (len(clean) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(clean) - 1)
    return clean[lo] + (clean[hi] - clean[lo]) * (pos - lo)


def watts_at(points: Sequence[tuple[datetime, float]], t: datetime, *,
             max_gap_s: float = MAX_SAMPLE_GAP_S, times: Optional[Sequence[datetime]] = None) -> Optional[float]:
    """The card's watts at ``t``: linear between the bracketing samples. Outside the samples the
    nearest one holds for ``max_gap_s``; a longer gap is unknown (None), never 0."""
    if not points:
        return None
    times = times if times is not None else [p[0] for p in points]
    i = bisect.bisect_left(times, t)
    if i < len(points) and points[i][0] == t:
        return points[i][1]
    if i == 0:
        return points[0][1] if (points[0][0] - t).total_seconds() <= max_gap_s else None
    if i == len(points):
        return points[-1][1] if (t - points[-1][0]).total_seconds() <= max_gap_s else None
    (ta, wa), (tb, wb) = points[i - 1], points[i]
    span = (tb - ta).total_seconds()
    if span > 2.0 * max_gap_s:
        return None
    return wa + (wb - wa) * (t - ta).total_seconds() / span


@dataclass(frozen=True)
class CardEnergy:
    wh: float
    samples_in_window: int
    mean_share: float  # 1.0 = the act's lease held the card alone the whole time


def card_energy(points: Sequence[tuple[datetime, float]], *, idle_w: float, start: datetime, end: datetime,
                others: Sequence[tuple[datetime, datetime]] = (), step_s: float = 1.0,
                max_gap_s: float = MAX_SAMPLE_GAP_S) -> Optional[CardEnergy]:
    """Above-idle energy one lease drew on one card over [start, end], split by lease share.

    ``points``: (ts, watts) ascending. ``others``: (start, end) of every OTHER lease on the card.
    None when any second of the window has no reading within ``max_gap_s`` (absent, never 0)."""
    elapsed = (end - start).total_seconds()
    if elapsed <= 0.0 or not math.isfinite(idle_w):
        return None
    steps = max(1, math.ceil(elapsed / step_s))
    dt = elapsed / steps
    joules = 0.0
    share_sum = 0.0
    times = [p[0] for p in points]
    for k in range(steps):
        t = start + timedelta(seconds=(k + 0.5) * dt)
        w = watts_at(points, t, max_gap_s=max_gap_s, times=times)
        if w is None:
            return None
        n = 1 + sum(1 for (s, e) in others if s <= t < e)
        part = incremental_joules(w, idle_w, dt)
        if part is None:
            return None
        joules += part / n
        share_sum += 1.0 / n
    inside = sum(1 for (ts, _) in points if start <= ts <= end)
    return CardEnergy(wh=joules / JOULES_PER_WH, samples_in_window=inside, mean_share=share_sum / steps)


def heat_c(wh: float, sensitivity_c_per_100w: float) -> float:
    """Cabinet degrees C one act adds, from the fitted heat sensitivity.

    config/physiology/heat_sensitivity.yaml fits degC per 100 W step of HOURLY MEAN compute watts
    (2 h response). An act that draws E Wh raises its hour's mean by E W, so its share is
    sensitivity x E / 100. Linear superposition of the fitted step response: an estimate of the
    act's contribution, not a thermometer reading."""
    return float(sensitivity_c_per_100w) * max(0.0, float(wh)) / 100.0


def usd(wh: float, marginal_usd_per_kwh: float) -> float:
    return max(0.0, float(wh)) / 1000.0 * float(marginal_usd_per_kwh)


__all__ = [
    "CardEnergy",
    "JOULES_PER_KWH",
    "JOULES_PER_WH",
    "card_energy",
    "heat_c",
    "idle_floor",
    "incremental_joules",
    "usd",
    "watts_at",
]
