"""Measured energy, money and heat for one act: the pure arithmetic.

The act ledger (orion/grammar/act_energy_fill.py) reads the inputs; this module only computes.

ENERGY (``ActCostV2.wh_by_host``) is the act's share of the WHOLE MACHINE's power above rest,
measured at the wall (``chassis_watts``). Two steps:

1. GPU above idle (``wh_gpu_by_host``, the per-card rule below): what the act's own cards drew.
2. Whole machine (``machine_energy``). At each second the host's chassis watts above rest are split
   over everything drawing above its own floor then: every card's above-idle watts plus the CPU
   packages' watts above their fitted idle. The act gets the part that follows its card shares:

       act_w(t) = sum_cards share(t) x chassis_above_rest(t) / load_above_rest(t)

   THEORY: fans, PSU conversion loss, memory and VRMs draw in proportion to the component load
   that makes the heat (PSU loss is a near-constant fraction of DC load; fans follow heat). Live
   circe 2026-10-16..30, 339 hourly means, rest 541 W, CPU idle 100 W: chassis above rest =
   2.69 x (GPU above idle + CPU above idle), R^2 0.79; the residual is uncorrelated with fan_pct
   (r = -0.05) while load and fan_pct correlate r = 0.82. The component VIEWS (a card's, a
   container's or a GPU-only run's watts in the physiology report) use the same relation as a
   per-host fitted slope: orion/energy/overhead.py (fallback: this 2.69).
   CLOSURE, by construction: at each second the shares of every lease, the CPU part and the cards
   nobody leased sum to chassis_above_rest (``attribute_host_window`` reports all the parts).
   REST: pinned per host in config/physiology/rest_body.yaml (orion/physiology/rest_body.py), never
   a rolling percentile: a rolling rest calls sustained work "rest".
   CPU FLOOR: the row's own ``cpu_idle_watts``, the fitted linear-utilisation intercept that the
   per-container CPU split uses (orion/telemetry/biometrics_pipeline.py), so the CPU part here and
   ``container_cpu_watts.*`` are the same watts.

GPU above idle. Watts ABOVE IDLE, never total: an idle card draws its floor
whether or not Orion acts, so charging total watts would land the always-on floor on Orion's
choices. Same rule as the power-intent loop's ``incremental_over_baseline``
(orion/energy/run_cost.py), and the same function (``incremental_joules``) does the subtraction.

* Per card: the card's watts (``gpu{index}_watts`` in orion_biometrics_summary, ~30 s cadence),
  linearly interpolated over each second of the lease, minus the card's idle floor.
* Idle floor: the 5th percentile of the card's own watts over the 24 h before the UTC day
  (``idle_floor``; day-keyed so every act on the card that day reads the same floor). It only
  weights the split between cards and CPU; the total split is the pinned chassis rest.
  The power-intent loop uses one sample just before the window; on a shared card at 30 s cadence
  that sample is often another lease's draw, so the floor is taken from the card's own history.
* Lease share: at each second the card's above-idle watts are split equally among the GPU-pool
  leases holding it then (``max_holds: 2`` cards are shared). The shares of every lease on a card
  sum to the card's above-idle energy, so no watt is charged twice.

MONEY (``usd``): whole-machine Wh / 1000 x the house tariff's marginal $/kWh (energy_cost_accrued,
the energy service's own ledger). HEAT (``heat_c_by_host``): ``heat_c`` of the GPU-above-idle Wh,
below: the heat fit's regressor is compute watts, so its slope already carries the overhead.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Mapping, Optional, Sequence

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


# Cap on chassis_above_rest / load_above_rest. Near-zero load makes the per-sample ratio explode
# (live circe 10-29 23Z..10-30 03Z: median 2.3-2.8, hourly max up to 78), so a card a few watts
# above idle would take the whole surplus. 8 is ~3x the fitted slope (2.69) and the old p99 (8.4).
# Watts over the cap are charged to nobody: they land in unattributed in both paths.
RATIO_CAP = 8.0


@dataclass(frozen=True)
class HostFloors:
    """One host's floors for one UTC day: pinned chassis rest, each card's idle floor (by index)."""

    chassis_w: float
    card_idle_w: Mapping[int, float]


@dataclass(frozen=True)
class HostSeries:
    """One host's readings over a window, (ts, watts) ascending.

    ``cpu_above``: CPU package watts above the row's fitted idle (``cpu_watts_total -
    cpu_idle_watts``, clamped at 0). ``sponsored_cpu``: the sponsored containers' CPU watts
    (``container_cpu_watts.<name>``), part of ``cpu_above``."""

    chassis: Sequence[tuple[datetime, float]]
    cpu_above: Sequence[tuple[datetime, float]]
    cards: Mapping[int, Sequence[tuple[datetime, float]]]
    sponsored_cpu: Sequence[tuple[datetime, float]] = ()


class _At:
    """watts_at with each series' timestamps indexed once."""

    def __init__(self, series: HostSeries, max_gap_s: float):
        self._g = max_gap_s
        self._s = {"chassis": series.chassis, "cpu_above": series.cpu_above,
                   "sponsored_cpu": series.sponsored_cpu, **{i: p for i, p in series.cards.items()}}
        self._t = {k: [p[0] for p in v] for k, v in self._s.items()}

    def __call__(self, key: Any, t: datetime) -> Optional[float]:
        pts = self._s.get(key)
        return watts_at(pts, t, max_gap_s=self._g, times=self._t[key]) if pts else None


def _day(t: datetime) -> date:
    return t.astimezone(timezone.utc).date()


@dataclass(frozen=True)
class _Split:
    above: float                 # chassis watts above rest
    ratio: float                 # above / load (0 when nothing draws above its floor)
    card_above: Mapping[int, float]
    cpu_above: float


def _split(at: _At, floors: HostFloors, t: datetime) -> Optional[_Split]:
    """The host at ``t``. None when any reading the split needs is unknown (chassis, CPU, or any
    card with a floor): a split with a hole in its denominator would overcharge the rest."""
    ch, cpu = at("chassis", t), at("cpu_above", t)
    if ch is None or cpu is None:
        return None
    cards: dict[int, float] = {}
    for idx, idle in floors.card_idle_w.items():
        w = at(idx, t)
        if w is None:
            return None
        cards[idx] = max(0.0, w - idle)
    cpu = max(0.0, cpu)
    load = cpu + sum(cards.values())
    above = max(0.0, ch - floors.chassis_w)
    ratio = min(above / load, RATIO_CAP) if load > 0.0 else 0.0
    return _Split(above=above, ratio=ratio, card_above=cards, cpu_above=cpu)


@dataclass(frozen=True)
class MachineEnergy:
    wh: float       # whole machine, this lease's share
    wh_gpu: float   # the card's above-idle watts, this lease's share (the step-1 number)


def machine_energy(series: HostSeries, floors_by_day: Mapping[date, HostFloors], *, card: int,
                   start: datetime, end: datetime, others: Sequence[tuple[datetime, datetime]] = (),
                   step_s: float = 1.0, max_gap_s: float = MAX_SAMPLE_GAP_S) -> Optional[MachineEnergy]:
    """One lease's share of the host's chassis watts above rest over [start, end].

    At each step: the lease's card share (card above idle / leases holding it) times
    chassis_above_rest / load_above_rest. None when any second lacks a floor or a reading."""
    elapsed = (end - start).total_seconds()
    if elapsed <= 0.0:
        return None
    at = _At(series, max_gap_s)
    steps = max(1, math.ceil(elapsed / step_s))
    dt = elapsed / steps
    wh = wh_gpu = 0.0
    for k in range(steps):
        t = start + timedelta(seconds=(k + 0.5) * dt)
        floors = floors_by_day.get(_day(t))
        if floors is None or card not in floors.card_idle_w:
            return None
        sp = _split(at, floors, t)
        if sp is None:
            return None
        n = 1 + sum(1 for (s, e) in others if s <= t < e)
        share_w = sp.card_above[card] / n
        wh_gpu += share_w * dt
        wh += share_w * sp.ratio * dt
    return MachineEnergy(wh=wh / JOULES_PER_WH, wh_gpu=wh_gpu / JOULES_PER_WH)


@dataclass(frozen=True)
class LeaseSpan:
    start: datetime
    end: datetime
    metered: bool  # the lease's turn has an act (the act fill charges it)


@dataclass(frozen=True)
class HostAttribution:
    """A host's above-rest energy over a window, split by who it can be charged to (Wh).

    ``above_rest_wh = sponsored_wh + metered_wh + unattributed_wh`` (unattributed is the
    remainder, NOT clipped: a negative value means over-attribution and the caller logs it).
    ``leased_unmetered_wh`` is the part of unattributed on leased cards whose turn has no act.
    ``split_unknown_s``: seconds with a chassis reading but a hole in the split (all of their
    above-rest energy lands in unattributed)."""

    rest_wh: float
    above_rest_wh: float
    sponsored_wh: float
    metered_wh: float
    leased_unmetered_wh: float
    covered_s: float
    split_unknown_s: float

    @property
    def unattributed_wh(self) -> float:
        return self.above_rest_wh - self.sponsored_wh - self.metered_wh


def attribute_host_window(series: HostSeries, floors_by_day: Mapping[date, HostFloors],
                          leases_by_card: Mapping[int, Sequence[LeaseSpan]], *, start: datetime, end: datetime,
                          rest_w: Optional[float] = None, step_s: float = 1.0,
                          max_gap_s: float = MAX_SAMPLE_GAP_S) -> HostAttribution:
    """The host's chassis energy above rest over [start, end], split the same way the act fill
    splits it: each leased card's above-idle share, each lease 1/n of its card (metered when the
    lease's turn has an act), the sponsored containers' CPU share, and the remainder.

    ``rest_w``: the pinned rest, so a day whose card floors are missing still measures the chassis
    (its seconds count as ``split_unknown_s``) instead of vanishing."""
    elapsed = (end - start).total_seconds()
    if elapsed <= 0.0:
        return HostAttribution(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    at = _At(series, max_gap_s)
    steps = max(1, math.ceil(elapsed / step_s))
    dt = elapsed / steps
    rest = above = sponsored = metered = leased_un = covered = unknown = 0.0
    # Sweep each card's leases in time order: a busy lane closes ~800 leases an hour, so testing
    # every lease at every second would be quadratic. Steps are increasing, so a pointer suffices.
    order = {c: sorted(spans, key=lambda x: x.start) for c, spans in leases_by_card.items()}
    nxt = {c: 0 for c in order}
    active: dict[int, list[LeaseSpan]] = {c: [] for c in order}
    for k in range(steps):
        t = start + timedelta(seconds=(k + 0.5) * dt)
        for c, spans in order.items():
            while nxt[c] < len(spans) and spans[nxt[c]].start <= t:
                active[c].append(spans[nxt[c]])
                nxt[c] += 1
            active[c] = [x for x in active[c] if x.end > t]
        floors = floors_by_day.get(_day(t))
        ch = at("chassis", t)
        r_w = floors.chassis_w if floors is not None else rest_w
        if r_w is None or ch is None:
            continue
        covered += dt
        rest += r_w * dt
        sp = _split(at, floors, t) if floors is not None else None
        if sp is None:
            above += max(0.0, ch - r_w) * dt
            unknown += dt
            continue
        above += sp.above * dt
        sp_cpu = at("sponsored_cpu", t) or 0.0
        sponsored += min(max(0.0, sp_cpu), sp.cpu_above) * sp.ratio * dt
        for card, live in active.items():
            if card not in sp.card_above:
                continue
            held = [x.metered for x in live]
            if not held:
                continue
            w = sp.card_above[card] * sp.ratio * dt
            m = sum(held) / len(held)
            metered += w * m
            leased_un += w * (1.0 - m)
    j = JOULES_PER_WH
    return HostAttribution(rest_wh=rest / j, above_rest_wh=above / j, sponsored_wh=sponsored / j,
                           metered_wh=metered / j, leased_unmetered_wh=leased_un / j, covered_s=covered,
                           split_unknown_s=unknown)


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
    "HostAttribution",
    "HostFloors",
    "HostSeries",
    "LeaseSpan",
    "MachineEnergy",
    "RATIO_CAP",
    "attribute_host_window",
    "machine_energy",
    "JOULES_PER_KWH",
    "JOULES_PER_WH",
    "card_energy",
    "heat_c",
    "idle_floor",
    "incremental_joules",
    "usd",
    "watts_at",
]
