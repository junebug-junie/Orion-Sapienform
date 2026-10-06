"""orion-hardware-watch's rules: pure functions from readings to verdicts.

No I/O, no clock, no randomness: the service feeds them what it read from Postgres, and the replay
eval (services/orion-hardware-watch/evals/run_rules_replay_eval.py) feeds them real history tick by
tick. Same code both ways.

Spec: docs/superpowers/specs/2026-09-28-urgent-curiosity-and-hardware-watch-design.md (Part 4).
Plan: docs/superpowers/plans/2026-09-29-urgent-curiosity-plan-4-5-hardware-watch-and-shedding.md.
Thermal controller v2: docs/superpowers/specs/2026-10-06-thermal-controller-redesign-design.md --
``cooling_verdict_v2`` (D5: AC power is a diagnosis, not a trigger) and the ceiling-only heat rules
(D7). ``cooling_verdict`` / ``shed_verdict`` are the v1 rules, kept only while
``HARDWARE_WATCH_HEAT_CONTROLLER=v1`` is the one-week rollback path.

"Held for" everywhere means: the contiguous run of readings, newest first, that satisfy the
condition; the run's age is ``now - oldest reading in the run``. The newest reading must be recent
(``max_gap_sec``), otherwise nothing is "held" -- silence is the freshness arm's job, not a
continuation of the last value.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Sequence, TypeVar

from orion.autonomy.thermal_gate import DEFAULT_ELEVATED_C, DEFAULT_MAX_READING_AGE_SEC

T = TypeVar("T")


# --- readings ------------------------------------------------------------------------------

@dataclass(frozen=True)
class CoolingPoint:
    """One home_cooling_sample row."""

    ts: datetime
    watts: float | None
    stale: bool | None          # None = producer predates the flag (before #2382)
    device_online: bool = True
    controller_ready: bool = True

    @property
    def live(self) -> bool:
        """A reading the room's cooling can be judged from."""
        return (self.stale is not True and self.device_online and self.controller_ready
                and self.watts is not None and math.isfinite(self.watts))


@dataclass(frozen=True)
class TempPoint:
    ts: datetime
    value: float


@dataclass(frozen=True)
class Verdict:
    open_reason: str | None      # an opening arm holds now (named)
    resolve: bool                # the resolve condition holds now
    detail: dict = field(default_factory=dict)


def _age(now: datetime, ts: datetime) -> float:
    return (now - ts).total_seconds()


def held_for(points: Sequence[T], pred: Callable[[T], bool], now: datetime, *,
             ts: Callable[[T], datetime], max_gap_sec: float) -> float:
    """Seconds the condition has held continuously up to ``now``; 0 when the newest point fails it
    or is older than ``max_gap_sec``. ``points`` ascending by time."""
    if not points or _age(now, ts(points[-1])) > max_gap_sec or not pred(points[-1]):
        return 0.0
    oldest = points[-1]
    for p in reversed(points):
        if not pred(p):
            break
        oldest = p
    return max(0.0, _age(now, ts(oldest)))


# --- cooling (subject cabinet_ac) ------------------------------------------------------------

@dataclass(frozen=True)
class CoolingRuleConfig:
    low_watts: float = 150.0
    low_sec: float = 180.0
    stale_sec: float = 300.0
    frozen_sec: float = 3600.0
    resolve_watts: float = 500.0
    resolve_sec: float = 600.0
    # The newest live reading must be at most this old for a held-for arm to count.
    max_gap_sec: float = 120.0


def _cooling_ok_sec(live: Sequence[CoolingPoint], now: datetime, cfg: CoolingRuleConfig) -> float:
    """Seconds the AC has been "working" up to ``now`` (0 when the newest live reading is stale).

    A reading works if it is >= ``resolve_watts``, or it is a dip lasting less than ``low_sec`` after
    such a reading: the same dip length that cannot open ``low_power``. A thermostat-cycling AC
    (compressor ~750 W, idle ~105 W) never holds 500 W for 10 minutes, but it is cooling."""
    last_high: datetime | None = None
    working: list[bool] = []
    for p in live:
        if p.watts >= cfg.resolve_watts:
            last_high = p.ts
        working.append(last_high is not None and _age(p.ts, last_high) <= cfg.low_sec)
    return held_for(list(zip(live, working)), lambda x: x[1], now, ts=lambda x: x[0].ts, max_gap_sec=cfg.max_gap_sec)


def _freshness_arm(points: Sequence[CoolingPoint], live: Sequence[CoolingPoint], now: datetime,
                   cfg: CoolingRuleConfig, detail: dict) -> str | None:
    """No live reading for stale_sec -> named from the newest row (or no_samples: no row at all)."""
    newest = points[-1] if points else None
    last_live = live[-1] if live else None
    silent_sec = _age(now, last_live.ts) if last_live else math.inf
    detail["silent_sec"] = None if math.isinf(silent_sec) else round(silent_sec, 1)
    if silent_sec < cfg.stale_sec:
        return None
    if newest is None or _age(now, newest.ts) >= cfg.stale_sec:
        return "no_samples"
    if not newest.device_online:
        return "device_offline"
    if not newest.controller_ready:
        return "controller_not_ready"
    return "no_fresh_sample"


def cooling_verdict(points: Sequence[CoolingPoint], now: datetime,
                    cfg: CoolingRuleConfig = CoolingRuleConfig()) -> Verdict:
    """``points``: every home_cooling_sample row in the lookback (>= frozen_sec + margin), ascending.

    Opening arms, first match named:
      no live reading for stale_sec -> device_offline | controller_not_ready | no_fresh_sample
                                        (named from the newest row) | no_samples (no row at all)
      live cooling_watts < low_watts held low_sec      -> low_power
      identical live cooling_watts held frozen_sec      -> frozen
    Resolve: the AC 'working' (>= resolve_watts, or a dip shorter than low_sec after one -- a cycling
    compressor) for resolve_sec AND no opening arm holds (a
    frozen 850 W would otherwise open and resolve on alternate ticks).
    """
    live = [p for p in points if p.live]
    newest = points[-1] if points else None
    last_live = live[-1] if live else None
    detail: dict = {
        "rows": len(points),
        "live_rows": len(live),
        "newest_ts": newest.ts.isoformat() if newest else None,
        "last_live_ts": last_live.ts.isoformat() if last_live else None,
        "last_live_watts": last_live.watts if last_live else None,
    }

    open_reason = _freshness_arm(points, live, now, cfg, detail)

    ts = lambda p: p.ts  # noqa: E731
    low = held_for(live, lambda p: p.watts < cfg.low_watts, now, ts=ts, max_gap_sec=cfg.max_gap_sec)
    detail["low_power_sec"] = round(low, 1)
    if open_reason is None and low >= cfg.low_sec:
        open_reason = "low_power"

    frozen = 0.0
    if last_live is not None:
        frozen = held_for(live, lambda p: p.watts == last_live.watts, now, ts=ts, max_gap_sec=cfg.max_gap_sec)
    detail["identical_sec"] = round(frozen, 1)
    if open_reason is None and frozen >= cfg.frozen_sec:
        open_reason = "frozen"

    good = _cooling_ok_sec(live, now, cfg)
    detail["cooling_ok_sec"] = round(good, 1)
    return Verdict(open_reason, open_reason is None and good >= cfg.resolve_sec, detail)


# --- cooling v2 (D5): AC power is a diagnosis, the cabinet is the trigger -------------------

@dataclass(frozen=True)
class AcLowConfig:
    """"AC looks low" = mean live watts over ``window_sec`` below ``low_mean_w``. A healthy cycling
    AC averaged 187-375 W per 10 min on the 10-06 cool night (compressor ~60 s on, ~3:10 fan-only at
    ~104 W); a dead one is ~100 W flat. Resolve needs ``resolve_factor`` x the floor (hysteresis, C4)."""

    low_mean_w: float = 140.0
    window_sec: float = 900.0
    resolve_factor: float = 1.5
    lookahead_min: float = 20.0
    # The window only counts when live readings cover it: none older than this apart, at either end.
    max_gap_sec: float = 120.0


def ac_mean_w(live: Sequence[CoolingPoint], now: datetime, cfg: AcLowConfig) -> float | None:
    """Mean watts of the live readings in ``[now - window_sec, now]``; None unless they cover the
    window (first within max_gap of its start, no gap > max_gap, newest within max_gap of now). A
    half-filled window right after an outage is not evidence of a low AC (10-03 21:16)."""
    start = now - timedelta(seconds=cfg.window_sec)
    win = [p for p in live if start <= p.ts <= now]
    if not win or _age(now, win[-1].ts) > cfg.max_gap_sec or (win[0].ts - start).total_seconds() > cfg.max_gap_sec:
        return None
    if any((b.ts - a.ts).total_seconds() > cfg.max_gap_sec for a, b in zip(win, win[1:])):
        return None
    return sum(p.watts for p in win) / len(win)


FRESHNESS_REASONS = frozenset({"no_samples", "device_offline", "controller_not_ready", "no_fresh_sample"})


def cooling_verdict_v2(points: Sequence[CoolingPoint], now: datetime, cabinet, *,
                       cfg: CoolingRuleConfig = CoolingRuleConfig(), ac: AcLowConfig = AcLowConfig(),
                       open_reason: str | None = None) -> Verdict:
    """v2 cooling rule. ``cabinet``: an ``orion.autonomy.cabinet_heat.CabinetHeatReading`` (no AC
    input). ``open_reason``: the open incident's reason, if one is open (picks the resolve rule).

    Opening arms, first match named:
      no live reading for stale_sec -> device_offline | controller_not_ready | no_fresh_sample |
                                       no_samples   (a dead plug is real: these ALERT; they never shed)
      AC low AND the cabinet is elevated/hot (unknown counts as elevated) or projected to reach hot
      within lookahead_min                          -> low_power (no bare "rising" arm: C2)
      identical live watts held frozen_sec          -> frozen
    Resolve (no opening arm holds AND a covered window): low_power needs the mean >= resolve_factor x
    the floor; any other reason needs the mean back at or above the floor.
    ``detail["ac_low"]``: True/False from a covered window, None when the window is not covered."""
    live = [p for p in points if p.live]
    last_live = live[-1] if live else None
    detail: dict = {"rows": len(points), "live_rows": len(live),
                    "last_live_ts": last_live.ts.isoformat() if last_live else None,
                    "last_live_watts": last_live.watts if last_live else None}
    reason = _freshness_arm(points, live, now, cfg, detail)
    mean = ac_mean_w(live, now, ac)
    ac_low = None if mean is None else mean < ac.low_mean_w
    mth = getattr(cabinet, "minutes_to_hot", None)
    warm = cabinet.effective_state in ("elevated", "hot") or (mth is not None and mth <= ac.lookahead_min)
    detail.update({"ac_mean_w": None if mean is None else round(mean, 1), "ac_low": ac_low,
                   "ac_low_mean_w": ac.low_mean_w, "ac_window_sec": ac.window_sec,
                   "cabinet": cabinet.as_dict(), "cabinet_warm": warm})
    if reason is None and ac_low and warm:
        reason = "low_power"
    if reason is None and last_live is not None:
        frozen = held_for(live, lambda p: p.watts == last_live.watts, now, ts=lambda p: p.ts,
                          max_gap_sec=cfg.max_gap_sec)
        detail["identical_sec"] = round(frozen, 1)
        if frozen >= cfg.frozen_sec:
            reason = "frozen"
    floor = ac.low_mean_w * (ac.resolve_factor if open_reason in (None, "low_power") else 1.0)
    resolve = reason is None and mean is not None and mean >= floor
    detail["resolve_floor_w"] = floor
    return Verdict(reason, resolve, detail)


# --- heat (subjects athena, circe, circe/gpuN) -----------------------------------------------

def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear interpolation between closest ranks (Postgres percentile_cont). None when empty."""
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return float(xs[lo])
    return float(xs[lo] + (xs[hi] - xs[lo]) * (pos - lo))


@dataclass(frozen=True)
class Baseline:
    p75: float | None
    p95: float | None
    n: int
    history_sec: float          # now - oldest reading the baseline was built from

    @classmethod
    def from_points(cls, points: Sequence[TempPoint], now: datetime) -> "Baseline":
        values = [p.value for p in points]
        oldest = min((p.ts for p in points), default=None)
        return cls(percentile(values, 0.75), percentile(values, 0.95), len(values),
                   _age(now, oldest) if oldest else 0.0)


@dataclass(frozen=True)
class HeatRuleConfig:
    sustain_sec: float = 600.0
    # The p95 arm arms only once the baseline covers this much history (CPU 1 day, GPU 3 days).
    min_history_sec: float = 86400.0
    ceiling_c: float | None = None          # GPU: always-armed absolute ceiling
    ceiling_sustain_sec: float = 120.0
    ceiling_rearm_c: float | None = None    # resolve below this while the p95 arm is not armed
    max_gap_sec: float = 300.0
    # v2 D7: the p95 is context, not a trigger (it is true ~5 % of the time by construction, C9).
    p95_opens: bool = True
    # v2 D7: an open incident whose sensor is silent this long resolves as sensor_lost (C12). None = never.
    lost_sec: float | None = None


def heat_verdict(points: Sequence[TempPoint], now: datetime, base: Baseline,
                 cfg: HeatRuleConfig = HeatRuleConfig()) -> Verdict:
    """Open: above the ceiling held ceiling_sustain_sec (above_ceiling), else above the own p95
    held sustain_sec with the baseline armed (above_p95). Resolve: newest reading below p75 when
    armed, else below ceiling_rearm_c. A silent sensor neither opens nor resolves."""
    armed = base.p95 is not None and base.history_sec >= cfg.min_history_sec
    newest = points[-1] if points else None
    fresh = newest is not None and _age(now, newest.ts) <= cfg.max_gap_sec
    ts = lambda p: p.ts  # noqa: E731
    detail: dict = {"armed": armed, "p75": base.p75, "p95": base.p95, "baseline_n": base.n,
                    "baseline_history_h": round(base.history_sec / 3600, 1),
                    "newest": newest.value if newest else None,
                    "newest_ts": newest.ts.isoformat() if newest else None}
    open_reason = None
    if cfg.ceiling_c is not None:
        over = held_for(points, lambda p: p.value >= cfg.ceiling_c, now, ts=ts, max_gap_sec=cfg.max_gap_sec)
        detail["above_ceiling_sec"] = round(over, 1)
        if over >= cfg.ceiling_sustain_sec:
            open_reason = "above_ceiling"
    if newest is not None and base.p95 is not None:
        detail["above_p95_c"] = round(newest.value - base.p95, 2)   # annotation ("22 C above its usual")
    if open_reason is None and armed and cfg.p95_opens:
        over = held_for(points, lambda p: p.value > base.p95, now, ts=ts, max_gap_sec=cfg.max_gap_sec)
        detail["above_p95_sec"] = round(over, 1)
        if over >= cfg.sustain_sec:
            open_reason = "above_p95"
    resolve = False
    if fresh and open_reason is None:
        if armed and cfg.p95_opens and base.p75 is not None:
            resolve = newest.value < base.p75
        elif cfg.ceiling_rearm_c is not None:
            resolve = newest.value < cfg.ceiling_rearm_c
    elif cfg.lost_sec is not None and (newest is None or _age(now, newest.ts) > cfg.lost_sec):
        resolve = True
        detail["resolve_reason"] = "sensor_lost"
    return Verdict(open_reason, resolve, detail)


# --- shed (while a cooling incident is open) --------------------------------------------------

@dataclass(frozen=True)
class ShedRuleConfig:
    rise_c: float = 1.0
    window_sec: float = 900.0
    elevated_c: float = DEFAULT_ELEVATED_C
    max_age_sec: float = DEFAULT_MAX_READING_AGE_SEC


@dataclass(frozen=True)
class ShedVerdict:
    requested: bool
    reason: str | None            # cabinet_elevated | cabinet_rising | cabinet_unreadable
    temp_c: float | None
    rise_c: float | None


def cabinet_rise_c(cabinet: Sequence[TempPoint], now: datetime, window_sec: float = 900.0) -> float | None:
    """How much the cabinet has warmed within the last ``window_sec``: newest reading minus the
    lowest reading in ``[now - window_sec, now]``, in C (>= 0). None when no finite reading falls
    in the window. No threshold here: the caller applies its own.

    Shared on purpose. The hardware-watch reflex (``shed_verdict``, threshold 1.0 C) and Orion's
    later learned ``shed_background_gpu`` trigger (attend-to-act loop, threshold 0.5 C;
    docs/superpowers/specs/2026-09-29-attend-to-act-loop-design.md "Amendment 2026-09-29") must
    agree on what "rising" means, so the learned action imports this instead of writing its own.
    ``cabinet``: athena's ``cabinet_temp_c`` readings, ascending. Does not check freshness: a
    caller that needs "fresh" checks the newest reading's age itself.
    """
    window = [p.value for p in cabinet if 0 <= _age(now, p.ts) <= window_sec and math.isfinite(p.value)]
    if not window:
        return None
    newest = next(p.value for p in reversed(cabinet) if 0 <= _age(now, p.ts) <= window_sec and math.isfinite(p.value))
    return round(newest - min(window), 2)


def shed_verdict(cabinet: Sequence[TempPoint], now: datetime,
                 cfg: ShedRuleConfig = ShedRuleConfig()) -> ShedVerdict:
    """Is the cabinet warming? ``cabinet``: athena's cabinet_temp_c readings, ascending.

    Rise = ``cabinet_rise_c`` (newest - lowest reading in the last window_sec). An unreadable sensor (no reading within
    max_age_sec, thermal_gate's own staleness bound) counts as warming: with the AC down, a room
    nobody can measure is treated as hot, like the pool's thermal swap guard does."""
    newest = cabinet[-1] if cabinet else None
    if newest is None or _age(now, newest.ts) > cfg.max_age_sec or not math.isfinite(newest.value):
        return ShedVerdict(True, "cabinet_unreadable", None, None)
    rise = cabinet_rise_c(cabinet, now, cfg.window_sec)
    if newest.value >= cfg.elevated_c:
        return ShedVerdict(True, "cabinet_elevated", newest.value, rise)
    if rise is not None and rise >= cfg.rise_c:
        return ShedVerdict(True, "cabinet_rising", newest.value, rise)
    return ShedVerdict(False, None, newest.value, rise)
