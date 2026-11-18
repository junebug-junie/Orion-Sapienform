"""Learning progress: are Orion's forecasts getting better or worse? Pure: no I/O.

Spec: docs/superpowers/specs/2026-11-13-choice-as-expected-free-energy-design.md, "Step 1: learning progress as a
need". Theory: Oudeyer and Kaplan's learning progress, the drop in error over time, not low error. Here the error
is forecast skill against the baseline, so a harder day (weather, a host coming back) moves both and cancels:

    skill(day) = 1 - MAE(champion) / MAE(baseline)      paired rows, one horizon, one UTC day
    learning_progress = mean skill(last 7 days) - mean skill(prior 7 days)

Rules, each from the metric gate (recorded in the PR report):

* **Only the gated series.** The gate passed for cabinet and office temperature only (``SERIES``). Chat bets and
  the legacy prediction-error nodes failed it.
* **Same models on both sides.** A change to the set of models fitted for a series (the roster) is our deploy, not
  Orion learning: cabinet's skill stepped 0.09 -> 0.30 the day ``direct_arx_overlap`` joined (2026-11-03). Only days
  after the last roster change count. Too few such days on either side: ``insufficient``, never 0.
* **Rest is a real state.** Daily skill swings by about 0.15 with how hard the day was. A difference of window means
  is ``flat`` unless it is at least ``REST_T`` standard errors from 0 (Welch, days as samples). Flat is the calm
  reading: a calibrated forecaster that is not improving reads flat in week 1 and in week 10.
* **Too small to matter reads flat too.** While the champion serves the baseline, skill sits at about 0 with almost
  no spread, so a 0.02 drift passes the t rule (cabinet "worse" on 2026-11-02, live replay). A change must also be
  at least ``MIN_EFFECT`` in skill.
* **Complete days only.** Today is partial and is never used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal, Optional, Sequence

SERIES: tuple[str, ...] = ("athena.cabinet_temp_c", "office.temp_c")
HORIZON_MIN = 15  # the gate's horizon
WINDOW_DAYS = 7
MIN_DAYS = 5  # usable days needed in each window
MIN_PAIRS = 200  # paired scored forecasts needed for a day to count (of 288 at a 5 min grain)
REST_T = 2.0  # |t| below this reads flat (about 95% two-sided)
MIN_EFFECT = 0.05  # [default] |progress| below this reads flat: 5% of the baseline's error

Direction = Literal["better", "worse", "flat"]


@dataclass(frozen=True)
class DayStat:
    day: date
    pairs: int
    mae_champion: float
    mae_baseline: float

    @property
    def skill(self) -> Optional[float]:
        if self.pairs < MIN_PAIRS or not self.mae_baseline > 0 or not math.isfinite(self.mae_champion):
            return None
        return 1.0 - self.mae_champion / self.mae_baseline


@dataclass(frozen=True)
class Window:
    start: date
    end: date  # inclusive
    days: int
    mean_skill: Optional[float]


@dataclass(frozen=True)
class SeriesProgress:
    series_id: str
    status: Literal["ok", "insufficient"]
    reason: Optional[str]
    learning_progress: Optional[float]
    se: Optional[float]
    t: Optional[float]
    direction: Optional[Direction]
    recent: Window
    prior: Window
    roster_changed_on: Optional[date]


def last_roster_change(rosters: dict[date, frozenset[str]]) -> Optional[date]:
    """The latest day whose fitted model set differs from the previous day that had fits."""
    change, prev = None, None
    for d in sorted(rosters):
        if prev is not None and rosters[d] != prev:
            change = d
        prev = rosters[d]
    return change


def _mean(xs: Sequence[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def _var(xs: Sequence[float]) -> float:
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def progress(series_id: str, days: Sequence[DayStat], rosters: dict[date, frozenset[str]],
             today: date) -> SeriesProgress:
    """Learning progress of one series as of ``today`` (UTC; today itself is excluded)."""
    recent_start, recent_end = today - timedelta(days=WINDOW_DAYS), today - timedelta(days=1)
    prior_start, prior_end = today - timedelta(days=2 * WINDOW_DAYS), today - timedelta(days=WINDOW_DAYS + 1)
    change = last_roster_change(rosters)

    def usable(lo: date, hi: date) -> list[float]:
        out = []
        for d in days:
            if not lo <= d.day <= hi or (change is not None and d.day <= change):
                continue
            s = d.skill
            if s is not None:
                out.append(s)
        return out

    recent, prior = usable(recent_start, recent_end), usable(prior_start, prior_end)
    rw = Window(recent_start, recent_end, len(recent), _mean(recent))
    pw = Window(prior_start, prior_end, len(prior), _mean(prior))
    if len(recent) < MIN_DAYS or len(prior) < MIN_DAYS:
        why = (f"usable days: recent {len(recent)}, prior {len(prior)}; need {MIN_DAYS} each"
               + (f" (only days after the {change.isoformat()} model change count)" if change else ""))
        return SeriesProgress(series_id, "insufficient", why, None, None, None, None, rw, pw, change)
    lp = rw.mean_skill - pw.mean_skill  # type: ignore[operator]
    se = math.sqrt(_var(recent) / len(recent) + _var(prior) / len(prior))
    t = lp / se if se > 0 else (0.0 if lp == 0 else math.copysign(math.inf, lp))
    moved = abs(lp) >= MIN_EFFECT
    direction: Direction = "better" if moved and t >= REST_T else "worse" if moved and t <= -REST_T else "flat"
    return SeriesProgress(series_id, "ok", None, lp, se, t if math.isfinite(t) else None, direction, rw, pw, change)
