"""Learning progress: are Orion's forecasts getting better or worse? Pure: no I/O.

Spec: docs/superpowers/specs/2026-11-13-choice-as-expected-free-energy-design.md, "Step 1: learning progress as a
need". Theory: Oudeyer and Kaplan's learning progress, the drop in error over time, not low error. Here the error
is forecast skill against the baseline, so a harder day (weather, a host coming back) moves both and cancels:

    skill(day) = 1 - MAE(champion) / MAE(baseline)      paired rows, one horizon, one UTC day
    learning_progress = mean skill(last 7 days) - mean skill(prior 7 days)

Rules, each from the metric gate (recorded in the PR report):

* **Only the gated series.** The gate passed for cabinet and office temperature only (``SERIES``). Chat bets and
  the legacy prediction-error nodes failed it.
* **Same models on both sides.** A change to how a series is forecast is our deploy, not Orion learning: cabinet's
  skill stepped 0.09 -> 0.30 the day ``direct_arx_overlap`` joined (2026-11-03). The roster key per day is the set
  of fitted model names plus the selection's config as stored on the day's last ``champion_blend`` run: the gate
  rule version (``gate_version``) and the covariate lists (``covariates``, ``overlap_covariates``; cabinet widened
  them under the same names on 2026-10-21). Built in ``app/store.py`` ``daily_rosters``. Known gap: a change to a
  model's own code under the same name is still not seen, because no table records a code or image version
  (issue #3691). Only days after the last roster change count. Too few such days on either side: ``insufficient``,
  never 0.
* **Rest is a real state.** Daily skill swings by about 0.15 with how hard the day was. A difference of window means
  is ``flat`` unless its t passes the two-sided ``ALPHA`` critical value of Student's t at the Welch-Satterthwaite
  degrees of freedom (5-7 days a side: about 6-12 df, a cut of 2.2-2.4, not the normal 2.0). Flat is the calm
  reading: a calibrated forecaster that is not improving reads flat in week 1 and in week 10.
* **Weather spells are not independent days.** Daily skill is autocorrelated (live lag-1 2026-11-18: cabinet +0.07
  over 14 days, office +0.21 over 12; the plain estimate is biased low by about (1 + 4 rho) / n). Treating days as
  independent made the rule false-move about 9-12% per look at rho 0.2-0.3, not 5%. So the standard error is the
  exact one for an AR(1) day-to-day correlation ``rho``: both window means, their covariance (adjacent windows), each
  window's variance rescaled for the bias of the sample variance under ``rho``, and each window's df from the same
  quadratic form. ``rho`` comes from the longer history (``RHO_HISTORY_DAYS``, same roster only): residuals of each
  7-day block after removing its own mean and linear trend (so a real trend or step does not read as persistence),
  lag-1 over consecutive days, then the ``rho`` whose exact expected value of that statistic matches (moment
  matching, so it is not biased low at small n), clamped to [0, ``RHO_MAX``], then shrunk toward ``RHO_PRIOR``
  (worth ``RHO_PRIOR_WEIGHT`` pairs); under ``RHO_MIN_PAIRS`` pairs it is ``RHO_PRIOR``. At ``rho`` = 0 every number equals the plain Welch test. Simulated null false-move rate per daily
  look (``evals/run_learning_progress_null_eval.py``, 2026-11-18), 14 or 56 days of history: 1.7-5.1% at rho 0-0.3
  (Welch 4.0-13.3%), 6.9-9.4% at rho 0.5 (Welch 20-21%). Known limit: the reading is still re-checked daily on
  overlapping windows. Prime reads each look's ``p_value``, so the 5% is a per-look rate; per week (any of 7 looks)
  it is about 12-21% at rho 0-0.3 (Welch 19-47%).
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

import numpy as np
from scipy.stats import t as student_t

SERIES: tuple[str, ...] = ("athena.cabinet_temp_c", "office.temp_c")
HORIZON_MIN = 15  # the gate's horizon
WINDOW_DAYS = 7
MIN_DAYS = 5  # usable days needed in each window
MIN_PAIRS = 200  # paired scored forecasts needed for a day to count (of 288 at a 5 min grain)
ALPHA = 0.05  # two-sided: |t| below Student's t critical value at the Welch df reads flat
MIN_EFFECT = 0.05  # [default] |progress| below this reads flat: 5% of the baseline's error
RHO_HISTORY_DAYS = 56  # days of history (same roster) the day-to-day correlation is estimated from
RHO_BLOCK_DAYS = 7  # residual blocks, aligned to the comparison windows
RHO_MIN_PAIRS = 12  # consecutive-day residual pairs needed to estimate rho; fewer: RHO_PRIOR
RHO_PRIOR = 0.3  # [default] live 2026-11-18, 12 pairs each: moment-matched 0.37 (cabinet), 0.00 (office); raw
#                  lag-1 +0.07 / +0.21. In simulation the Welch rule's false moves double by rho 0.3.
RHO_PRIOR_WEIGHT = 24  # [default] the prior counts as this many day pairs: 12 live pairs alone swing 0 <-> 0.4
RHO_MAX = 0.6  # [default] clamp: 7-day blocks cannot tell rho above this apart

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
    t_crit: Optional[float]
    p_value: Optional[float]  # two-sided, Student's t at the Welch df (not "P(the change is real)")
    direction: Optional[Direction]
    recent: Window
    prior: Window
    roster_changed_on: Optional[date]
    # Day-to-day correlation used for the standard error; how many consecutive-day residual pairs informed it; and
    # where it came from: "estimated" (history, shrunk toward RHO_PRIOR), "prior" (too few pairs), "given" (caller).
    # ``se_iid`` is the plain Welch SE, for comparison.
    rho: Optional[float] = None
    rho_pairs: Optional[int] = None
    se_iid: Optional[float] = None
    rho_source: Optional[Literal["estimated", "prior", "given"]] = None


def last_roster_change(rosters: dict[date, frozenset[str]]) -> Optional[date]:
    """The latest day whose fitted model set differs from the previous day that had fits."""
    change, prev = None, None
    for d in sorted(rosters):
        if prev is not None and rosters[d] != prev:
            change = d
        prev = rosters[d]
    return change


def roster_keys(model_rows: Sequence[dict], config_rows: Sequence[dict]) -> list[dict]:
    """Per series and day: fitted model names plus the selection config tokens, as ``{series_id, day, models}``.

    ``model_rows``: ``{series_id, day, models}`` (names fitted that day). ``config_rows``: ``{series_id, day,
    gate_version, covariates, overlap_covariates}`` from the day's last ``champion_blend`` run (values as stored,
    None when absent). A day with fits but no blend run keeps the previous config. Same output shape as before
    the config tokens, so a change of gate rule or covariates under the same model names reads as a roster change."""
    config = {(r["series_id"], r["day"]): r for r in config_rows}
    out = []
    for sid in sorted({r["series_id"] for r in model_rows}):
        last: tuple[str, ...] = ()
        days = {r["day"]: r for r in model_rows if r["series_id"] == sid}
        cfg_days = sorted(d for (s, d) in config if s == sid)
        for day in sorted(days):
            for d in cfg_days:  # latest config at or before this day
                if d <= day:
                    c = config[(sid, d)]
                    last = tuple(f"{k}={c.get(k)}" for k in ("gate_version", "covariates", "overlap_covariates"))
            out.append({"series_id": sid, "day": day, "models": sorted(set(days[day]["models"]) | set(last))})
    return out


def _mean(xs: Sequence[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def _var(xs: Sequence[float]) -> float:
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def _corr(a: Sequence[int], b: Sequence[int], rho: float) -> np.ndarray:
    """AR(1) correlation between days (ordinals) ``a`` and ``b``: rho ** |gap|."""
    gap = np.abs(np.subtract.outer(np.asarray(a, float), np.asarray(b, float)))
    return np.where(gap == 0, 1.0, rho ** gap) if rho > 0 else (gap == 0).astype(float)


def _residual_maker(ts: Sequence[int]) -> np.ndarray:
    """Projection off a block's own mean and linear trend."""
    t = np.asarray(ts, float)
    x = np.column_stack([np.ones_like(t), t - t.mean()])
    return np.eye(len(t)) - x @ np.linalg.pinv(x)


def moment_rho(points: Sequence[tuple[int, float]], end: int) -> tuple[Optional[float], int]:
    """Day-to-day AR(1) correlation of skill, from ``(day ordinal, skill)`` points before ``end``, unshrunk.

    Each ``RHO_BLOCK_DAYS`` block (aligned to ``end``) loses its own mean and linear trend, so a step between
    windows or a real trend does not read as persistence. The statistic is lag-1 over consecutive days pooled over
    blocks; ``rho`` is the value whose exact AR(1) expectation of that statistic (ratio of quadratic-form
    expectations) equals it, clamped to [0, ``RHO_MAX``]. Returns (rho, pairs); (None, pairs) under
    ``RHO_MIN_PAIRS`` pairs or with no residual spread."""
    by_block: dict[int, list[tuple[int, float]]] = {}
    for o, v in points:
        if o < end:
            by_block.setdefault((end - 1 - o) // RHO_BLOCK_DAYS, []).append((o, v))
    parts, num, den, pairs = [], 0.0, 0.0, 0
    for blk in by_block.values():
        if len(blk) < 4:  # mean + trend leave too little
            continue
        blk.sort()
        ts = [o for o, _ in blk]
        m = _residual_maker(ts)
        lag = np.zeros((len(ts), len(ts)))
        for i in range(len(ts) - 1):
            if ts[i + 1] - ts[i] == 1:
                lag[i, i + 1] = lag[i + 1, i] = 0.5
                pairs += 1
        e = m @ np.asarray([v for _, v in blk], float)
        num, den = num + float(e @ lag @ e), den + float(e @ e)
        parts.append((ts, m, m @ lag @ m))
    if pairs < RHO_MIN_PAIRS or not den > 0:
        return None, pairs
    r = num / den

    def expected(rho: float) -> float:
        cs = [_corr(ts, ts, rho) for ts, _, _ in parts]
        return (sum(float(np.trace(ml @ c)) for (_, _, ml), c in zip(parts, cs))
                / sum(float(np.trace(m @ c)) for (_, m, _), c in zip(parts, cs)))

    if r <= expected(0.0):
        return 0.0, pairs
    if r >= expected(RHO_MAX):
        return RHO_MAX, pairs
    lo, hi = 0.0, RHO_MAX
    for _ in range(40):  # expected() rises with rho
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if expected(mid) < r else (lo, mid)
    return (lo + hi) / 2, pairs


def estimate_rho(points: Sequence[tuple[int, float]], end: int) -> tuple[float, int, Literal["estimated", "prior"]]:
    """``moment_rho`` shrunk toward ``RHO_PRIOR`` (weight ``RHO_PRIOR_WEIGHT`` pairs); the prior alone when it has
    no estimate. 12 pairs (two weeks, live today) give a third of the weight to the data, 48 pairs two thirds:
    without shrinkage a 12-pair estimate clamps to 0 (plain Welch) about 30% of the time."""
    r, pairs = moment_rho(points, end)
    if r is None:
        return RHO_PRIOR, pairs, "prior"
    return (pairs * r + RHO_PRIOR_WEIGHT * RHO_PRIOR) / (pairs + RHO_PRIOR_WEIGHT), pairs, "estimated"


def _window_moments(ts: Sequence[int], xs: Sequence[float], rho: float) -> tuple[float, float, float]:
    """(day variance, variance of the window mean, its df) for one window under AR(1) ``rho``.
    At rho 0: (s^2, s^2/n, n - 1)."""
    n = len(ts)
    c = _corr(ts, ts, rho)
    ac = (np.eye(n) - 1.0 / n) @ c
    tr = float(np.trace(ac))
    x = np.asarray(xs, float)
    s2 = float(((x - x.mean()) ** 2).sum()) / tr  # unbiased day variance under rho
    return s2, s2 * float(c.sum()) / n ** 2, tr ** 2 / float(np.trace(ac @ ac))


def progress(series_id: str, days: Sequence[DayStat], rosters: dict[date, frozenset[str]],
             today: date, rho: Optional[float] = None) -> SeriesProgress:
    """Learning progress of one series as of ``today`` (UTC; today itself is excluded).

    ``rho``: the day-to-day correlation to use; None estimates it from ``days`` (``estimate_rho``)."""
    recent_start, recent_end = today - timedelta(days=WINDOW_DAYS), today - timedelta(days=1)
    prior_start, prior_end = today - timedelta(days=2 * WINDOW_DAYS), today - timedelta(days=WINDOW_DAYS + 1)
    change = last_roster_change(rosters)

    def usable(lo: date, hi: date) -> list[tuple[int, float]]:
        out = []
        for d in sorted(days, key=lambda d: d.day):
            if not lo <= d.day <= hi or (change is not None and d.day <= change):
                continue
            s = d.skill
            if s is not None:
                out.append((d.day.toordinal(), s))
        return out

    recent_pts, prior_pts = usable(recent_start, recent_end), usable(prior_start, prior_end)
    recent, prior = [v for _, v in recent_pts], [v for _, v in prior_pts]
    rw = Window(recent_start, recent_end, len(recent), _mean(recent))
    pw = Window(prior_start, prior_end, len(prior), _mean(prior))
    if len(recent) < MIN_DAYS or len(prior) < MIN_DAYS:
        why = (f"usable days: recent {len(recent)}, prior {len(prior)}; need {MIN_DAYS} each"
               + (f" (only days after the {change.isoformat()} model change count)" if change else ""))
        return SeriesProgress(series_id, "insufficient", why, None, None, None, None, None, None, rw, pw, change)
    if rho is None:
        history = usable(today - timedelta(days=RHO_HISTORY_DAYS), recent_end)
        rho, rho_pairs, rho_source = estimate_rho(history, today.toordinal())
    else:
        rho_pairs, rho_source = 0, "given"
    lp = rw.mean_skill - pw.mean_skill  # type: ignore[operator]
    se_iid = math.sqrt(_var(recent) / len(recent) + _var(prior) / len(prior))
    rt, pt = [o for o, _ in recent_pts], [o for o, _ in prior_pts]
    (sr, vr, dfr), (sp, vp, dfp) = _window_moments(rt, recent, rho), _window_moments(pt, prior, rho)
    # Adjacent windows share weather spells: their means co-move, which shrinks the variance of the difference.
    cov = math.sqrt(sr * sp) * float(_corr(rt, pt, rho).sum()) / (len(recent) * len(prior))
    se = math.sqrt(max(vr + vp - 2 * cov, 0.0))
    if se > 0:
        t = lp / se
        df = se ** 4 / ((vr ** 2 / dfr if vr else 0.0) + (vp ** 2 / dfp if vp else 0.0))
        t_crit = float(student_t.ppf(1 - ALPHA / 2, df))
        p_value = float(2 * student_t.sf(abs(t), df))
    else:  # no spread on either side: any difference is exact
        t, t_crit = (0.0 if lp == 0 else math.copysign(math.inf, lp)), 0.0
        p_value = 1.0 if lp == 0 else 0.0
    moved = abs(lp) >= MIN_EFFECT and abs(t) > t_crit
    direction: Direction = ("better" if lp > 0 else "worse") if moved else "flat"
    return SeriesProgress(series_id, "ok", None, lp, se, t if math.isfinite(t) else None, t_crit, p_value, direction,
                          rw, pw, change, rho, rho_pairs, se_iid, rho_source)
