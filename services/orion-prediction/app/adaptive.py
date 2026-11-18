"""Adaptive champion ranges: calibrated on recent live scores, scaled by how hard the last hour was.

Why: the backtest pool gives every hour one width. Live 2026-11-10 that read calibrated on average
(80 % coverage ~0.84) but far too wide on quiet days, so the surprise calm check failed (quiet 24 h
excess -0.3..-0.8). Scaling by how fast the READING moved (mean |diff| over the last hour) was tried
first and overcorrects: smooth ramps the model tracks read "busy" (busy-tercile excess -0.44,
calm +0.19, cabinet 15 min). Scaling by how badly the FORECASTER missed in the last hour works.

Method: normalized split conformal ("locally weighted", Lei et al. 2018). Score = residual /
difficulty, where difficulty = mean |actual - point| of this series' champion forecasts at its
smallest horizon whose target fell in the hour before the origin (all already known at issue time).
For any difficulty computed from past data, exchangeable scores keep the conformal guarantee; the
scale only moves width between calm and busy hours. The calibration pool is the last POOL of
scored champion rows, rolling, so slow drift (a quieter week) is absorbed too.

Replay (evals/run_adaptive_interval_replay_eval.py, this code, live rows 10-27..11-10, the surprise
calm check per day, same days and same forecasts for both): days calm at 5/15 min -- cabinet 13/13,
12/13 (served backtest ranges 2/13, 7/13), office 6/6, 6/6 (served 1/6, 2/6); 80 % coverage
0.76-0.81 (served 0.85-0.93). Only horizons up to MAX_HORIZON are adaptive (see there).
Caveat: the method and the 15 min cut were chosen on these same 14 days; live is the holdout.

Falls back to the backtest pool (today's behavior) per horizon when the live pool or the current
difficulty is too thin, and always beyond MAX_HORIZON. Exceedance probabilities still use the backtest paths (they need whole paths
across steps; the live pool has one row per horizon).
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from app.backtest import residual_quantiles

POOL = timedelta(days=3)
DIFFICULTY_WINDOW = timedelta(minutes=60)
MIN_DIFFICULTY_N = 6          # scored smallest-horizon forecasts inside the window
MIN_POOL = 288                # one day of 5-min origins per horizon
FLOOR_Q = 0.05                # difficulty floor: this quantile of the pool's difficulties
# Longest horizon served adaptively. Replay 11-10 on matched days, owner-filtered pools: 60 min went
# TOO NARROW (office excess +0.33, cov80 0.73; cabinet +0.12) while its backtest range was already
# near calm (office +0.09, cabinet -0.02); 360 min was worse than backtest too. 5 and 15 min gain.
MAX_HORIZON = timedelta(minutes=15)


def difficulty_series(rows: Iterable[Mapping], smallest_h: int) -> pd.Series:
    """Rolling mean |miss| of the smallest-horizon forecasts, indexed by target time (when it became
    known). NaN where fewer than MIN_DIFFICULTY_N fell in the window."""
    pts = [(pd.Timestamp(r["origin_ts"]) + pd.Timedelta(minutes=smallest_h), abs(float(r["actual"]) - float(r["point"])))
           for r in rows if int(r["horizon_min"]) == smallest_h and r.get("actual") is not None]
    if not pts:
        return pd.Series(dtype=float)
    s = pd.Series([v for _, v in pts], index=pd.DatetimeIndex([t for t, _ in pts])).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.rolling(DIFFICULTY_WINDOW, min_periods=MIN_DIFFICULTY_N).mean()


def difficulty_at(diff: pd.Series, times: pd.DatetimeIndex, max_age: timedelta) -> np.ndarray:
    """Difficulty known at each of ``times``: the latest value with target <= t, NaN when none or
    older than ``max_age`` (a gap in scoring must not carry an old hour forward)."""
    out = np.full(len(times), np.nan)
    if diff.empty or not len(times):
        return out
    idx = diff.index.as_unit("ns").asi8
    t = times.as_unit("ns").asi8
    pos = np.searchsorted(idx, t, side="right") - 1
    ok = pos >= 0
    age_ok = np.zeros(len(t), dtype=bool)
    age_ok[ok] = (t[ok] - idx[pos[ok]]) <= pd.Timedelta(max_age).value
    out[age_ok] = diff.to_numpy()[pos[age_ok]]
    return out


FALLBACK_SOURCE = "backtest:model_fallback"


def adaptive_offsets(rows: list[Mapping], origin: pd.Timestamp, horizons_min: Iterable[int],
                     grain_min: int, owners: Mapping[int, str] | None = None
                     ) -> tuple[dict[int, np.ndarray], dict[int, str]]:
    """Per horizon: residual offsets at QUANTILES for a forecast issued at ``origin``, and why a
    horizon fell back. ``rows``: scored champion rows (origin_ts, horizon_min, point, actual, and
    optionally ``owner`` = the model that served that horizon, ``interval_source``) with targets
    inside POOL before ``origin``.

    ``owners`` (horizon -> model serving it now): a horizon's pool keeps only rows that model
    served. Different models miss by different amounts (live 11-10, office 60 min: direct_ar MAE
    0.48 vs mstl_arima 0.26), and the 60 min model switched 8 times in 4 days, so a mixed pool sizes
    the range for the wrong model. Difficulty is NOT owner-filtered: it measures how hard the hour
    was, and refiltering it would void the whole pool on every 5-min model switch. Rows a failed
    model handed to the baseline (``interval_source`` = FALLBACK_SOURCE) are excluded everywhere."""
    horizons = sorted(int(h) for h in horizons_min)
    rows = [r for r in rows if r.get("interval_source") != FALLBACK_SOURCE]
    diff = difficulty_series(rows, horizons[0])
    max_age = timedelta(minutes=2 * grain_min)
    now_d = difficulty_at(diff, pd.DatetimeIndex([pd.Timestamp(origin)]), max_age)[0]
    if not np.isfinite(now_d):
        return {}, {hm: "no_recent_difficulty" for hm in horizons}
    rows = [r for r in rows if r.get("actual") is not None]
    if not rows:
        return {}, {hm: f"live_pool_0_of_{MIN_POOL}" for hm in horizons}
    h = np.array([int(r["horizon_min"]) for r in rows])
    owner = np.array([str(r.get("owner")) for r in rows])
    res = np.array([float(r["actual"]) - float(r["point"]) for r in rows])
    ds = difficulty_at(diff, pd.DatetimeIndex([pd.Timestamp(r["origin_ts"]) for r in rows]), max_age)
    offsets: dict[int, np.ndarray] = {}
    reasons: dict[int, str] = {}
    for hm in horizons:
        if timedelta(minutes=hm) > MAX_HORIZON:
            reasons[hm] = "horizon_not_adaptive"
            continue
        m = (h == hm) & np.isfinite(ds)
        if owners is not None:
            m &= owner == str(owners.get(hm))
        n = int(m.sum())
        if n < MIN_POOL:
            reasons[hm] = f"live_pool_{n}_of_{MIN_POOL}" + ("_same_model" if owners is not None else "")
            continue
        # Floor: a flat or quantized hour (difficulty ~0) must not blow a score up to ~1e8.
        floor = max(float(np.quantile(ds[m], FLOOR_Q)), 0.1 * float(np.median(ds[m])), 1e-9)
        scores = res[m] / np.maximum(ds[m], floor)
        offsets[hm] = residual_quantiles(scores[:, None])[:, 0] * max(float(now_d), floor)
    return offsets, reasons


def pool_since(origin: datetime) -> datetime:
    return origin - POOL

