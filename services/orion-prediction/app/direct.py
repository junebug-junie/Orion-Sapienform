"""Direct multi-horizon models with covariates: predict y[t+k] - y[t] from what is known at t.

"Direct" means one regression per lead time k, trained on every past origin t. Every feature is
measured at or before the origin, so serving never needs a future covariate value -- nothing has
to forecast chassis watts or AC watts first, and nothing can leak the future into the fit.

Features at origin t (for each lead k):
- target: y_t - y_{t-1}, y_t - y_{t-3}, y_t - y_{t-12}, y_t - mean(y over 1 h),
  y_t - y_{t-1 day} when a day of history exists
- each covariate: x_t, mean over 30 min, mean over 2 h, x_t - mean over 2 h
- time of day of the TARGET bucket (sin/cos of the daily phase at t+k)

Models are fitted at anchor leads (1, 2, 3, 6, 12, 36, 72, 144, 288 steps, capped at h, plus h)
and the path is linearly interpolated between anchors, so a 288-step path costs ~10 fits, not 288.

Two variants, both ridge regression (numpy, closed form), so the scoreboard measures what the
covariates add instead of assuming it:
- ``direct_ar``: target-history features only.
- ``direct_arx``: the same plus every declared covariate.
On the real cabinet history (2026-10-16 replay, 40 windows, MAE vs persistence) the target-history
features carry the short-horizon gain (0.79 @5 min with no covariates); covariates add only at 1 h
(0.98 -> 0.93, mostly the AC). A gradient-boosting variant was measured and dropped: it lost at
every horizon and cost ~6 min of CPU per selection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANCHORS = (1, 2, 3, 6, 12, 36, 72, 144, 288)
MIN_ROWS = 288           # one day of complete origins per lead
MAX_COVARIATE_GAPS = 0.10
RIDGE_ALPHA = 1.0


def _roll_mean(a: np.ndarray, k: int) -> np.ndarray:
    return pd.Series(a).rolling(k, min_periods=1).mean().to_numpy()


def _lag(a: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(a, np.nan)
    out[k:] = a[:-k]
    return out


def origin_features(y: np.ndarray, X: np.ndarray, grain_min: int) -> np.ndarray:
    """(n, f) features at each origin index, from y (n,) and covariates X (n, k). NaN where undefined."""
    hour = max(1, 60 // grain_min)
    day = 1440 // grain_min
    cols = [y - _lag(y, 1), y - _lag(y, 3), y - _lag(y, 12), y - _roll_mean(y, hour)]
    if len(y) > day:
        cols.append(y - _lag(y, day))
    for j in range(X.shape[1]):
        x = X[:, j]
        m2h = _roll_mean(x, max(1, 120 // grain_min))
        cols += [x, _roll_mean(x, max(1, 30 // grain_min)), m2h, x - m2h]
    return np.column_stack(cols)


def _tod(index: pd.DatetimeIndex, lead_min: int) -> np.ndarray:
    minutes = (index.asi8 // 60_000_000_000) + lead_min
    phase = 2 * np.pi * (minutes % 1440) / 1440.0
    return np.column_stack([np.sin(phase), np.cos(phase)])


def _anchors(h: int) -> list[int]:
    return sorted({a for a in ANCHORS if a <= h} | {h})


def _ridge_fit_predict(F: np.ndarray, t: np.ndarray, f_now: np.ndarray) -> float:
    mu, sd = F.mean(axis=0), F.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (F - mu) / sd
    Z1 = np.column_stack([np.ones(len(Z)), Z])
    reg = RIDGE_ALPHA * np.eye(Z1.shape[1])
    reg[0, 0] = 0.0
    beta = np.linalg.solve(Z1.T @ Z1 + reg, Z1.T @ t)
    return float(np.concatenate([[1.0], (f_now - mu) / sd]) @ beta)


def _direct(grain_min: int, with_covariates: bool):
    def fn(y: np.ndarray, h: int, idx: pd.DatetimeIndex, X: np.ndarray | None = None) -> np.ndarray:
        if not with_covariates:
            X = np.empty((len(y), 0))
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] != len(y):
            raise ValueError("covariates must be (n, k) aligned with y")
        gaps = float(np.mean(~np.isfinite(X))) if X.size else 0.0
        if gaps > MAX_COVARIATE_GAPS:
            raise ValueError(f"covariates are {gaps:.0%} gaps in the training window")
        X = pd.DataFrame(X).interpolate(limit_direction="both").to_numpy()
        base = origin_features(y, X, grain_min)
        now = base[-1]
        if not np.all(np.isfinite(now)):
            raise ValueError("features at the origin are undefined")
        anchors = _anchors(h)
        deltas = []
        for k in anchors:
            F = np.column_stack([base[:-k], _tod(idx[:-k], k * grain_min)])
            target = y[k:] - y[:-k]
            ok = np.all(np.isfinite(F), axis=1) & np.isfinite(target)
            if ok.sum() < MIN_ROWS:
                raise ValueError(f"only {int(ok.sum())} complete training rows at lead {k}")
            f_now = np.concatenate([now, _tod(idx[-1:], k * grain_min)[0]])
            deltas.append(_ridge_fit_predict(F[ok], target[ok], f_now))
        path = np.interp(np.arange(1, h + 1), anchors, deltas)
        return y[-1] + path

    return fn


def direct_ar(grain_min: int):
    return _direct(grain_min, with_covariates=False)


def direct_arx(grain_min: int):
    return _direct(grain_min, with_covariates=True)
