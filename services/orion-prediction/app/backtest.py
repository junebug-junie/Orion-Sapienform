"""Rolling-origin backtest, split-conformal intervals, exceedance probability, and the champion gate.

Backtest: for each origin ``o`` (the last ``n_windows`` origins, ``step`` buckets apart, the newest
one ``h`` buckets before the end) fit on the training view up to ``o`` and forecast ``h`` steps.
The residual path is ``actual - forecast`` per step; NaN where the actual bucket is a gap.

Conformal: the interval at step j is ``point_j + empirical quantile of residuals at step j``, with
the finite-sample level correction. Coverage is reported on a time-ordered holdout (see horizon_metrics).

Gate, per horizon: the reference is the *better* of the two baselines at that horizon (persistence
wins short horizons, same-time-yesterday can win long ones; live 2026-10-15 cabinet backtest:
naive MAE 0.08 C vs seasonal-naive 1.87 C at 5 min). A model serves a horizon only when its MAE
beats that reference on the windows both produced a residual for (``rel_mae < 1``, and on
>= 90% of the reference's windows) AND the win is significant: one-sided paired t-test over
windows, p < ``GATE_ALPHA``. Lowest-average-wins on a near tie is how noise gets served. Different horizons may be served by different models; the served path is
stitched step by step, and so are its residual paths, so the conformal intervals stay matched.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from app.frame import train_ready, training_slice
from app.models import BASELINES, Forecaster, split_seasonalities
from app.series import Seasonality, SeriesSpec

QUANTILES = (0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975)
# Live-coverage alert band (engine). Not part of the gate: see horizon_metrics.
COVERAGE_BAND = (0.72, 0.88)
# A candidate serves a horizon only if it beats the better baseline with p < GATE_ALPHA (one-sided
# paired t-test over windows). Live 2026-10-16: the two served picks that lost to the baseline live
# (window_average_1h @1 h, 1.57x; mstl_arima @15 min, 2.0x) had backtest p = 0.21 and 0.31 -- lower
# averages by noise. No family correction across candidates; documented trade-off.
GATE_ALPHA = 0.05
MAX_FILLED_SHARE = 0.10
MIN_TRAIN_STEPS_DAYS = 2
MIN_PATHS = 10
# A candidate must have a residual on at least this share of the reference baseline's windows.
MIN_WINDOW_SHARE = 0.9


def q_key(q: float) -> str:
    return f"{q:g}"


@dataclass
class BacktestResult:
    model: str
    residuals: np.ndarray            # (n_windows, h); NaN = no actual / window skipped
    origins: list[pd.Timestamp]
    fit_seconds: float
    errors: list[str] = field(default_factory=list)

    @property
    def n_windows(self) -> int:
        return int(np.sum(np.any(np.isfinite(self.residuals), axis=1))) if self.residuals.size else 0


def origins_for(n: int, h: int, n_windows: int, step: int, min_train: int) -> list[int]:
    last = n - h
    out = [last - k * step for k in range(n_windows)]
    return sorted(o for o in out if o >= min_train)


def run_backtest(series: pd.Series, spec: SeriesSpec, name: str,
                 make_model: Callable[[list[Seasonality]], Forecaster], n_windows: int, step: int,
                 actuals: pd.Series | None = None, covariates: pd.DataFrame | None = None) -> BacktestResult:
    """``series``: the full regular view (short gaps filled). Training windows come from training_slice.

    ``make_model(usable)`` is called per window with the seasonalities *that window's* training data
    supports, so an early window never fits a cycle it has seen fewer than MIN_CYCLES times.
    ``actuals``: what residuals are measured against -- real readings only (no gap fill, exclusions
    masked), the same thing live scoring compares with. Defaults to ``series``.
    ``covariates``: columns in spec.covariates order, on ``series``' index, real buckets only (no gap
    fill, so nothing after an origin is interpolated into it); sliced to each window's training rows.
    """
    h = spec.max_horizon_steps
    min_train = MIN_TRAIN_STEPS_DAYS * 1440 // spec.grain_min
    idx = series.index
    truth = (actuals if actuals is not None else series).reindex(idx)
    origins = origins_for(len(series), h, n_windows, step, min_train)
    res = np.full((len(origins), h), np.nan)
    errors: list[str] = []
    t0 = time.monotonic()
    for i, o in enumerate(origins):
        train = training_slice(series, spec, end=idx[o])
        y, filled = train_ready(train)
        if len(y) < min_train or filled > MAX_FILLED_SHARE:
            continue
        try:
            usable, _ = split_seasonalities(spec, len(train))
            model = make_model(usable)
            X = covariates.reindex(train.index).to_numpy(dtype=float) if (
                model.uses_covariates and covariates is not None) else None
            pred = model.forecast(y, h, train.index, X)
        except Exception as exc:  # noqa: BLE001 -- one bad window is recorded, not fatal
            errors.append(f"{idx[o].isoformat()}: {type(exc).__name__}: {exc}"[:300])
            continue
        actual = truth.iloc[o:o + h].to_numpy(dtype=float)
        res[i, :len(actual)] = actual - pred[:len(actual)]
    return BacktestResult(name, res, [idx[o] for o in origins], time.monotonic() - t0, errors)


def _level(n: int, q: float) -> float:
    """Finite-sample split-conformal level: ceil((n+1)q)/n for the upper tail, mirrored for the lower.

    When ceil((n+1)q) > n the exact guarantee needs an infinite bound; this clips to the sample
    extreme instead (for q=0.975 that happens whenever n < 39). Coverage there is near, not at,
    nominal -- live scoring (in_95) is what measures it.
    """
    if n <= 0:
        return q
    if q >= 0.5:
        return min(1.0, math.ceil((n + 1) * q) / n)
    return max(0.0, 1.0 - min(1.0, math.ceil((n + 1) * (1 - q)) / n))


def residual_quantiles(residuals: np.ndarray, quantiles=QUANTILES) -> np.ndarray:
    """(len(quantiles), h) residual offsets per step. NaN where a step has < MIN_PATHS residuals."""
    h = residuals.shape[1]
    out = np.full((len(quantiles), h), np.nan)
    for j in range(h):
        col = residuals[:, j]
        col = col[np.isfinite(col)]
        if len(col) < MIN_PATHS:
            continue
        for k, q in enumerate(quantiles):
            # Order statistic, rounded outward (not interpolated inward).
            out[k, j] = np.quantile(col, _level(len(col), q), method="higher" if q >= 0.5 else "lower")
    # Monotone in level (guards rounding when levels collapse at small n).
    return np.maximum.accumulate(out, axis=0)


def quantile_paths(point: np.ndarray, residuals: np.ndarray, quantiles=QUANTILES) -> dict[str, np.ndarray]:
    offs = residual_quantiles(residuals, quantiles)
    h = len(point)
    return {q_key(q): point + offs[k, :h] for k, q in enumerate(quantiles)}


def exceedance_probability(point: np.ndarray, residuals: np.ndarray, threshold: float,
                           window_steps: int) -> tuple[float | None, int]:
    """Share of complete residual paths for which point+residual reaches threshold within the window."""
    w = min(window_steps, len(point), residuals.shape[1])
    paths = residuals[:, :w]
    complete = paths[np.all(np.isfinite(paths), axis=1)]
    if len(complete) < MIN_PATHS:
        return None, len(complete)
    hits = np.any(point[:w][None, :] + complete >= threshold, axis=1)
    return float(np.mean(hits)), len(complete)


def horizon_metrics(residuals: np.ndarray, step_idx: int) -> dict[str, float | int | None]:
    """Error metrics at one step, plus a TIME-ORDERED holdout coverage check.

    Coverage: interval quantiles from the older windows (all but the newest third), scored on the
    newest third. Leave-one-out coverage over the same pool is not used: for exchangeable residuals
    it depends only on ranks, so it is the same number for every model (0.85 at 40 windows, live
    2026-10-16) and can never tell a calibrated model from a broken one. The holdout can (drift,
    regime change), but it is noisy (~13 windows), so it is reported, not gated; live scoring is
    the calibration check that alerts.
    """
    col = residuals[:, step_idx]
    ok = np.isfinite(col)
    n = int(ok.sum())
    empty = {"n": n, "mae": None, "rmse": None, "bias": None, "coverage80_holdout": None,
             "coverage95_holdout": None, "width80": None, "n_holdout": 0}
    if n == 0:
        return empty
    vals = col[ok]
    out = {**empty, "mae": float(np.mean(np.abs(vals))), "rmse": float(np.sqrt(np.mean(vals ** 2))),
           "bias": float(np.mean(vals))}
    n_test = n // 3
    calib, test = vals[:n - n_test], vals[n - n_test:]   # rows are in origin order: oldest first
    if n_test and len(calib) >= MIN_PATHS:
        m = len(calib)
        lo80 = np.quantile(calib, _level(m, 0.1), method="lower")
        hi80 = np.quantile(calib, _level(m, 0.9), method="higher")
        lo95 = np.quantile(calib, _level(m, 0.025), method="lower")
        hi95 = np.quantile(calib, _level(m, 0.975), method="higher")
        out.update(coverage80_holdout=float(np.mean((test >= lo80) & (test <= hi80))),
                   coverage95_holdout=float(np.mean((test >= lo95) & (test <= hi95))),
                   width80=float(hi80 - lo80), n_holdout=int(n_test))
    return out


def paired_win_p(candidate: np.ndarray, reference: np.ndarray) -> tuple[float | None, int]:
    """One-sided paired t-test p-value that the candidate's absolute error is LOWER than the
    reference's, over the windows where both have a residual."""
    from scipy import stats

    both = np.isfinite(candidate) & np.isfinite(reference)
    d = np.abs(candidate[both]) - np.abs(reference[both])
    if len(d) < 2:
        return None, int(len(d))
    if np.all(d == d[0]):
        return (0.0 if d[0] < 0 else 1.0), int(len(d))
    return float(stats.ttest_1samp(d, 0.0, alternative="less").pvalue), int(len(d))


@dataclass
class Selection:
    """Per-horizon assignment. ``assignment[h]`` serves forecast steps in (previous horizon, h]."""
    assignment: dict[int, str]
    baseline_assignment: dict[int, str]
    served_baseline: bool
    metrics: dict[str, dict[str, dict]]   # model -> horizon_min (str) -> metrics (+ rel_mae, passed)

    def models_needed(self) -> set[str]:
        return set(self.assignment.values()) | set(self.baseline_assignment.values())


def step_owner(assignment: dict[int, str], grain_min: int, h_steps: int) -> list[str]:
    """Model name serving each forecast step 1..h_steps."""
    horizons = sorted(assignment)
    out = []
    for j in range(1, h_steps + 1):
        minute = j * grain_min
        owner = next((assignment[hm] for hm in horizons if minute <= hm), assignment[horizons[-1]])
        out.append(owner)
    return out


def stitch(paths: dict[str, np.ndarray], owners: list[str], axis: int = -1) -> np.ndarray:
    """Take step j from the path (or residual matrix) of owners[j]."""
    cols = [np.take(paths[name], [j], axis=axis) for j, name in enumerate(owners)]
    return np.concatenate(cols, axis=axis)


def select_champion(results: dict[str, BacktestResult], spec: SeriesSpec) -> Selection:
    missing = [b for b in BASELINES if b not in results]
    if missing:
        raise ValueError(f"baselines must be backtested: {missing}")
    metrics: dict[str, dict[str, dict]] = {}
    for name, r in results.items():
        metrics[name] = {str(hm): horizon_metrics(r.residuals, spec.steps(hm) - 1) for hm in spec.horizons_min}
    assignment, baseline_assignment = {}, {}
    for hm in spec.horizons_min:
        key = str(hm)
        base_maes = {b: metrics[b][key]["mae"] for b in BASELINES if metrics[b][key]["mae"] is not None}
        if not base_maes:
            raise ValueError(f"no baseline produced a residual at {hm} min")
        best_base = min(base_maes, key=lambda b: (base_maes[b], b))
        baseline_assignment[hm] = best_base
        j = spec.steps(hm) - 1
        ref_col = results[best_base].residuals[:, j]
        ref_n = int(np.isfinite(ref_col).sum())
        winners = {}
        for name, per_h in metrics.items():
            m = per_h[key]
            # Paired: candidate and reference on the windows where BOTH have a residual. A model
            # that fails on hard windows must not have them silently dropped from its MAE only.
            col = results[name].residuals[:, j]
            both = np.isfinite(col) & np.isfinite(ref_col)
            ref_pair = float(np.mean(np.abs(ref_col[both]))) if both.any() else 0.0
            m["rel_mae"] = (float(np.mean(np.abs(col[both]))) / ref_pair) if ref_pair > 0 else None
            m["n_paired"] = int(both.sum())
            if name in BASELINES:
                m["passed"] = name == best_base
                continue
            m["p_win"], _ = paired_win_p(col, ref_col)
            m["passed"] = bool(m["rel_mae"] is not None and m["rel_mae"] < 1.0
                               and m["p_win"] is not None and m["p_win"] < GATE_ALPHA
                               and m["n_paired"] >= max(MIN_PATHS + 1, math.ceil(MIN_WINDOW_SHARE * ref_n)))
            if m["passed"]:
                winners[name] = m["rel_mae"]
        assignment[hm] = min(winners, key=lambda n: (winners[n], n)) if winners else best_base
    served_baseline = all(v in BASELINES for v in assignment.values())
    return Selection(assignment, baseline_assignment, served_baseline, metrics)
