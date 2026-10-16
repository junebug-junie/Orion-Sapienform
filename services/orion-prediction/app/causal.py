"""Cause and effect, in three levels. Every result carries ``claim_kind`` so no caller can mistake
one level for another.

1. ``conditional`` -- ``scenario()``. Regression with ARMA errors (SARIMAX): target ~ covariate
   rolling means + daily Fourier terms. Supply a covariate path; get the forecast under it and
   under "covariate held at its last-hour mean". This is correlation. It does not separate the
   covariate from anything that moves with it (time of day, the AC, the 34 C reflex).

2. ``observational_effect`` -- ``intervention_effect()``. Interrupted time series: fit a forecaster
   on the days before a logged action, forecast the action window as the counterfactual, and take
   mean(actual - counterfactual). The interval and p-value come from *placebo* runs of the same
   procedure at earlier start times with no action, so they reflect how big this method's error is
   on this series, not a model assumption. Still confounded by anything else that changed at the
   same time.

3. ``experimental_effect`` -- ``design_switchback()`` + ``analyze_switchback()``. Randomised ON/OFF
   blocks. The analysis removes the time-of-day profile (variance reduction only; randomisation
   is what makes it unbiased), drops a burn-in at each block start (heat carry-over), and reports
   the ON-OFF difference with a Welch interval and a permutation p-value. This is the only level
   that can answer "doing X raises the temperature by Y".
"""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Sequence

import numpy as np
import pandas as pd

from app.frame import floor_ts, train_ready
from app.models import Forecaster

ROLL_WINDOWS_MIN = (0, 30, 120)
MIN_PLACEBOS = 8
MIN_BLOCKS_PER_ARM = 4


class CausalError(ValueError):
    pass


# --------------------------------------------------------------------------------------------
# 1. conditional scenario
# --------------------------------------------------------------------------------------------

def _rolling(values: pd.Series, grain_min: int) -> pd.DataFrame:
    cols = {}
    for w in ROLL_WINDOWS_MIN:
        k = max(1, w // grain_min)
        cols[f"ma{w}"] = values if w == 0 else values.rolling(k, min_periods=1).mean()
    return pd.DataFrame(cols)


def _fourier(index: pd.DatetimeIndex, period_min: float, order: int) -> pd.DataFrame:
    t = (index.asi8 // 60_000_000_000).astype(float)  # minutes since epoch
    cols = {}
    for k in range(1, order + 1):
        cols[f"sin{k}"] = np.sin(2 * np.pi * k * t / period_min)
        cols[f"cos{k}"] = np.cos(2 * np.pi * k * t / period_min)
    return pd.DataFrame(cols, index=index)


@dataclass
class ScenarioResult:
    claim_kind: str
    covariate: str
    target_ts: list[datetime]
    under_scenario: list[float]
    under_hold: list[float]
    delta: list[float]
    lo80: list[float]
    hi80: list[float]
    long_run_effect_per_unit: float
    hold_value: float
    n_train: int
    note: str


def scenario(target: pd.Series, covariate: pd.Series, covariate_name: str, future_path: Sequence[float],
             grain_min: int, daily_order: int = 4) -> ScenarioResult:
    """``target``/``covariate``: aligned regular series ending at the forecast origin."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    h = len(future_path)
    if h < 1:
        raise CausalError("future_path is empty")
    if not np.all(np.isfinite(np.asarray(future_path, dtype=float))):
        raise CausalError("future_path must be finite numbers")
    # Keep the grid regular: dropping rows would make the rolling means span gaps.
    df = pd.DataFrame({"y": target, "x": covariate})
    if len(df) < 2 * 1440 // grain_min:
        raise CausalError(f"need >= 2 days of target and {covariate_name}; have {len(df)} buckets")
    x_gap = float(df["x"].isna().mean())
    if x_gap > 0.10:
        raise CausalError(f"{covariate_name} is {x_gap:.0%} gaps over the window")
    recent = df["x"].iloc[-max(1, 60 // grain_min):]
    if recent.isna().all():
        raise CausalError(f"{covariate_name} has no reading in the last hour; nothing to hold it at")
    hold = float(recent.mean())
    df["x"] = df["x"].interpolate(method="linear", limit_direction="both")
    y, filled = train_ready(df["y"])
    if filled > 0.10:
        raise CausalError(f"target is {filled:.0%} gaps over the window")
    freq = pd.Timedelta(minutes=grain_min)
    fut_idx = pd.date_range(df.index[-1] + freq, periods=h, freq=freq)

    def design(x_future: Sequence[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
        x_all = pd.concat([df["x"], pd.Series(list(x_future), index=fut_idx, dtype=float)])
        ex = pd.concat([_rolling(x_all, grain_min), _fourier(x_all.index, 1440.0, daily_order)], axis=1)
        return ex.iloc[:len(df)], ex.iloc[len(df):]

    ex_train, ex_future = design(future_path)
    _, ex_hold = design([hold] * h)
    fit = SARIMAX(y, exog=ex_train.to_numpy(), order=(2, 0, 1), trend="c",
                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    fc = fit.get_forecast(h, exog=ex_future.to_numpy())
    base = fit.get_forecast(h, exog=ex_hold.to_numpy()).predicted_mean
    ci = np.asarray(fc.conf_int(alpha=0.2))
    mean = np.asarray(fc.predicted_mean)
    beta = np.asarray(fit.params[1:1 + len(ROLL_WINDOWS_MIN)])  # after the constant
    return ScenarioResult(
        claim_kind="conditional", covariate=covariate_name,
        target_ts=[t.to_pydatetime() for t in fut_idx], under_scenario=mean.tolist(),
        under_hold=np.asarray(base).tolist(), delta=(mean - np.asarray(base)).tolist(),
        lo80=ci[:, 0].tolist(), hi80=ci[:, 1].tolist(), long_run_effect_per_unit=float(beta.sum()),
        hold_value=hold, n_train=len(df),
        note=("conditional forecast from correlation: a sustained +1 unit of the covariate goes with "
              "long_run_effect_per_unit of the target. Not a causal effect."),
    )


# --------------------------------------------------------------------------------------------
# 2. interrupted time series with placebo calibration
# --------------------------------------------------------------------------------------------

@dataclass
class EffectEstimate:
    claim_kind: str
    effect: float | None
    ci_lo: float | None
    ci_hi: float | None
    p_value: float | None
    n: int
    details: dict


def _window_effect(series: pd.Series, model: Forecaster, start: pd.Timestamp, end: pd.Timestamp,
                   pre_steps: int) -> float | None:
    idx = series.index
    i0 = idx.searchsorted(start)
    i1 = idx.searchsorted(end)
    if i0 - pre_steps < 0 or i1 <= i0 or i1 > len(series):
        return None
    pre = series.iloc[i0 - pre_steps:i0]
    y, filled = train_ready(pre)
    if filled > 0.10:
        return None
    actual = series.iloc[i0:i1].to_numpy(dtype=float)
    if np.isfinite(actual).mean() < 0.8:
        return None
    try:
        cf = model.forecast(y, i1 - i0, pre.index)
    except Exception:  # noqa: BLE001 -- a placebo that cannot fit is skipped, the count shows it
        return None
    return float(np.nanmean(actual - cf))


def _overlaps(a0, a1, spans) -> bool:
    return any(a0 < b1 and b0 < a1 for b0, b1 in spans)


def intervention_effect(series: pd.Series, model: Forecaster, start: datetime, end: datetime,
                        grain_min: int, pre_days: float = 3.0, lag_min: int = 30, n_placebo: int = 20,
                        busy: Sequence[tuple[datetime, datetime]] = (), seed: int = 0) -> EffectEstimate:
    """Effect of an action over [start, end + lag) against a pre-period counterfactual.

    ``busy``: other logged interventions (already extended by their lag); neither a placebo window
    nor its pre-period may overlap them.
    """
    s0 = pd.Timestamp(floor_ts(start, grain_min))
    s1 = pd.Timestamp(floor_ts(end + timedelta(minutes=lag_min), grain_min))
    if s1 <= s0:
        raise CausalError("intervention window is empty")
    pre_steps = int(pre_days * 1440 // grain_min)
    effect = _window_effect(series, model, s0, s1, pre_steps)
    if effect is None:
        return EffectEstimate("observational_effect", None, None, None, None, 0,
                              {"reason": "not enough clean data before or during the action"})
    dur = s1 - s0
    rng = np.random.default_rng(seed)
    first_ok = series.index[0] + pd.Timedelta(minutes=pre_days * 1440)
    latest = s0 - dur
    spans = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in busy] + [(s0, s1)]
    placebos: list[float] = []
    if latest > first_ok:
        span_steps = int((latest - first_ok) / pd.Timedelta(minutes=grain_min))
        pre = pd.Timedelta(minutes=pre_steps * grain_min)
        # Distinct start offsets, without replacement: a repeated placebo is not new evidence.
        for k in rng.permutation(span_steps + 1):
            if len(placebos) >= n_placebo:
                break
            p0 = first_ok + pd.Timedelta(minutes=grain_min * int(k))
            p1 = p0 + dur
            # Neither the placebo window nor its pre-period may contain a logged action.
            if _overlaps(p0 - pre, p1, spans):
                continue
            pe = _window_effect(series, model, p0, p1, pre_steps)
            if pe is not None:
                placebos.append(pe)
    details = {"window_start": s0.isoformat(), "window_end": s1.isoformat(), "pre_days": pre_days,
               "model": model.name, "placebo_effects": placebos}
    if len(placebos) < MIN_PLACEBOS:
        details["reason"] = f"only {len(placebos)} placebo windows (need {MIN_PLACEBOS}); interval unknown"
        return EffectEstimate("observational_effect", effect, None, None, None, len(placebos), details)
    pl = np.asarray(placebos)
    # The placebo distribution is this method's error with no action: effect - error = truth.
    lo, hi = effect - np.quantile(pl, 0.975), effect - np.quantile(pl, 0.025)
    p = (1 + int(np.sum(np.abs(pl) >= abs(effect)))) / (1 + len(pl))
    return EffectEstimate("observational_effect", effect, float(lo), float(hi), float(p), len(pl), details)


# --------------------------------------------------------------------------------------------
# 3. switchback experiments
# --------------------------------------------------------------------------------------------

def design_switchback(start_at: datetime, block_min: int, n_blocks: int, seed: int | None = None) -> dict:
    if n_blocks < 2 * MIN_BLOCKS_PER_ARM or n_blocks % 2:
        raise CausalError(f"n_blocks must be even and >= {2 * MIN_BLOCKS_PER_ARM}")
    if block_min < 5:
        raise CausalError("block_min must be >= 5")
    seed = int(seed if seed is not None else uuid.uuid4().int % (2 ** 31))
    arms = np.array(["on"] * (n_blocks // 2) + ["off"] * (n_blocks // 2))
    np.random.default_rng(seed).shuffle(arms)
    start = start_at.astimezone(timezone.utc)
    blocks = [{"index": i, "arm": str(a), "start": (start + timedelta(minutes=i * block_min)).isoformat(),
               "end": (start + timedelta(minutes=(i + 1) * block_min)).isoformat()}
              for i, a in enumerate(arms)]
    return {"seed": seed, "block_min": block_min, "n_blocks": n_blocks, "blocks": blocks}


def time_of_day_profile(history: pd.Series, grain_min: int) -> pd.Series:
    """Mean value per time-of-day bucket (minute of day) from history before the experiment."""
    tod = (history.index.hour * 60 + history.index.minute) // grain_min * grain_min
    return history.groupby(tod).mean()


def analyze_switchback(series: pd.Series, blocks: Sequence[dict], burn_in_min: int, grain_min: int,
                       profile: pd.Series | None = None, n_perm: int = 10000, seed: int = 0) -> EffectEstimate:
    from scipy import stats

    adj = series
    if profile is not None and len(profile):
        tod = (series.index.hour * 60 + series.index.minute) // grain_min * grain_min
        adj = series - profile.reindex(tod).to_numpy()
    means = {"on": [], "off": []}
    per_block = []
    for b in blocks:
        b0 = pd.Timestamp(b["start"]) + pd.Timedelta(minutes=burn_in_min)
        b1 = pd.Timestamp(b["end"])
        vals = adj[(adj.index >= b0) & (adj.index < b1)].dropna()
        if len(vals) == 0:
            per_block.append({**b, "mean": None})
            continue
        m = float(vals.mean())
        means[b["arm"]].append(m)
        per_block.append({**b, "mean": m})
    on, off = np.asarray(means["on"]), np.asarray(means["off"])
    details = {"burn_in_min": burn_in_min, "blocks": per_block, "profile_adjusted": profile is not None}
    if len(on) < MIN_BLOCKS_PER_ARM or len(off) < MIN_BLOCKS_PER_ARM:
        details["reason"] = f"need >= {MIN_BLOCKS_PER_ARM} measured blocks per arm; have on={len(on)} off={len(off)}"
        return EffectEstimate("experimental_effect", None, None, None, None, len(on) + len(off), details)
    diff = float(on.mean() - off.mean())
    va, vb = on.var(ddof=1) / len(on), off.var(ddof=1) / len(off)
    se = math.sqrt(va + vb)
    dof = (va + vb) ** 2 / ((va ** 2) / (len(on) - 1) + (vb ** 2) / (len(off) - 1)) if se > 0 else len(on) + len(off) - 2
    t = float(stats.t.ppf(0.975, dof))
    pooled = np.concatenate([on, off])
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        hits += abs(perm[:len(on)].mean() - perm[len(on):].mean()) >= abs(diff) - 1e-12
    details.update({"se": se, "dof": dof, "n_on": len(on), "n_off": len(off)})
    return EffectEstimate("experimental_effect", diff, diff - t * se, diff + t * se,
                          (hits + 1) / (n_perm + 1), len(on) + len(off), details)
