"""Model zoo behind one call: ``forecast(y, h, index) -> point path of length h``.

Every model returns a point path only. Intervals come from split-conformal residual paths
(app/backtest.py), the same way for every model, so their widths are comparable and calibrated
against what each model actually got wrong.

A seasonality is *usable* only when the training window holds ``MIN_CYCLES`` full cycles of it.
The rest are refused and reported, never silently fitted on one cycle.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from app.direct import direct_ar, direct_arx
from app.series import Seasonality, SeriesSpec

MIN_CYCLES = 2
BASELINES = ("naive", "seasonal_naive")
# STL at a weekly period (2016 five-minute buckets) costs ~26 s per fit vs ~2.3 s at daily (measured
# 2026-10-15 on 20 days of cabinet data), so MSTL fits cycles up to a day and Prophet's Fourier
# terms carry the longer ones.
MSTL_MAX_PERIOD_MIN = 1440

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)


def split_seasonalities(spec: SeriesSpec, n_obs: int) -> tuple[list[Seasonality], list[str]]:
    span_min = n_obs * spec.grain_min
    usable, refused = [], []
    for s in spec.seasonalities:
        (usable if span_min >= MIN_CYCLES * s.period_min else refused).append(s)
    return usable, [s.name for s in refused]


@dataclass(frozen=True)
class Forecaster:
    name: str
    fn: Callable[..., np.ndarray]
    transform: str = "none"
    # True: fn(y, h, index, X) with X the (n, k) covariate history aligned to y (spec.covariates order).
    uses_covariates: bool = False

    def forecast(self, y: np.ndarray, h: int, index: pd.DatetimeIndex,
                 covariates: np.ndarray | None = None) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        args = (index,)
        if self.uses_covariates:
            if covariates is None:
                raise ValueError(f"{self.name} needs covariates and got none")
            args = (index, covariates)
        if self.transform == "log":
            if np.any(y <= 0):
                raise ValueError("log transform needs strictly positive values")
            out = np.exp(self.fn(np.log(y), h, *args))
        else:
            out = self.fn(y, h, *args)
        out = np.asarray(out, dtype=float).reshape(-1)
        if out.shape[0] != h or not np.all(np.isfinite(out)):
            raise ValueError(f"{self.name} returned a non-finite or wrong-length path")
        return out


def _naive(y, h, _idx):
    return np.full(h, y[-1])


def _seasonal_naive(season: int):
    def fn(y, h, _idx):
        if season <= 1 or len(y) < season:
            return np.full(h, y[-1])
        return np.resize(y[-season:], h)
    return fn


def _window_average(window: int):
    def fn(y, h, _idx):
        return np.full(h, float(np.mean(y[-window:])))
    return fn


def _mstl(periods: list[int], trend: str):
    def fn(y, h, _idx):
        from statsforecast.models import MSTL, AutoARIMA, AutoETS

        trend_model = AutoARIMA(seasonal=False) if trend == "arima" else AutoETS(model="ZZN")
        if not periods:
            return trend_model.forecast(y=y, h=h)["mean"]
        return MSTL(season_length=periods, trend_forecaster=trend_model).forecast(y=y, h=h)["mean"]
    return fn


def _prophet(seasonalities: list[Seasonality], grain_min: int):
    def fn(y, h, idx):
        from prophet import Prophet

        m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False,
                    uncertainty_samples=0, changepoint_prior_scale=0.05)
        for s in seasonalities:
            m.add_seasonality(name=s.name, period=s.period_min / 1440.0, fourier_order=s.fourier_order)
        ds = idx.tz_convert("UTC").tz_localize(None)
        m.fit(pd.DataFrame({"ds": ds, "y": y}))
        future = pd.DataFrame({"ds": pd.date_range(ds[-1] + pd.Timedelta(minutes=grain_min), periods=h,
                                                    freq=f"{grain_min}min")})
        return m.predict(future)["yhat"].to_numpy()
    return fn


def build_models(spec: SeriesSpec, usable: list[Seasonality]) -> dict[str, Forecaster]:
    """Baselines first; both are always present (the better one is served where nothing passes)."""
    day = 1440 // spec.grain_min
    periods = sorted({s.period_min // spec.grain_min for s in usable if s.period_min <= MSTL_MAX_PERIOD_MIN})
    t = spec.transform
    zoo = [
        Forecaster("naive", _naive, t),
        Forecaster("seasonal_naive", _seasonal_naive(day), t),
        Forecaster("window_average_1h", _window_average(max(1, 60 // spec.grain_min)), t),
        Forecaster("mstl_ets", _mstl(periods, "ets"), t),
        Forecaster("mstl_arima", _mstl(periods, "arima"), t),
        Forecaster("prophet", _prophet(usable, spec.grain_min), t),
        Forecaster("direct_ar", direct_ar(spec.grain_min), t),
    ]
    if spec.covariates:
        zoo.append(Forecaster("direct_arx", direct_arx(spec.grain_min), t, uses_covariates=True))
    return {f.name: f for f in zoo}
