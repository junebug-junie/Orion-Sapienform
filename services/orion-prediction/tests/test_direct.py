from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest import run_backtest
from app.direct import direct_ar, direct_arx
from app.models import Forecaster, build_models
from tests.helpers import T0, spec


def heat_balance(days: float = 8, seed: int = 0):
    """dT/dt = a * (P - P_ref) - b * (T - T_amb): a power covariate genuinely drives the target."""
    rng = np.random.default_rng(seed)
    n = int(days * 288)
    idx = pd.date_range(T0, periods=n, freq="5min", tz="UTC")
    p = np.where(np.sin(np.arange(n) / 37.0) + rng.normal(0, 0.3, n) > 0.4, 700.0, 350.0)
    p = pd.Series(p).rolling(3, min_periods=1).mean().to_numpy()
    t = np.empty(n)
    t[0] = 28.0
    for i in range(1, n):
        t[i] = t[i - 1] + 0.004 * (p[i - 1] - 500) - 0.05 * (t[i - 1] - 28.0) + rng.normal(0, 0.03)
    return idx, t, p[:, None]


def test_direct_arx_uses_the_covariate_and_beats_persistence_and_its_history_only_twin():
    idx, t, X = heat_balance()
    h, model, ar, errs, errs_ar, naive = 12, direct_arx(5), direct_ar(5), [], [], []
    for cut in range(len(t) - 2 * 288, len(t) - h, 58):   # 10 origins over the last 2 days
        pred = model(t[:cut], h, idx[:cut], X[:cut])
        assert pred.shape == (h,) and np.all(np.isfinite(pred))
        truth = t[cut:cut + h]
        errs.append(np.mean(np.abs(pred - truth)))
        errs_ar.append(np.mean(np.abs(ar(t[:cut], h, idx[:cut]) - truth)))
        naive.append(np.mean(np.abs(t[cut - 1] - truth)))
    assert np.mean(errs) < 0.7 * np.mean(naive)
    # Power genuinely drives this target, so the covariate twin must beat the history-only one.
    assert np.mean(errs) < 0.9 * np.mean(errs_ar)


def test_direct_model_refuses_mostly_missing_covariates_and_missing_input():
    idx, t, X = heat_balance(days=3)
    X = X.copy()
    X[: len(X) // 2] = np.nan
    with pytest.raises(ValueError, match="gaps"):
        direct_arx(5)(t, 12, idx, X)
    f = Forecaster("direct_arx", direct_arx(5), uses_covariates=True)
    with pytest.raises(ValueError, match="needs covariates"):
        f.forecast(t, 12, idx)


def test_backtest_never_shows_a_model_covariates_after_its_origin():
    sp = spec(horizons=["5min", "1h"], train_window="5d", covariates=[], seasonalities=[])
    idx, t, X = heat_balance(days=8)
    series = pd.Series(t, index=idx)
    cov = pd.DataFrame({"p": X[:, 0]}, index=idx)
    seen_lengths = []

    def spy(y, h, index, Xw):
        assert len(Xw) == len(y) and Xw.shape[1] == 1
        seen_lengths.append((index[-1], len(Xw)))
        return np.full(h, y[-1])
    r = run_backtest(series, sp, "spy", lambda u: Forecaster("spy", spy, uses_covariates=True), 6, 36,
                     covariates=cov)
    assert r.n_windows == 6
    # Each window's covariates end exactly at its training end (the origin bucket), never later.
    assert all(last < origin for (last, _), origin in zip(seen_lengths, r.origins))


def test_zoo_adds_direct_models_only_when_the_series_declares_covariates():
    assert "direct_arx" not in build_models(spec(), []) and "direct_ar" in build_models(spec(), [])
    reg_spec = __import__("app.series", fromlist=["load_registry"]).load_registry(
        __import__("tests.conftest", fromlist=["REGISTRY_PATH"]).REGISTRY_PATH).get("athena.cabinet_temp_c")
    zoo = build_models(reg_spec, [])
    assert zoo["direct_arx"].uses_covariates and not zoo["direct_ar"].uses_covariates
