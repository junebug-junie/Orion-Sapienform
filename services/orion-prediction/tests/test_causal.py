from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from app.causal import (
    CausalError,
    analyze_switchback,
    design_switchback,
    intervention_effect,
    scenario,
    time_of_day_profile,
)
from app.models import build_models, split_seasonalities
from tests.helpers import T0, daily_series, spec


def _switchback(effect: float, seed: int, n_blocks: int = 16, block_min: int = 30, burn_in: int = 10):
    design = design_switchback(T0 + timedelta(days=14), block_min, n_blocks, seed=seed)
    s = daily_series(16, amp=1.0, noise=0.1, seed=seed)
    for b in design["blocks"]:
        if b["arm"] == "on":
            # Heat ramps in over 10 min: the burn-in must drop the ramp, not bias the estimate.
            m = (s.index >= pd.Timestamp(b["start"])) & (s.index < pd.Timestamp(b["end"]))
            ramp = np.minimum(1.0, (np.arange(m.sum()) + 1) * 5 / 10)
            s[m] += effect * ramp
    history = s[s.index < pd.Timestamp(design["blocks"][0]["start"])]
    return s, design, time_of_day_profile(history, 5), burn_in


def test_design_is_balanced_contiguous_and_seeded():
    d = design_switchback(T0, 20, 10, seed=42)
    arms = [b["arm"] for b in d["blocks"]]
    assert arms.count("on") == arms.count("off") == 5
    assert d["blocks"][1]["start"] == d["blocks"][0]["end"]
    assert design_switchback(T0, 20, 10, seed=42)["blocks"] == d["blocks"]
    with pytest.raises(CausalError):
        design_switchback(T0, 20, 7)


def test_switchback_recovers_a_planted_effect():
    s, d, prof, burn = _switchback(0.8, seed=1)
    est = analyze_switchback(s, d["blocks"], burn, 5, prof, n_perm=2000)
    assert est.claim_kind == "experimental_effect"
    assert est.ci_lo < 0.8 < est.ci_hi and abs(est.effect - 0.8) < 0.15
    assert est.p_value < 0.01


def test_switchback_null_effect_is_not_significant_most_of_the_time():
    false_pos = 0
    for seed in range(20):
        s, d, prof, burn = _switchback(0.0, seed=seed)
        est = analyze_switchback(s, d["blocks"], burn, 5, prof, n_perm=500)
        false_pos += est.p_value < 0.05
    assert false_pos <= 3


def test_switchback_refuses_with_too_few_measured_blocks():
    s, d, prof, burn = _switchback(0.5, seed=3)
    s[s.index >= pd.Timestamp(d["blocks"][4]["start"])] = np.nan
    est = analyze_switchback(s, d["blocks"], burn, 5, prof)
    assert est.effect is None and "need" in est.details["reason"]


def _its_setup(effect: float, seed: int):
    sp = spec(seasonalities=[{"name": "daily", "period": "1d", "fourier_order": 3}])
    s = daily_series(14, amp=1.0, noise=0.05, seed=seed)
    start = T0 + timedelta(days=12, hours=3)
    end = start + timedelta(hours=2)
    m = (s.index >= pd.Timestamp(start)) & (s.index < pd.Timestamp(end + timedelta(minutes=30)))
    s[m] += effect
    usable, _ = split_seasonalities(sp, 3 * 288)
    return sp, s, start, end, build_models(sp, usable)["mstl_ets"]


def test_interrupted_time_series_recovers_a_step_effect_with_placebo_interval():
    sp, s, start, end, model = _its_setup(1.0, seed=5)
    est = intervention_effect(s, model, start, end, 5, pre_days=3, lag_min=30, n_placebo=12)
    assert est.claim_kind == "observational_effect"
    assert abs(est.effect - 1.0) < 0.1
    # Interval calibration is a many-seed property: evals/run_causal_calibration_eval.py checks it.
    assert est.ci_lo is not None and 0 < est.ci_lo <= est.effect <= est.ci_hi and est.ci_hi - est.ci_lo < 0.3
    assert est.p_value <= 1 / 13 + 1e-9
    assert len(est.details["placebo_effects"]) == est.n >= 8


def test_its_without_enough_history_for_placebos_reports_unknown_interval():
    sp, s, start, end, model = _its_setup(1.0, seed=6)
    short = s[s.index >= pd.Timestamp(start - timedelta(days=3, hours=1))]
    est = intervention_effect(short, model, start, end, 5, pre_days=3, lag_min=30)
    assert est.effect is not None and est.ci_lo is None and "placebo" in est.details["reason"]


def test_scenario_recovers_a_known_covariate_coefficient_and_is_labelled_conditional():
    rng = np.random.default_rng(9)
    idx = pd.date_range(T0, periods=7 * 288, freq="5min", tz="UTC")
    x = pd.Series(400 + 100 * (rng.random(len(idx)) > 0.5).cumsum() % 300, index=idx, dtype=float)
    x = x.rolling(6, min_periods=1).mean()
    minutes = (idx.hour * 60 + idx.minute).to_numpy()
    noise = np.zeros(len(idx))
    for i in range(1, len(idx)):
        noise[i] = 0.7 * noise[i - 1] + rng.normal(0, 0.05)
    y = pd.Series(20 + 0.01 * x.to_numpy() + 0.5 * np.sin(2 * np.pi * minutes / 1440) + noise, index=idx)
    res = scenario(y, x, "watts", [x.iloc[-12:].mean() + 100] * 24, 5)
    assert res.claim_kind == "conditional"
    assert abs(res.long_run_effect_per_unit - 0.01) < 0.003
    assert abs(res.delta[-1] - 1.0) < 0.35
    assert "Not a causal effect" in res.note


def test_scenario_refuses_with_too_little_overlap():
    idx = pd.date_range(T0, periods=100, freq="5min", tz="UTC")
    with pytest.raises(CausalError):
        scenario(pd.Series(1.0, index=idx), pd.Series(2.0, index=idx), "w", [1.0], 5)


def test_scenario_refuses_a_covariate_with_no_recent_reading_or_a_nan_path():
    idx = pd.date_range(T0, periods=3 * 288, freq="5min", tz="UTC")
    y = pd.Series(28.0 + np.sin(np.arange(len(idx)) / 50), index=idx)
    x = pd.Series(500.0, index=idx)
    x.iloc[-12:] = np.nan
    with pytest.raises(CausalError, match="last hour"):
        scenario(y, x, "w", [600.0], 5)
    with pytest.raises(CausalError, match="finite"):
        scenario(y, pd.Series(500.0, index=idx), "w", [float("nan")], 5)
