"""Family A slice 1: hygiene gate, rungs, significance, joint surprise (app/family_a.py)."""
from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
import yaml

from app import family_a as fa
from app.family_a_load import build_queries
from tests.conftest import REPO

CONFIG_PATH = REPO / "config" / "prediction" / "family_a.v1.yaml"
T0 = datetime(2026, 10, 1, tzinfo=timezone.utc)


def _cfg() -> fa.FamilyAConfig:
    return fa.parse_config(yaml.safe_load(CONFIG_PATH.read_text()))


def _hyg() -> fa.Hygiene:
    return _cfg().hygiene


def _series(values, start=T0, grain=5) -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq=f"{grain}min", tz="UTC")
    return pd.Series(np.asarray(values, dtype=float), index=idx)


def _daily(days=14, seed=0, noise=0.3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(days * 288)
    return 25 + 2 * np.sin(2 * np.pi * t / 288) + noise * rng.standard_normal(len(t))


# --- config ----------------------------------------------------------------------------------------

def test_checked_in_config_loads_and_reads_only_raw_tables():
    cfg = _cfg()
    assert cfg.horizons_min == (5, 15)
    assert all(s.source.table in fa.RAW_TABLES for s in cfg.inputs)
    assert not any(fa.HINT_PATTERN.search(s.id) for s in cfg.inputs)


def _doc(**inp):
    base = yaml.safe_load(CONFIG_PATH.read_text())
    base["inputs"] = [{"id": "x.reading_c", "unit": "C", "source": {"table": "biometrics", "node": "athena",
                                                                    "key": "cabinet_temp_c"}, **inp}]
    return base


@pytest.mark.parametrize("key", ["thermal_pressure", "prediction_error_bus_synaptic", "action_warrant"])
def test_hint_channel_key_is_refused(key):
    with pytest.raises(fa.FamilyAConfigError, match="hint channel"):
        fa.parse_config(_doc(source={"table": "biometrics", "node": "athena", "key": key}))


def test_hint_channel_id_is_refused():
    with pytest.raises(fa.FamilyAConfigError, match="hint channel"):
        fa.parse_config(_doc(id="field.thermal_pressure"))


def test_non_raw_table_is_refused():
    # The mood-arc corpus (pressure / hint channels) is not a raw-reading table.
    with pytest.raises(fa.FamilyAConfigError, match="raw-reading"):
        fa.parse_config(_doc(source={"table": "field_channels", "node": "athena", "key": "x"}))


def test_missing_threshold_is_refused_not_defaulted():
    doc = _doc()
    del doc["hygiene"]["max_mode_share"]
    with pytest.raises(fa.FamilyAConfigError, match="max_mode_share"):
        fa.parse_config(doc)


def test_barometric_pressure_reading_is_not_a_hint_channel():
    cfg = _cfg()
    assert any(s.source.key == "cabinet_pressure_hpa" for s in cfg.inputs)


# --- hygiene ---------------------------------------------------------------------------------------

def test_healthy_daily_series_is_a_target():
    v = fa.screen_series(_series(_daily()), _hyg(), "t")
    assert v.status == fa.TARGET, v


def test_flat_series_is_refused():
    v = fa.screen_series(_series(np.full(14 * 288, 21.5)), _hyg())
    assert v.status == fa.REFUSED and {"few_distinct", "one_value"} <= set(v.reasons)


def test_mostly_zero_series_over_85_percent_is_refused():
    rng = np.random.default_rng(1)
    x = np.where(rng.uniform(size=14 * 288) < 0.9, 0.0, rng.uniform(1, 100, size=14 * 288))
    v = fa.screen_series(_series(x), _hyg())
    assert v.status == fa.REFUSED and "one_value" in v.reasons


def test_sparse_event_series_is_covariate_not_target():
    rng = np.random.default_rng(2)
    x = np.where(rng.uniform(size=14 * 288) < 0.6, 0.0, rng.uniform(1, 100, size=14 * 288))
    v = fa.screen_series(_series(x), _hyg())
    assert v.status == fa.COVARIATE and v.reasons == ("event_like",)


def test_monotone_drift_is_refused_even_though_it_varies():
    rng = np.random.default_rng(3)
    x = np.linspace(0, 100, 14 * 288) + rng.standard_normal(14 * 288)
    v = fa.screen_series(_series(x), _hyg())
    assert v.status == fa.REFUSED and "drift" in v.reasons


def test_decayed_to_zero_artifact_is_refused():
    # The 0.92-per-tick staleness decay (CLAUDE.md metric gate): keeps "moving", never comes back.
    x = 0.8 * 0.92 ** (np.arange(14 * 288) / 12.0)
    v = fa.screen_series(_series(x), _hyg())
    assert v.status == fa.REFUSED and ("drift" in v.reasons or "no_rest" in v.reasons)


def test_absent_series_is_refused():
    x = _daily()
    x[: int(0.6 * len(x))] = np.nan
    assert fa.screen_series(_series(x), _hyg()).reasons == ("absent",)


def test_window_edge_minutes_into_a_day_is_not_a_day():
    # Regression: a 24 h window starting 00:20 left a 20-min "day" that never touched the band -> no_rest.
    s = _series(_daily(days=2)[: 288 + 4], start=T0 + timedelta(minutes=20))
    days, share, _rho = fa._rest_shape(s, *np.quantile(s, [0.1, 0.9]))
    assert len(days) == 1 and share == 1.0


def test_same_reading_twice_is_refused_as_a_duplicate():
    # The CCS811 computes eCO2 from TVOC: two columns, one reading.
    tvoc = _daily(seed=11) * 10
    f = pd.DataFrame({"office.eco2": 400 + 2.1 * tvoc, "office.tvoc": tvoc, "office.temp": _daily(seed=12)},
                     index=pd.date_range(T0, periods=len(tvoc), freq="5min", tz="UTC"))
    v = fa.screen_frame(f, _hyg())
    assert v["office.eco2"].status == fa.TARGET and v["office.temp"].status == fa.TARGET
    assert v["office.tvoc"].reasons == ("duplicate_of:office.eco2",)


# --- run-time gate ---------------------------------------------------------------------------------

def _frame(cols: dict[str, np.ndarray], start=T0) -> pd.DataFrame:
    n = len(next(iter(cols.values())))
    idx = pd.date_range(start, periods=n, freq="5min", tz="UTC")
    return pd.DataFrame(cols, index=idx)


def test_gate_refuses_renamed_input():
    f = _frame({"a.x": _daily()})
    with pytest.raises(fa.HygieneRefused) as exc:
        fa.gate_named_inputs({"a.x": fa.TARGET, "a.renamed": fa.TARGET}, f, _hyg())
    assert exc.value.refusals == {"a.renamed": ("not_loaded",)}


def test_gate_refuses_target_that_turned_event_like():
    rng = np.random.default_rng(4)
    x = np.where(rng.uniform(size=14 * 288) < 0.6, 0.0, rng.uniform(1, 100, size=14 * 288))
    with pytest.raises(fa.HygieneRefused, match="role_changed:covariate"):
        fa.gate_named_inputs({"a.x": fa.TARGET}, _frame({"a.x": x}), _hyg())


def test_gate_refuses_stale_input():
    f = _frame({"a.x": _daily()})
    end = f.index[-1].to_pydatetime()
    with pytest.raises(fa.HygieneRefused, match="stale"):
        fa.gate_named_inputs({"a.x": fa.TARGET}, f, _hyg(), {"a.x": end - timedelta(hours=2)}, end)


def test_short_scoring_window_passes_a_coarse_healthy_series():
    # A 1 C-resolution GPU temperature: < 50 distinct 5-min means in one day is honest, not degenerate.
    rng = np.random.default_rng(5)
    x = np.round(60 + 5 * np.sin(np.arange(288) / 30) + rng.standard_normal(288))
    f = _frame({"gpu.t": x}, start=T0 + timedelta(minutes=20))
    fa.gate_named_inputs({"gpu.t": fa.TARGET}, f, _hyg(), bands={"gpu.t": [55.0, 65.0]})


def test_short_scoring_window_refuses_stuck_and_out_of_regime_inputs():
    f = _frame({"stuck.x": np.full(288, 3.0), "moved.x": 100 + np.random.default_rng(6).standard_normal(288)})
    with pytest.raises(fa.HygieneRefused) as exc:
        fa.gate_named_inputs({"stuck.x": fa.TARGET, "moved.x": fa.TARGET}, f, _hyg(),
                             bands={"stuck.x": [0.0, 10.0], "moved.x": [20.0, 30.0]})
    assert exc.value.refusals == {"stuck.x": ("one_value",), "moved.x": ("left_trained_band",)}


# --- standardisation + features --------------------------------------------------------------------

def test_standardiser_uses_training_rows_only_and_missing_stays_nan():
    f = _frame({"a.x": np.r_[np.zeros(100) + 1.0, np.zeros(100) + 50.0]})
    f.iloc[150, 0] = np.nan
    stats = fa.fit_standardizer(f.iloc[:100].assign(**{"a.x": f["a.x"].iloc[:100] + np.linspace(0, 1, 100)}))
    assert stats["a.x"]["mean"] < 2.0          # the 50.0 rows (not training) never touched the statistics
    z = fa.standardize(f, stats)
    assert np.isnan(z[150, 0])


def test_missing_is_a_mask_channel_never_a_raw_zero():
    z = np.array([[1.0, 2.0], [np.nan, 3.0], [0.5, np.nan]])
    idx = pd.date_range(T0, periods=3, freq="5min", tz="UTC")
    X = fa.var_features(z, idx)
    m = len(fa.VAR_LAGS) * 2
    mask = X[:, m:m + 2]
    assert mask.tolist() == [[0, 0], [1, 0], [0, 1]]
    assert X[1, 0] == 0.0     # standardised 0 = the training mean, flagged by the mask -- not raw 0.0


# --- rungs + significance --------------------------------------------------------------------------

def _lead_lag_frame(seed: int, coupled: bool, days=12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = days * 288
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.9 * x[t - 1] + rng.standard_normal()
    y = np.zeros(n)
    noise = rng.standard_normal(n)
    for t in range(1, n):
        y[t] = 0.5 * y[t - 1] + (0.8 * x[t - 1] if coupled else 0.0) + 0.5 * noise[t]
    return _frame({"s.x": x, "s.y": y})


def test_ridge_beats_baselines_when_another_reading_leads():
    f = _lead_lag_frame(0, coupled=True)
    run = fa.run_horizon(f, ["s.x", "s.y"], ["s.x", "s.y"], 1, 5, seed=0)
    base, _pick = fa.best_baseline(run)
    cmp_ = fa.compare(run, "ridge_var", base, 5, 0)
    assert cmp_["wins"] and cmp_["ratio"] < 0.9


@pytest.mark.parametrize("seed", [1, 2])
def test_ridge_does_not_win_on_independent_readings(seed):
    f = _lead_lag_frame(seed, coupled=False)
    run = fa.run_horizon(f, ["s.x", "s.y"], ["s.x", "s.y"], 1, 5, seed=seed)
    base, _pick = fa.best_baseline(run)
    assert not fa.compare(run, "ridge_var", base, 5, seed)["wins"]


def test_ridge_at_huge_alpha_is_the_ar_forecast():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((500, 8))
    y = rng.standard_normal(500)
    fit = fa.ridge_path(X, y, alphas=(1e12,))[0]
    assert np.allclose(fit.predict(X), y.mean(), atol=1e-6)


def test_split_is_purged_and_seeds_move_the_blocks():
    tr0, he0 = fa.blocked_split(2000, 3, seed=0)
    tr1, he1 = fa.blocked_split(2000, 3, seed=1)
    assert not set(tr0) & set(he0)
    gap = fa.purge_rows(3)
    for h in he0:
        assert not any(0 < h - t <= gap - 1 for t in tr0[(tr0 < h) & (tr0 > h - gap)])
    assert set(he0) != set(he1)


def test_block_bootstrap_separates_a_real_gap_from_noise():
    rng = np.random.default_rng(8)
    assert fa.block_bootstrap_p(rng.standard_normal(1000) - 0.5, 24, 0)[0] < 0.01
    assert fa.block_bootstrap_p(rng.standard_normal(1000), 24, 0)[0] > 0.05


# --- joint surprise --------------------------------------------------------------------------------

def test_calm_point_is_the_expected_surprisal_of_a_uniform_conformal_p():
    n = 99
    p = np.arange(1, n + 2) / (n + 1)
    assert fa.calm_point(n) == pytest.approx(float(np.mean(-np.log(p))))
    assert fa.calm_point(10 ** 6) == pytest.approx(1.0, abs=1e-4)


def test_correlated_misses_count_once_not_per_reading():
    # One event moving four strongly coupled readings by 3 sd each: whitened d^2 ~ one 3-sd miss (9),
    # not four of them (36). This is the double-count fix.
    rho = 0.95
    cov = np.full((4, 4), rho) + np.eye(4) * (1 - rho)
    d2, _tail, _ = fa.mahalanobis_tail(np.full((1, 4), 3.0), cov, 1)
    assert d2[0] < 10.0
    d2i, _t, _ = fa.mahalanobis_tail(np.full((1, 4), 3.0), np.eye(4), 1)
    assert d2i[0] == pytest.approx(36.0)


def test_joint_excess_rests_at_zero_on_a_calibrated_model():
    means, covered = [], 0
    for seed in (9, 10, 11, 12):
        f = _lead_lag_frame(seed, coupled=True, days=14)
        run = fa.run_horizon(f, ["s.x", "s.y"], ["s.x", "s.y"], 1, 5, seed=0)
        js = fa.joint_surprise(run)
        ok = np.isfinite(js.excess)
        assert ok.sum() > 500
        assert abs(float(np.mean(js.excess[ok]))) < 0.15
        assert 0.02 < float(np.mean(js.p[ok] <= 0.05)) < 0.09
        means.append(float(np.mean(js.excess[ok])))
        rest = fa.rest_summary(js.excess, f.index[run.held], 0)
        covered += rest["mean_ci95"][0] < 0 < rest["mean_ci95"][1]
        assert rest["longest_hours_above_0_5"] <= 6
    assert abs(float(np.mean(means))) < 0.06
    assert covered >= 3     # a 95 % band on 4 independent histories


def test_conformal_p_is_bounded_and_monotone():
    cal = np.linspace(0.01, 1, 100)
    p = fa.conformal_p(np.array([0.0, 0.5, 1.0, np.nan]), cal)
    assert p[0] == pytest.approx(1 / 101) and p[2] == pytest.approx(1.0) and np.isnan(p[3])
    assert p[0] < p[1] < p[2]


# --- manifest + loader -----------------------------------------------------------------------------

def test_manifest_records_roles_reasons_and_train_only_standardisation():
    cfg = _cfg()
    sid = cfg.inputs[0].id
    v = {sid: fa.SeriesVerdict(sid, fa.TARGET, (), {"band": [1.0, 2.0]}),
         cfg.inputs[1].id: fa.SeriesVerdict(cfg.inputs[1].id, fa.REFUSED, ("absent",), {})}
    man = fa.build_manifest(cfg, v, {sid: {"mean": 1.5, "std": 0.2}}, (T0, T0 + timedelta(days=1)), T0)
    assert man["inputs"][sid]["standardize"] == {"mean": 1.5, "std": 0.2}
    assert man["inputs"][cfg.inputs[1].id]["reasons"] == ["absent"]
    assert fa.named_from_manifest(man) == {sid: fa.TARGET}
    assert man["hygiene"]["max_mode_share"] == 0.85


def test_queries_bind_every_key_and_read_measurements_only():
    cfg = _cfg()
    for sql, args, ids in build_queries(cfg.inputs, 5, T0, T0 + timedelta(days=1)):
        assert sql.count("%s") == len(args)
        assert "pressures" not in sql and "headroom" not in sql and "composites" not in sql
        for spec in cfg.inputs:
            if spec.source.key:
                assert not re.search(rf"\b{spec.source.key}\b", sql)   # keys are parameters, never SQL text
        assert ids
    assert sum(len(ids) for _s, _a, ids in build_queries(cfg.inputs, 5, T0, T0)) == len(cfg.inputs)


def test_purge_covers_longest_lag_and_lead():
    assert fa.purge_rows(3) >= max(fa.VAR_LAGS) + 3
    assert math.isfinite(fa.calm_point(1))
