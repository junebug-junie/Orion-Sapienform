from __future__ import annotations

import numpy as np

from app.backtest import (
    BacktestResult,
    exceedance_probability,
    horizon_metrics,
    residual_quantiles,
    run_backtest,
    select_champion,
    step_owner,
    stitch,
)
from app.models import build_models, split_seasonalities
from tests.helpers import daily_series, spec


def _result(name, res):
    return BacktestResult(name, np.asarray(res, dtype=float), [], 0.0)


def test_conformal_80_interval_covers_about_80_percent_of_fresh_draws():
    rng = np.random.default_rng(1)
    res = rng.normal(0, 1, (400, 3))
    q = residual_quantiles(res)
    fresh = rng.normal(0, 1, 20000)
    lo, hi = q[2, 0], q[6, 0]  # 0.1 and 0.9
    cover = np.mean((fresh >= lo) & (fresh <= hi))
    assert 0.77 <= cover <= 0.83


def test_quantiles_are_monotone_and_unknown_below_min_paths():
    rng = np.random.default_rng(2)
    res = rng.normal(0, 1, (12, 4))
    res[3:, 3] = np.nan  # only 3 residuals at step 4
    q = residual_quantiles(res)
    assert np.all(np.diff(q[:, :3], axis=0) >= 0)
    assert np.all(np.isnan(q[:, 3]))


def test_holdout_coverage_is_near_nominal_for_iid_noise_and_catches_drift():
    rng = np.random.default_rng(3)
    iid = rng.normal(0, 1, (600, 1))
    m = horizon_metrics(iid, 0)
    assert m["n"] == 600 and m["n_holdout"] == 200 and 0.74 <= m["coverage80_holdout"] <= 0.86
    assert abs(m["mae"] - np.sqrt(2 / np.pi)) < 0.1
    drift = iid.copy()
    drift[400:] *= 3          # errors triple in the newest third (regime change)
    assert horizon_metrics(drift, 0)["coverage80_holdout"] < 0.5


def test_leave_one_out_coverage_could_not_have_caught_that():
    # Why the gate dropped it: rank-only, so the drifted set reads the same as the calm one.
    from app.backtest import _level

    def loo(col):
        hits = 0
        for i in range(len(col)):
            o = np.delete(col, i)
            lo = np.quantile(o, _level(len(o), 0.1), method="lower")
            hi = np.quantile(o, _level(len(o), 0.9), method="higher")
            hits += lo <= col[i] <= hi
        return hits / len(col)
    rng = np.random.default_rng(4)
    calm = rng.normal(0, 1, 40)
    drifted = calm.copy()
    drifted[27:] *= 3
    assert loo(calm) == loo(drifted)


def test_exceedance_from_residual_paths():
    point = np.array([30.0, 30.0, 30.0])
    res = np.zeros((20, 3))
    res[:5, 2] = 2.5            # 5 of 20 paths reach 32.5 at step 3
    p, n = exceedance_probability(point, res, 32.0, 3)
    assert n == 20 and p == 0.25
    p, _ = exceedance_probability(point, res, 32.0, 2)   # window ends before step 3
    assert p == 0.0
    res[:15, 0] = np.nan       # only 5 complete paths -> unknown, never 0
    assert exceedance_probability(point, res, 32.0, 3) == (None, 5)


def test_step_owner_and_stitch():
    owners = step_owner({5: "a", 15: "b"}, 5, 4)
    assert owners == ["a", "b", "b", "b"]   # steps beyond the last horizon belong to the last owner
    paths = {"a": np.array([1, 1, 1, 1.0]), "b": np.array([2, 2, 2, 2.0])}
    assert stitch(paths, owners).tolist() == [1, 2, 2, 2]
    mats = {"a": np.zeros((3, 4)), "b": np.ones((3, 4))}
    assert stitch(mats, owners).tolist()[0] == [0, 1, 1, 1]


def test_gate_uses_the_better_baseline_per_horizon_and_falls_back():
    sp = spec(horizons=["5min", "1h"])
    rng = np.random.default_rng(4)
    h = sp.max_horizon_steps
    naive = rng.normal(0, 0.1, (30, h))           # persistence is excellent at both horizons
    snaive = rng.normal(0, 2.0, (30, h))
    good_long = rng.normal(0, 0.05, (30, h))       # better than naive -> serves
    bad = rng.normal(0, 1.0, (30, h))              # beats seasonal-naive but not naive -> must NOT serve
    sel = select_champion({"naive": _result("naive", naive), "seasonal_naive": _result("seasonal_naive", snaive),
                           "good": _result("good", good_long), "bad": _result("bad", bad)}, sp)
    assert sel.baseline_assignment == {5: "naive", 60: "naive"}
    assert sel.assignment == {5: "good", 60: "good"} and not sel.served_baseline
    assert sel.metrics["bad"]["60"]["passed"] is False
    sel2 = select_champion({"naive": _result("naive", naive), "seasonal_naive": _result("seasonal_naive", snaive),
                            "bad": _result("bad", bad)}, sp)
    assert sel2.served_baseline and sel2.assignment == {5: "naive", 60: "naive"}


def test_a_lower_average_that_is_noise_is_not_served_but_a_real_win_is():
    sp = spec(horizons=["5min"])
    h = sp.max_horizon_steps
    rng = np.random.default_rng(6)
    naive = rng.normal(0, 1.0, (40, h))
    # A lower average (rel_mae ~0.86) that window-to-window is not consistent: p ~ 0.2, like the
    # live 2026-10-16 losers (p 0.21 / 0.31).
    noisy = rng.normal(0, 0.97, (40, h))
    real = naive * 0.7 + rng.normal(0, 0.02, (40, h))
    res = {"naive": _result("naive", naive), "seasonal_naive": _result("seasonal_naive", naive * 3)}
    sel = select_champion({**res, "noisy": _result("noisy", noisy)}, sp)
    m = sel.metrics["noisy"]["5"]
    assert m["p_win"] > 0.05 and m["passed"] is False and sel.assignment[5] == "naive"
    sel = select_champion({**res, "noisy": _result("noisy", noisy), "real": _result("real", real)}, sp)
    assert sel.metrics["real"]["5"]["p_win"] < 0.001 and sel.assignment[5] == "real"


def test_real_backtest_loop_on_synthetic_daily_series_prefers_seasonal_structure_at_6h():
    sp = spec(horizons=["5min", "6h"], train_window="6d")
    s = daily_series(10, amp=1.5, noise=0.05, seed=7)
    usable, refused = split_seasonalities(sp, 6 * 288)
    assert refused == []
    res = {n: run_backtest(s, sp, n, lambda u, n=n: build_models(sp, u)[n], 12, 36)
           for n in ("naive", "seasonal_naive", "mstl_ets")}
    assert all(r.n_windows == 12 for r in res.values())
    sel = select_champion(res, sp)
    # A clean daily sine: persistence is fine at 5 min but loses badly at 6 h to anything seasonal.
    assert sel.metrics["naive"]["360"]["mae"] > sel.metrics["seasonal_naive"]["360"]["mae"]
    assert sel.baseline_assignment[360] == "seasonal_naive"


def test_gate_is_paired_a_model_that_fails_the_hard_windows_cannot_win():
    # Review repro: candidate is 1.5x worse than naive on shared windows but NaN on the hard ones.
    sp = spec(horizons=["5min"])
    h = sp.max_horizon_steps
    rng = np.random.default_rng(11)
    naive = rng.normal(0, 0.1, (30, h))
    naive[:20] *= 10                       # 20 hard windows
    cand = naive * 1.5
    cand[:20] = np.nan                     # candidate "fails" exactly there
    sel = select_champion({"naive": _result("naive", naive), "seasonal_naive": _result("seasonal_naive", naive * 3),
                           "cand": _result("cand", cand)}, sp)
    m = sel.metrics["cand"]["5"]
    assert m["rel_mae"] > 1.4 and m["n_paired"] == 10
    assert sel.assignment[5] == "naive"


def test_gate_refuses_a_good_model_that_skipped_too_many_windows():
    sp = spec(horizons=["5min"])
    h = sp.max_horizon_steps
    naive = np.random.default_rng(12).normal(0, 1.0, (40, h))
    cand = naive * 0.5
    cand[:6] = np.nan                      # 34/40 = 85% < 90%
    sel = select_champion({"naive": _result("naive", naive), "seasonal_naive": _result("seasonal_naive", naive * 3),
                           "cand": _result("cand", cand)}, sp)
    assert sel.metrics["cand"]["5"]["rel_mae"] < 1 and sel.assignment[5] == "naive"


def test_backtest_decides_seasonalities_per_window_and_scores_against_real_actuals():
    sp = spec(horizons=["5min"], train_window="10d", seasonalities=[{"name": "weekly", "period": "7d"}])
    s = daily_series(17, seed=13)
    seen = []

    def make(usable):
        seen.append(tuple(x.name for x in usable))
        return build_models(sp, usable)["naive"]
    actuals = s.copy()
    actuals.iloc[-200:] = np.nan             # gaps the filled series would have hidden
    r = run_backtest(s, sp, "naive", make, 20, 36, actuals=actuals)
    assert () in seen                        # early windows: under 14 days of training -> no weekly
    assert np.isnan(r.residuals[-1, 0])      # newest window's actual is a real gap -> no residual


def test_paired_test_keeps_false_wins_near_alpha_on_correlated_windows():
    from scipy import stats

    from app.backtest import paired_win_p
    rng = np.random.default_rng(8)
    n, phi, trials = 40, 0.5, 400
    naive_fp = fp = power = 0
    for _ in range(trials):
        d = np.empty(n)
        d[0] = rng.normal()
        for i in range(1, n):
            d[i] = phi * d[i - 1] + rng.normal(0, np.sqrt(1 - phi ** 2))
        ref = np.full(n, 2.0)
        fp += paired_win_p(ref + 0.3 * d, ref, lag=1)[0] < 0.05          # no true advantage
        naive_fp += stats.ttest_1samp(0.3 * d, 0, alternative="less").pvalue < 0.05
        power += paired_win_p(ref + 0.3 * d - 0.3, ref, lag=1)[0] < 0.05  # a real 1-sd advantage
    assert naive_fp / trials > 0.12           # the iid t-test is fooled by correlated windows
    assert fp / trials <= 0.09                # this one stays near alpha
    assert power / trials >= 0.8              # and still finds a real win


def test_overlap_lag_grows_with_horizon():
    from app.backtest import overlap_lag
    assert overlap_lag(5, 180) == 1 and overlap_lag(360, 180) == 1 and overlap_lag(1440, 180) == 7
