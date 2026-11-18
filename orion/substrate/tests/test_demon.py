"""Field demon: synthetic systems with known answers (spec rev 2, "Acceptance checks", patch 1)."""
from __future__ import annotations

import numpy as np
import pytest

from orion.substrate import demon


def simulate(A, T, *, seed, noise=1.0, A2=None, burn=200):
    rng = np.random.default_rng(seed)
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    x = np.zeros((T + burn, n))
    for t in range(2, T + burn):
        x[t] = A @ x[t - 1] + rng.normal(0, noise, n)
        if A2 is not None:
            x[t] += np.asarray(A2) @ x[t - 2]
    return x[burn:]


COUPLED = [[0.5, 0.6], [0.0, 0.5]]  # organ 1 drives organ 0
UNCOUPLED = [[0.5, 0.0], [0.0, 0.5]]


def test_fit_var1_recovers_a():
    x = simulate(COUPLED, 20000, seed=1)
    model = demon.fit_var1(x)
    assert np.allclose(model.A, COUPLED, atol=0.03)
    assert np.allclose(model.sigma, np.eye(2), atol=0.05)


def test_fit_cut_is_block_diagonal():
    x = simulate(COUPLED, 5000, seed=2)
    cut = demon.fit_cut(x, demon.full_cut(2))
    assert cut.A[0, 1] == 0.0 and cut.A[1, 0] == 0.0


def test_bad_partition_refused():
    x = simulate(UNCOUPLED, 500, seed=3)
    with pytest.raises(ValueError):
        demon.fit_cut(x, [[0]])
    with pytest.raises(ValueError):
        demon.fit_cut(x, [[0, 1], [1]])


def test_coupled_organs_cut_costs_foresight_and_cost_shrinks():
    x = simulate(COUPLED, 288 * 12, seed=4)
    errs = demon.rolling_errors(x, partitions={"full_cut": demon.full_cut(2)})
    cost = demon.cut_cost_curve(errs, "full_cut")
    assert cost[0] > 0.05
    assert cost[0] > cost[5] > cost[11] - 0.01


def test_uncoupled_organs_cost_is_null_and_slightly_negative():
    x = simulate(UNCOUPLED, 288 * 12, seed=5)
    errs = demon.rolling_errors(x, partitions={"full_cut": demon.full_cut(2)})
    cost = demon.cut_cost_curve(errs, "full_cut")
    assert np.all(np.abs(cost) < 0.02)
    # The whole model fits extra cross terms that carry nothing, so it overfits a little.
    assert cost[:3].mean() <= 0.0


def test_time_shift_null_breaks_coupling():
    x = simulate(COUPLED, 288 * 14, seed=6)
    shifted = demon.time_shift_null(x, rng=np.random.default_rng(0))
    assert shifted.shape[1] == 2 and shifted.shape[0] == len(x) - demon.DEFAULT_NULL_GUARD
    errs = demon.rolling_errors(shifted, partitions={"full_cut": demon.full_cut(2)})
    assert np.all(np.abs(demon.cut_cost_curve(errs, "full_cut")) < 0.02)


def test_skill_against_clock_matches_analytic_curve():
    """With a long fit window, out-of-sample skill vs the clock is tr(A^k Γ0 A^kᵀ) / tr(Γ0)."""
    A = np.diag([0.8, 0.6])
    x = simulate(A, 40000, seed=7)
    errs = demon.rolling_errors(x, window=5000, step=500)
    skill = demon.skill_curve(errs.whole, errs.clock)
    gamma0 = np.diag([1 / (1 - 0.64), 1 / (1 - 0.36)])
    # Errors are summed over standardized organs, so each organ's variance counts as 1.
    std_gamma0 = np.eye(2)
    for k in (1, 2, 3):
        Ak = np.linalg.matrix_power(A, k)
        analytic = np.trace(Ak @ gamma0 @ Ak.T / np.diag(gamma0)[:, None]) / np.trace(std_gamma0)
        assert skill[k - 1] == pytest.approx(analytic, abs=0.03)


def test_slower_dynamics_see_further():
    fast = demon.rolling_errors(simulate(np.diag([0.4, 0.4]), 288 * 10, seed=8))
    slow = demon.rolling_errors(simulate(np.diag([0.8, 0.8]), 288 * 10, seed=8))
    h_fast = demon.horizon_steps(demon.skill_curve(fast.whole, fast.baseline))
    h_slow = demon.horizon_steps(demon.skill_curve(slow.whole, slow.baseline))
    assert h_fast < h_slow


def test_more_history_sees_at_least_as_far():
    x = simulate(np.diag([0.7, 0.7, 0.7, 0.7]), 288 * 20, seed=9)
    short = demon.rolling_errors(x, window=72, step=12)
    long = demon.rolling_errors(x, window=2880, step=12)
    s_short = demon.skill_curve(short.whole, short.baseline)
    s_long = demon.skill_curve(long.whole, long.baseline)
    assert demon.horizon_steps(s_long) >= demon.horizon_steps(s_short)
    assert s_long[-1] > s_short[-1]


def test_white_noise_has_no_horizon():
    x = np.random.default_rng(10).normal(size=(288 * 8, 3))
    errs = demon.rolling_errors(x)
    est = demon.bootstrap_horizon(errs, n_boot=200, rng=np.random.default_rng(0))
    assert est.steps == 0 or est.skill[0] < 0.01
    assert est.minutes == est.steps * 5
    assert est.p10_steps <= est.median_steps <= est.p90_steps


def test_bootstrap_horizon_spread_and_p_positive():
    x = simulate(np.diag([0.8, 0.8]), 288 * 10, seed=11)
    est = demon.bootstrap_horizon(demon.rolling_errors(x), n_boot=200, rng=np.random.default_rng(1))
    assert est.steps >= 3
    assert est.p_positive[0] > 0.95
    assert est.p10_steps <= est.median_steps <= est.p90_steps
    assert est.p_positive.shape == (demon.DEFAULT_HORIZON,)


def test_horizon_steps_uses_first_drop():
    assert demon.horizon_steps(np.array([0.3, 0.1, -0.01, 0.2])) == 2
    assert demon.horizon_steps(np.array([-0.1, 0.5])) == 0
    assert demon.horizon_steps(np.array([0.3, 0.2])) == 2
    assert demon.horizon_steps(np.array([np.nan, 0.2])) == 0


def test_direct_and_iterated_agree_on_var1():
    x = simulate([[0.6, 0.2], [0.1, 0.5]], 30000, seed=12)
    model = demon.fit_var1(x)
    for k in (1, 3, 6):
        Ak, sigk, g0 = demon.lag_k_model(x, k)
        direct = demon.predictive_information(g0, sigk)
        iterated = demon.iterated_predictive_information(model, g0, k)
        assert direct == pytest.approx(iterated, abs=0.02)


def test_direct_beats_iterated_on_var2():
    """A strong lag-2 term is invisible to VAR(1): the misfit check must see it."""
    x = simulate(np.diag([0.1, 0.1]), 30000, seed=13, A2=np.diag([0.7, 0.7]))
    model = demon.fit_var1(x)
    _, sig2, g0 = demon.lag_k_model(x, 2)
    assert demon.predictive_information(g0, sig2) > demon.iterated_predictive_information(model, g0, 2) + 0.3


def test_refusal_names_decaying_input():
    rng = np.random.default_rng(14)
    healthy = rng.normal(size=300)
    decaying = np.concatenate([rng.normal(size=100), 5.0 * 0.92 ** np.arange(200)])
    x = np.stack([healthy, decaying], axis=1)
    assert demon.refusal_reason(x, ["bus", "route_pe"]) == "decaying:route_pe"


def test_refusal_missing_and_constant_and_clean():
    rng = np.random.default_rng(15)
    x = rng.normal(size=(300, 2))
    assert demon.refusal_reason(x, ["a", "b"]) is None
    x_nan = x.copy()
    x_nan[10, 1] = np.nan
    assert demon.refusal_reason(x_nan, ["a", "b"]) == "missing:b"
    x_const = x.copy()
    x_const[:, 0] = 3.0
    assert demon.refusal_reason(x_const, ["a", "b"]) == "constant:a"


def test_flat_stretch_is_not_decay():
    rng = np.random.default_rng(16)
    col = np.concatenate([rng.normal(size=100), np.full(60, 2.0), rng.normal(size=100)])
    assert demon.refusal_reason(col[:, None], ["a"]) is None


def test_lag_k_model_shapes_feed_integration_rung():
    x = simulate(COUPLED, 2000, seed=17)
    Ak, sigk, g0 = demon.lag_k_model(x, 3)
    assert Ak.shape == sigk.shape == g0.shape == (2, 2)
    assert np.allclose(sigk, sigk.T)
    assert np.all(np.linalg.eigvalsh(sigk) > 0)


def test_too_short_series_refused():
    with pytest.raises(ValueError):
        demon.rolling_errors(np.zeros((50, 2)))
    with pytest.raises(ValueError):
        demon.time_shift_null(np.zeros((40, 3)), guard=36)
