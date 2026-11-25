"""Born machine + network forecaster: known answers (spec 2026-11-24 rev 3, "Acceptance checks").

Small synthetic systems, fixed seeds, pure numpy.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from orion.substrate import born_machine as bm
from orion.substrate import demon

FAST_DIMS = (2, 4, 8)


def simulate_var(A, T, *, seed, burn=200):
    rng = np.random.default_rng(seed)
    A = np.asarray(A, dtype=float)
    x = np.zeros((T + burn, A.shape[0]))
    for t in range(1, T + burn):
        x[t] = A @ x[t - 1] + rng.normal(0, 1, A.shape[0])
    return x[burn:]


def simulate_switch(T, *, seed):
    """Organ B jumps (up or down, at random) only when organ A was high (top third) one step before.

    B's mean never depends on A, so a linear model sees no coupling at all; only B's spread does.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros((T, 2))
    for t in range(1, T):
        x[t, 0] = 0.8 * x[t - 1, 0] + rng.normal()
        jump = 3.0 * rng.choice([-1.0, 1.0]) if x[t - 1, 0] > 0.72 else 0.0
        x[t, 1] = 0.3 * x[t - 1, 1] + jump + 0.5 * rng.normal()
    return x


def random_chain(L, d, D, *, seed):
    rng = np.random.default_rng(seed)
    dims = [1] + [min(D, d ** min(i, L - i)) for i in range(1, L)] + [1]
    return bm.BornMachine(tuple(rng.normal(size=(dims[i], d, dims[i + 1])) for i in range(L)))


def brute_force(chain):
    L, d = chain.n_sites, chain.d
    configs = np.array(list(itertools.product(range(d), repeat=L)))
    psi = bm.amplitudes(chain.tensors, configs)
    return configs, psi ** 2 / (psi ** 2).sum()


def network_clearly_beats_linear(errs, *, min_skill=0.05, min_p_real=0.9):
    """The threshold test's pass rule: network skill over linear at k=1, with P(real)."""
    s = demon.skill_spread(errs.whole, errs.linear, block=12, n_boot=300, rng=np.random.default_rng(0))
    return bool(s.skill[0] > min_skill and s.p_real[0] >= min_p_real), s


# --- Born machine ------------------------------------------------------------------------------


def test_log_prob_is_normalised_and_matches_brute_force():
    chain = random_chain(5, 3, 4, seed=1)
    configs, p = brute_force(chain)
    assert np.allclose(np.exp(bm.log_prob(chain, configs)), p)
    assert np.isclose(bm.full_probabilities(chain).sum(), 1.0)


def test_conditional_marginals_match_brute_force():
    chain = random_chain(6, 3, 3, seed=2)
    configs, p = brute_force(chain)
    for given, target in [([0, 2], [3]), ([4], [1]), ([1, 3, 5], [0, 4]), ([5, 0], [2])]:
        table = bm.conditional_table(chain, given, target)
        for g in itertools.product(range(3), repeat=len(given)):
            mask = np.all(configs[:, given] == g, axis=1)
            for t in itertools.product(range(3), repeat=len(target)):
                hit = mask & np.all(configs[:, target] == t, axis=1)
                assert np.isclose(table[g + t], p[hit].sum() / p[mask].sum(), atol=1e-10)


def test_marginal_table_matches_brute_force():
    chain = random_chain(6, 3, 4, seed=3)
    configs, p = brute_force(chain)
    table = bm.marginal_table(chain, [4, 1])  # axes come back in site order: (1, 4)
    for a, b in itertools.product(range(3), repeat=2):
        assert np.isclose(table[a, b], p[(configs[:, 1] == a) & (configs[:, 4] == b)].sum())


@pytest.mark.parametrize("seed", [4, 5, 6])
def test_entanglement_bounds_mutual_information_on_random_chains(seed):
    chain = random_chain(6, 3, 4, seed=seed)
    ent = bm.entanglement_entropies(chain)
    mi, se = bm.mutual_information(chain)
    assert np.all(se == 0)
    assert np.all(ent >= mi - 1e-9)


def test_mutual_information_exact_matches_brute_force_and_monte_carlo():
    chain = random_chain(5, 3, 3, seed=7)
    configs, p = brute_force(chain)
    exact, _ = bm.mutual_information(chain)
    P = p.reshape(9, 27)  # cut after site 1
    pl, pr = P.sum(1, keepdims=True), P.sum(0, keepdims=True)
    assert np.isclose(exact[1], float((P * np.log2(P / (pl @ pr))).sum()))
    mc, se = bm.mutual_information(chain, max_exact=10, n_samples=20000, rng=np.random.default_rng(0))
    assert np.all(np.abs(mc - exact) < 4 * se + 1e-3)


def test_fit_learns_a_correlation_the_product_model_cannot():
    rng = np.random.default_rng(8)
    x0 = rng.integers(0, 3, 3000)
    x1 = np.where(rng.random(3000) < 0.8, x0, rng.integers(0, 3, 3000))
    x2 = rng.integers(0, 3, 3000)
    data = np.stack([x0, x1, x2], axis=1)
    product = bm.fit(data, bond_dim=1, n_sweeps=4)
    chain, sel = bm.select_and_fit(data, bond_dims=FAST_DIMS)
    assert sel.bond_dim >= 2
    gain = float(np.mean(bm.log_prob(chain, data) - bm.log_prob(product, data))) / np.log(2)
    # True I(x0; x1) for this channel is about 0.80 bits.
    assert 0.7 < gain < 0.9
    ent = bm.entanglement_entropies(chain)
    mi, _ = bm.mutual_information(chain)
    assert np.all(ent >= mi - 1e-9)
    assert mi[1] < 0.02  # x2 is independent of (x0, x1)


# --- Network forecaster ------------------------------------------------------------------------


def test_network_is_within_spread_of_linear_on_var1_data():
    x = simulate_var([[0.6, 0.3], [0.0, 0.6]], 2400, seed=1)
    errs = demon.network_rolling_errors(x, window=1800, step=600, lags=(1,), bond_dims=FAST_DIMS)
    s = demon.skill_spread(errs.whole, errs.linear, block=12, n_boot=300, rng=np.random.default_rng(0))
    assert s.p10[0] <= 0.0 <= s.p90[0] or abs(s.skill[0]) < 0.01
    # Both carry real foresight against the level baselines.
    assert demon.skill_curve(errs.whole, errs.baseline)[0] > 0.05


def test_network_clearly_beats_linear_on_a_threshold_switch():
    errs = demon.network_rolling_errors(simulate_switch(2400, seed=2), window=1800, step=600, lags=(1,), bond_dims=FAST_DIMS)
    ok, s = network_clearly_beats_linear(errs)
    assert ok, s
    assert s.skill[0] > 0.08


def test_mutation_linear_in_place_of_network_fails_the_threshold_test(monkeypatch):
    """Spec mutation check: swap the network forecast for the linear one. The threshold test fails."""

    def linear_instead(self, now, k, partition=None):
        return self.linear_probs(now, k)

    monkeypatch.setattr(demon.NetworkForecaster, "network_probs", linear_instead)
    errs = demon.network_rolling_errors(simulate_switch(2400, seed=2), window=1800, step=600, lags=(1,), bond_dims=FAST_DIMS)
    ok, s = network_clearly_beats_linear(errs)
    assert not ok
    assert abs(s.skill[0]) < 1e-12  # it is the linear forecast, so it scores exactly the same


def test_independent_blocks_cost_about_zero_and_are_the_mip():
    b = [[0.2, 0.7], [0.7, 0.2]]
    A = np.zeros((4, 4))
    A[:2, :2] = b
    A[2:, 2:] = b
    res, _ = demon.held_out_lattice(simulate_var(A, 2400, seed=3), lags=(1,), train_rows=1600, bond_dims=FAST_DIMS)
    assert len(res.partitions) == 15  # Bell(4)
    split = res.partitions.index(((0, 1), (2, 3)))
    assert abs(res.network_bits[split, 0]) < 0.05
    assert res.mip("network", 0) == split
    assert res.mip("linear", 0) == split
    # Any cut through a block costs real foresight.
    others = [i for i, p in enumerate(res.partitions) if len(p) > 1 and i != split]
    assert res.network_bits[others, 0].min() > 0.3


def test_shifted_data_cut_cost_is_about_zero():
    x = simulate_var([[0.2, 0.7], [0.7, 0.2]], 2400, seed=4)
    full = [[0], [1]]
    coupled, _ = demon.held_out_lattice(x, lags=(1,), train_rows=1200, partitions=[full], bond_dims=FAST_DIMS)
    shifted = demon.time_shift_null(x, rng=np.random.default_rng(5))
    null, _ = demon.held_out_lattice(shifted, lags=(1,), train_rows=1200, partitions=[full], bond_dims=FAST_DIMS)
    assert coupled.network_bits[0, 0] > 0.3
    assert abs(null.network_bits[0, 0]) < 0.03


def test_lattice_vs_null_marks_a_real_cut_real():
    x = simulate_var([[0.2, 0.7], [0.7, 0.2]], 1500, seed=6)
    res = demon.lattice_vs_null(x, lags=(1,), n_null=5, rng=np.random.default_rng(7), bond_dims=(2, 4))
    assert res.null_network_bits.shape == (5, 2, 1)
    assert res.p_real[1, 0] == 1.0  # the full cut, against 5 shifted refits
    assert abs(res.null_network_bits[:, 1, 0]).max() < 0.05


def test_existing_readouts_work_unchanged_on_network_errors():
    x = simulate_var([[0.5, 0.6], [0.0, 0.5]], 1400, seed=8)
    kw = dict(window=900, step=250, horizon=2)
    errs = demon.network_rolling_errors(x, partitions={"cut": demon.full_cut(2)}, bond_dims=(2, 4), **kw)
    assert errs.whole.shape == errs.linear.shape == (498, 2) and errs.lags == (1, 2)
    est = demon.bootstrap_horizon(errs, n_boot=100, rng=np.random.default_rng(0))
    assert est.steps >= 1 and est.p_positive[0] > 0.9
    assert demon.cut_cost_curve(errs, "cut")[0] > 0
    res = demon.cut_cost_vs_null(
        x, demon.full_cut(2), n_null=3, rng=np.random.default_rng(1), guard=24,
        errors_fn=lambda xx, **k: demon.network_rolling_errors(xx, bond_dims=(2, 4), **k), **kw,
    )
    assert res.cost[0] > res.null_mean[0] and res.p_real[0] == 1.0


def test_set_partitions_counts_are_bell_numbers():
    assert [len(demon.set_partitions(n)) for n in range(1, 7)] == [1, 2, 5, 15, 52, 203]
    for p in demon.set_partitions(4):
        assert sorted(i for part in p for i in part) == [0, 1, 2, 3]


def test_partition_normalization():
    assert demon.partition_normalization([[0, 1, 2]]) == 0.0
    assert np.isclose(demon.partition_normalization([[0, 1], [2]]), np.log2(3))
    assert np.isclose(demon.partition_normalization([[0, 1], [2, 3]]), 2 * np.log2(3))


def test_persistence_and_gaussian_levels_are_distributions():
    fc_stay = np.array([0.7, 0.4])
    fc = demon.NetworkForecaster.__new__(demon.NetworkForecaster)
    object.__setattr__(fc, "stay", {1: fc_stay})
    object.__setattr__(fc, "edges", np.array([[-0.4, 0.4], [-0.4, 0.4]]))
    p = fc.persistence_probs(np.array([[0, 2], [1, 1]]), 1)
    assert np.allclose(p.sum(axis=2), 1.0)
    assert np.isclose(p[0, 0, 0], 0.7) and np.isclose(p[0, 1, 2], 0.4) and np.isclose(p[0, 1, 0], 0.3)
    g = demon.gaussian_level_probs(np.array([[0.0, 5.0]]), np.array([1.0, 1.0]), fc.edges)
    assert np.allclose(g.sum(axis=2), 1.0) and g[0, 1, 2] > 0.99
