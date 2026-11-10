"""Gate tests for the causal-geometry p-value (2026-11-10 floor-cliff fix).

Three claims, each pinned by synthetic data:

1. The FFT surrogate shortcut computes exactly the Pearson r of each
   circular shift (so the null is unchanged, only faster).
2. The cliff: with the old add-one rank p, a strong real lagged link among
   m=171 pairs gets 0 edges; with the continuous p it survives BH.
3. Calibration: on independent AR(1) pairs the raw p is below 0.05 / 0.01
   at about the nominal rate, across many seeds and up to phi=0.98; the full
   pipeline on 19 independent AR(1) channels (m=171) almost never emits an edge.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pytest

from orion.substrate import causal_geometry_engine as cge


def _ar1_batch(rng: np.random.Generator, k: int, n: int, phi: float, burn: int = 300) -> np.ndarray:
    """k independent AR(1) series of length n, unit innovations."""
    e = rng.standard_normal((k, n + burn))
    x = np.empty_like(e)
    x[:, 0] = e[:, 0]
    for t in range(1, n + burn):
        x[:, t] = phi * x[:, t - 1] + e[:, t]
    return x[:, burn:]


def _buckets(values: np.ndarray) -> Dict[int, float]:
    return {i: float(v) for i, v in enumerate(values)}


def _legacy_rank_pvalue(
    xs: np.ndarray,
    ys: np.ndarray,
    observed_r: float,
    *,
    n_surrogates: int = cge.DEFAULT_N_SURROGATES,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """The pre-2026-11-10 add-one rank p, kept here only to reproduce the cliff."""
    rs = cge.circular_shift_surrogate_rs(xs, ys, n_surrogates=n_surrogates, rng=rng)
    count_ge = int(np.sum(np.abs(rs) >= abs(observed_r)))
    return (count_ge + 1) / (n_surrogates + 1)


def _cliff_fixture(seed: int = 11, n: int = 600) -> Dict[str, Dict[int, float]]:
    """19 channels -> C(19,2) = 171 pairs (the live family size).

    `power` drives `thermal` two buckets (10 min) later; thermal is smooth
    (phi=0.9) like a real heat sensor. The other 17 channels are independent
    AR(1) noise with live-like persistence.
    """
    rng = np.random.default_rng(seed)
    power = _ar1_batch(rng, 1, n, 0.8)[0]
    noise = _ar1_batch(rng, 1, n, 0.5)[0]
    thermal = np.zeros(n)
    for t in range(2, n):
        thermal[t] = 0.6 * thermal[t - 1] + 0.5 * power[t - 2] + 1.2 * noise[t]
    channels = {"biometrics:power": _buckets(power), "biometrics:thermal": _buckets(thermal)}
    others = _ar1_batch(rng, 17, n, 0.9)
    for i, series in enumerate(others):
        channels[f"noise:c{i:02d}"] = _buckets(series)
    return channels


def _edges(results):
    return {frozenset((r.channel_a, r.channel_b)) for r in results if r.significant}


# --- 1. FFT shortcut == explicit roll + Pearson ------------------------------


def test_fft_surrogates_equal_explicit_roll_pearson():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(257)
    y = 0.4 * x + rng.standard_normal(257)
    rs = cge.circular_shift_surrogate_rs(x, y, n_surrogates=64, rng=np.random.default_rng(9))
    shifts = np.random.default_rng(9).integers(1, 257, size=64)
    expected = np.array([cge._pearson(x, np.roll(y, int(s))) for s in shifts])
    np.testing.assert_allclose(rs, expected, atol=1e-12)


def test_constant_series_gives_p_one():
    x = np.arange(50, dtype=float)
    flat = np.full(50, 3.0)
    assert cge.circular_shift_pvalue(x, flat, 0.0, rng=np.random.default_rng(0)) == 1.0
    assert cge.circular_shift_pvalue(x[:1], flat[:1], 0.0) == 1.0


def test_pvalue_is_continuous_and_never_zero():
    """A perfect link gets a tiny-but-positive p, far below any floor the old
    rank p could reach (1/(n_surrogates+1))."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(400)
    p = cge.circular_shift_pvalue(x, x.copy(), 1.0, n_surrogates=50, rng=rng)
    assert 0.0 < p < 1.0 / 51.0
    assert p >= cge.MIN_REPORTED_PVALUE
    # Monotone in |r|: a stronger observed r never gets a larger p.
    ps = [cge.circular_shift_pvalue(x, x, r, rng=np.random.default_rng(1)) for r in (0.05, 0.1, 0.2, 0.4)]
    assert ps == sorted(ps, reverse=True)
    assert len(set(ps)) == 4


# --- 2. The cliff -------------------------------------------------------------


def test_cliff_legacy_rank_p_finds_nothing_continuous_p_finds_the_real_link(monkeypatch):
    channels = _cliff_fixture()
    real = frozenset(("biometrics:power", "biometrics:thermal"))

    new_results = cge.compute_pairwise_results(channels, seed=1337)
    assert len(new_results) == 171
    by_pair = {frozenset((r.channel_a, r.channel_b)): r for r in new_results}
    winner = by_pair[real].winner
    assert winner is not None and winner.r > 0.5
    # Direction is recovered; the exact lag smears (thermal is smooth), but it is positive.
    assert (winner.source_id, winner.target_id) == ("biometrics:power", "biometrics:thermal")
    assert winner.lag_sec > 0
    new_edges = _edges(new_results)
    print(f"cliff fixture: real r={winner.r:.3f} p={by_pair[real].p_value:.3g} edges={len(new_edges)}")
    assert real in new_edges, f"real link p={by_pair[real].p_value} did not survive BH at m=171"
    assert len(new_edges - {real}) <= 1, f"too many noise edges: {sorted(map(sorted, new_edges - {real}))}"

    # Reproduce the live bug: the rank p cannot go below 1/2001, so after the
    # 10-way Bonferroni the real link sits at 0.005 > 0.05/171 and BH passes nothing.
    monkeypatch.setattr(cge, "circular_shift_pvalue", _legacy_rank_pvalue)
    old_results = cge.compute_pairwise_results(channels, seed=1337)
    old_by_pair = {frozenset((r.channel_a, r.channel_b)): r for r in old_results}
    assert old_by_pair[real].p_value == pytest.approx(10 / 2001)
    assert _edges(old_results) == set(), "the legacy rank p should reproduce the 0-edge cliff"


# --- 3. Calibration -----------------------------------------------------------


@pytest.mark.parametrize("phi,n", [(0.5, 2000), (0.9, 2000), (0.98, 2000), (0.9, 300)])
def test_raw_pvalue_false_positive_rate_is_nominal_on_independent_ar1(phi, n):
    """600 independent AR(1) pairs per setting. Binomial sd at 600 draws is
    ~0.009 at 5% and ~0.004 at 1%; bounds are ~3 sd. The old Bartlett
    effective-n test with an n/5 ACF truncation ran ~0.065-0.07 at 5% for
    phi>=0.95 on the same check -- that is why the surrogate fit was chosen."""
    rng = np.random.default_rng(int(phi * 1000) + n)
    k = 600
    xs = _ar1_batch(rng, k, n, phi)
    ys = _ar1_batch(rng, k, n, phi)
    ps = np.empty(k)
    for i in range(k):
        r = cge._pearson(xs[i], ys[i])
        ps[i] = cge.circular_shift_pvalue(xs[i], ys[i], r, n_surrogates=2000, rng=rng)
    rate05 = float(np.mean(ps < 0.05))
    rate01 = float(np.mean(ps < 0.01))
    print(f"phi={phi} n={n} rate<0.05={rate05:.4f} rate<0.01={rate01:.4f} median_p={np.median(ps):.3f}")
    assert 0.025 <= rate05 <= 0.078, rate05
    assert rate01 <= 0.024, rate01
    # Not trivially conservative either: p should spread over (0, 1).
    assert 0.35 <= float(np.median(ps)) <= 0.65


def test_full_pipeline_null_family_rarely_emits_any_edge():
    """19 independent AR(1) channels -> m=171 pairs, 12 seeds. Under the
    global null BH controls the chance of *any* edge at alpha=0.05, and the
    within-pair Bonferroni makes it more conservative still. Allow at most
    2 of 12 seeds with any edge (P(>=3 | 0.05) ~ 2%)."""
    seeds_with_edges = 0
    total_edges = 0
    for seed in range(12):
        rng = np.random.default_rng(100 + seed)
        series = _ar1_batch(rng, 19, 400, 0.9)
        channels = {f"noise:c{i:02d}": _buckets(s) for i, s in enumerate(series)}
        results = cge.compute_pairwise_results(channels, seed=seed, n_surrogates=1000)
        assert len(results) == 171
        found = len(_edges(results))
        total_edges += found
        seeds_with_edges += int(found > 0)
    print(f"null family m=171 x 12 seeds: seeds_with_edges={seeds_with_edges} total_edges={total_edges}")
    assert seeds_with_edges <= 2, (seeds_with_edges, total_edges)

