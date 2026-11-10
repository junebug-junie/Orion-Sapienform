"""Gate tests for the causal-geometry p-value (2026-11-10 floor-cliff fix).

1. The vectorized time-shift surrogates equal an explicit per-shift Pearson.
2. The cliff: with the old circular add-one rank p, a strong real lagged link
   among m=171 pairs gets 0 edges; with the new p it survives BH, at n=600
   and at the live n=2016.
3. Calibration across null shapes -- AR(1) up to phi=0.98, bursty spikes,
   daily cycles, slow on/off regimes: the raw p is not below 0.05 / 0.01 /
   1e-3 more often than nominal, and a full m=171 null family almost never
   emits an edge. Each shape here broke a method that was tried and dropped:
   a normal fit (bursty: false edges in every seed; daily: no power), a
   circular null with a guard (daily: 357 false edges over 5 seeds), and the
   circular null without one (real links lost to their own near-0 shifts).
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


def _bursty_batch(rng: np.random.Generator, k: int, n: int) -> np.ndarray:
    """Event-driven channel: 8 bursts of 12 buckets with exponential heights."""
    out = 0.05 * rng.standard_normal((k, n))
    for row in out:
        for start in rng.choice(n, 8, replace=False):
            row[start : start + 12] += rng.exponential(5.0)
    return out


def _daily_batch(rng: np.random.Generator, k: int, n: int) -> np.ndarray:
    """Daily cycle (288 five-minute buckets) with a random phase, plus noise."""
    t = np.arange(n)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(k, 1))
    return np.sin(2.0 * np.pi * t / 288.0 + phases) + 0.3 * rng.standard_normal((k, n))


def _regime_batch(rng: np.random.Generator, k: int, n: int) -> np.ndarray:
    """On/off channel that flips state rarely (~6 flips per week)."""
    flips = rng.random((k, n)) < 0.003
    return (np.cumsum(flips, axis=1) % 2).astype(float) + 0.05 * rng.standard_normal((k, n))


def _ar1_gen(phi: float):
    return lambda rng, k, n: _ar1_batch(rng, k, n, phi)


NULL_SHAPES = {
    "ar1_0.5": _ar1_gen(0.5),
    "ar1_0.9": _ar1_gen(0.9),
    "ar1_0.98": _ar1_gen(0.98),
    "bursty": _bursty_batch,
    "daily": _daily_batch,
    "regime": _regime_batch,
}


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
    """The pre-2026-11-10 p, kept here only to reproduce the cliff: circular
    shifts drawn with replacement, add-one rank p, floor 1/(n_surrogates+1)."""
    n = len(ys)
    xz = (xs - xs.mean()) / xs.std()
    yz = (ys - ys.mean()) / ys.std()
    cc = np.fft.irfft(np.fft.rfft(xz) * np.conj(np.fft.rfft(yz)), n) / n
    rs = cc[rng.integers(1, n, size=n_surrogates)]
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


# --- 1. Surrogates + p-value mechanics ---------------------------------------


def _guard(n):
    return max(cge.GUARD_MIN, int(n * cge.GUARD_FRACTION))


def test_time_shift_surrogates_equal_explicit_overlap_pearson():
    rng = np.random.default_rng(3)
    n = 300
    x = rng.standard_normal(n)
    y = 0.4 * x + rng.standard_normal(n)
    rs = cge.time_shift_surrogate_rs(x, y)
    shifts = range(_guard(n), int(n * cge.MAX_SHIFT_FRACTION) + 1)
    expected = [cge._pearson(x[s:], y[: n - s]) for s in shifts]
    expected += [cge._pearson(x[: n - s], y[s:]) for s in shifts]
    np.testing.assert_allclose(rs, np.array(expected), atol=1e-12)
    # More shifts than the cap: a subset, drawn without replacement.
    capped = cge.time_shift_surrogate_rs(x, y, n_surrogates=50, rng=np.random.default_rng(1))
    assert capped.size == 50 and np.all(np.isin(capped, rs))
    assert np.unique(capped).size == 50


def test_degenerate_inputs():
    x = np.arange(50, dtype=float)
    flat = np.full(50, 3.0)
    assert cge.time_shift_surrogate_rs(x, flat).size == 0
    assert cge.time_shift_pvalue(x, flat, 0.0) == 1.0
    assert cge.time_shift_pvalue(x[:3], x[:3], 1.0) == 1.0  # too short for any shift
    with pytest.raises(ValueError):
        cge.time_shift_surrogate_rs(x, x[:-1])
    with pytest.raises(ValueError):
        cge.time_shift_pvalue(x, x[:-1], 0.0)


def test_pvalue_is_continuous_past_the_rank_floor_and_never_zero():
    """Short-memory series (n_eff >= N_EFF_MIN): a strong link gets a
    tiny-but-positive p far below the rank floor 1/(N+1), and p falls
    strictly as |r| grows, across the rank/tail boundary."""
    rng = np.random.default_rng(0)
    x, noise = _ar1_batch(rng, 2, 600, 0.3)
    assert cge.effective_sample_size(x, noise) >= cge.N_EFF_MIN
    n_surr = cge.time_shift_surrogate_rs(x, noise).size
    ps = []
    for weight in (0.1, 0.2, 0.3, 0.5, 1.0, 3.0):
        y = weight * x + noise
        ps.append(cge.time_shift_pvalue(x, y, cge._pearson(x, y)))
    assert all(a > b for a, b in zip(ps, ps[1:])), ps
    assert cge.MIN_REPORTED_PVALUE <= ps[-1] < 1.0 / (n_surr + 1) / 100, ps


def test_low_effective_sample_size_never_extrapolates():
    """Two trending series (n_eff tiny): even a big |r| gets no p finer than
    1/(n_eff+1), so a pile of such pairs cannot tie at a floor and pass BH
    together (the old cliff in miniature)."""
    rng = np.random.default_rng(4)
    x = np.cumsum(rng.standard_normal(2000))
    y = np.cumsum(rng.standard_normal(2000))
    n_eff = cge.effective_sample_size(x, y)
    assert n_eff < cge.N_EFF_MIN
    p = cge.time_shift_pvalue(x, y, cge._pearson(x, y))
    assert p >= 1.0 / (n_eff + 1.0)
    assert p * 2 * len(cge.LAG_GRID_SECONDS) > cge.DEFAULT_ALPHA


def test_tail_fit_is_heavier_for_a_heavy_tailed_null():
    """Same observed value past every surrogate: a heavy-tailed null (bursty
    channels) gives a larger p than a light-tailed one, and a bounded null is
    not extrapolated to a near-zero p (xi clamped at 0)."""
    rng = np.random.default_rng(5)
    light = np.abs(rng.standard_normal(2000))
    heavy_raw = np.abs(rng.standard_t(2, size=2000))
    heavy = heavy_raw * (np.median(light) / np.median(heavy_raw))
    bounded = np.abs(np.sin(rng.uniform(0, 2 * np.pi, 2000)))
    obs = max(light.max(), heavy.max()) * 1.2
    assert cge.surrogate_tail_pvalue(heavy, obs) > cge.surrogate_tail_pvalue(light, obs)
    assert cge.surrogate_tail_pvalue(bounded, 1.05) > 1e-6
    # Body: plain add-one rank p; no tail fit when it is switched off.
    assert cge.surrogate_tail_pvalue(light, float(np.sort(light)[-50])) == pytest.approx(51 / 2001)
    assert cge.surrogate_tail_pvalue(light, obs, allow_tail_fit=False) == pytest.approx(1 / 2001)


# --- 2. The cliff -------------------------------------------------------------


@pytest.mark.parametrize("n", [600, 2016])
def test_cliff_legacy_rank_p_finds_nothing_new_p_finds_the_real_link(monkeypatch, n):
    channels = _cliff_fixture(n=n)
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
    print(f"cliff fixture n={n}: real r={winner.r:.3f} p={by_pair[real].p_value:.3g} edges={len(new_edges)}")
    assert real in new_edges, f"real link p={by_pair[real].p_value} did not survive BH at m=171"
    assert len(new_edges - {real}) <= 1, f"too many noise edges: {sorted(map(sorted, new_edges - {real}))}"

    # Reproduce the live bug: the rank p cannot go below 1/2001, so after the
    # 10-way Bonferroni the real link sits at 0.005 > 0.05/171 and BH passes nothing.
    monkeypatch.setattr(cge, "time_shift_pvalue", _legacy_rank_pvalue)
    old_results = cge.compute_pairwise_results(channels, seed=1337)
    old_by_pair = {frozenset((r.channel_a, r.channel_b)): r for r in old_results}
    assert old_by_pair[real].p_value == pytest.approx(10 / 2001)
    assert _edges(old_results) == set(), "the legacy rank p should reproduce the 0-edge cliff"


# --- 3. Calibration -----------------------------------------------------------


@pytest.mark.parametrize("shape", list(NULL_SHAPES))
def test_raw_pvalue_false_positive_rate_is_at_most_nominal_across_null_shapes(shape):
    """1000 independent pairs per null shape, n=2016 (7 days of 5-min
    buckets). Binomial sd at 1000 draws: ~0.007 at 5%, ~0.003 at 1%, ~0.001
    at 0.1%; upper bounds are ~3 sd over nominal. The 1e-3 bound is the one
    BH leans on. The test is allowed to be conservative (the guarded overlap
    null is wider than the true one for short-memory series, and daily-cycle
    pairs have n_eff ~7, so their p never goes below 1/8), but the
    short-memory and bursty shapes must keep some size, or the method would
    be dead rather than calibrated. (The dropped normal fit ran 1.8% below
    5e-4 on `bursty`.)"""
    rng = np.random.default_rng(sum(map(ord, shape)))
    k, n = 1000, 2016
    gen = NULL_SHAPES[shape]
    xs, ys = gen(rng, k, n), gen(rng, k, n)
    ps = np.array([cge.time_shift_pvalue(x, y, cge._pearson(x, y)) for x, y in zip(xs, ys)])
    rate05 = float(np.mean(ps < 0.05))
    rate01 = float(np.mean(ps < 0.01))
    rate_tail = float(np.mean(ps < 1e-3))
    print(f"{shape} rate<0.05={rate05:.4f} rate<0.01={rate01:.4f} rate<1e-3={rate_tail:.4f}")
    assert rate05 <= 0.072, rate05
    assert rate01 <= 0.02, rate01
    assert rate_tail <= 0.004, rate_tail
    if shape in ("ar1_0.5", "ar1_0.9", "bursty"):
        assert rate05 >= 0.01, f"{shape}: test has lost its size ({rate05})"


@pytest.mark.parametrize("shape", list(NULL_SHAPES))
def test_full_pipeline_null_family_rarely_emits_any_edge(shape):
    """19 independent channels -> m=171 pairs, 6 seeds, through the real
    lag/direction search + Bonferroni + BH. Under the global null BH bounds
    the chance of *any* edge by alpha=0.05; allow at most 1 of 6 seeds
    (P(>=2 | 0.05) ~ 3%). The dropped normal fit gave 34 false edges over 6
    seeds on `bursty`; the guarded circular null gave 357 over 5 on `daily`."""
    seeds_with_edges = 0
    total_edges = 0
    for seed in range(6):
        rng = np.random.default_rng(100 + seed)
        series = NULL_SHAPES[shape](rng, 19, 1000)
        channels = {f"noise:c{i:02d}": _buckets(s) for i, s in enumerate(series)}
        results = cge.compute_pairwise_results(channels, seed=seed)
        assert len(results) == 171
        found = len(_edges(results))
        total_edges += found
        seeds_with_edges += int(found > 0)
    print(f"null family {shape} m=171 x 6 seeds: seeds_with_edges={seeds_with_edges} total_edges={total_edges}")
    assert seeds_with_edges <= 1, (seeds_with_edges, total_edges)
