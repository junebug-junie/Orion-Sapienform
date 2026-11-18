"""Field demon: how far ahead Orion's own state can be forecast, and what each cut costs.

Spec: docs/superpowers/specs/2026-09-30-field-demon-forecaster-design.md (rev 2). Pure numpy, no
I/O. Input is a (T, n) array of organ surprise residuals in 5-minute buckets (#3290 builds them),
one column per organ.

One model, two readouts:

* The transition model is a Gaussian VAR(1): x_{t+1} = mean + A (x_t - mean) + noise.
  ``integration_rung`` (#3296) reads it through ``lag_k_model`` to compute Φ over the partition
  lattice.
* The demon reads it forward. ``rolling_errors`` fits on a trailing window, forecasts the next
  hour of origins at k = 1..K steps, and keeps every squared error. ``skill_curve``,
  ``horizon_steps`` and ``bootstrap_horizon`` turn those errors into a horizon with its spread.
  ``cut_cost_curve`` compares the whole model against a cut model, in which every part forecasts
  only from its own past.

Baselines: ``clock`` forecasts a residual of 0 (the time-of-day profile). ``persistence``
forecasts that the residual stays where it is. A model's skill at k is measured against the better
(lower-MSE) of the two at that k.

No alpha line anywhere (Juniper's rule): every readout is a point estimate with its spread.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

BUCKET_MINUTES = 5
DEFAULT_WINDOW = 288  # 24 h of 5-minute buckets
DEFAULT_STEP = 12  # refit every hour
DEFAULT_HORIZON = 12  # k = 1..12, 5 to 60 minutes
DEFAULT_NULL_GUARD = 36  # 3 h: every pair of organs is shifted apart by more than any scored lag

# Hygiene gate: a staleness-decay loop multiplies an unrefreshed value by a fixed factor every tick
# (2026-07-26 substrate.route: x0.92 per tick for 48 h). Its successive ratio is constant, and it
# reads as calm.
DECAY_MIN_RUN = 24  # 2 h of nonzero values
DECAY_SHRINK_CV_MAX = 0.05  # how constant the per-step shrink (1 - ratio) must be to read as decay
DECAY_SHRINK_MIN = 1e-4  # a per-step shrink smaller than this is not decay
NEAR_CONSTANT_REL_STD = 1e-8  # a training-window std below this (relative to |mean|) is float noise
DEFAULT_NULL_SPAN_FRACTION = 0.25  # the null may drop at most this share of rows to the shifts

Partition = Sequence[Sequence[int]]


@dataclass(frozen=True)
class Var1:
    """x_{t+1} = mean + A (x_t - mean) + e, with Cov(e) = sigma."""

    mean: np.ndarray
    A: np.ndarray
    sigma: np.ndarray

    def forecast(self, x0: np.ndarray, k: int) -> np.ndarray:
        """Point forecast k steps ahead from state(s) x0, shape (n,) or (m, n)."""
        dev = np.asarray(x0, dtype=float) - self.mean
        Ak = np.linalg.matrix_power(self.A, k)
        return self.mean + dev @ Ak.T


def fit_var1(x: np.ndarray) -> Var1:
    """Least-squares VAR(1) on a (T, n) array (T >= n + 2)."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[0] < x.shape[1] + 2:
        raise ValueError(f"fit_var1: need a (T, n) array with T >= n + 2, got {x.shape}")
    mean = x.mean(axis=0)
    past = x[:-1] - mean
    nxt = x[1:] - mean
    # Solve past @ A.T = nxt. lstsq handles a near-singular past without a crash.
    At, *_ = np.linalg.lstsq(past, nxt, rcond=None)
    A = At.T
    resid = nxt - past @ At
    sigma = resid.T @ resid / max(len(resid) - x.shape[1], 1)
    return Var1(mean=mean, A=A, sigma=sigma)


def fit_cut(x: np.ndarray, partition: Partition) -> Var1:
    """VAR(1) in which each part forecasts only from its own past: A is block-diagonal.

    ``sigma`` is the full covariance of the cut model's residuals, because the parts' noise may
    still be correlated at the same instant.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[1]
    _check_partition(partition, n)
    mean = x.mean(axis=0)
    A = np.zeros((n, n))
    for part in partition:
        idx = np.asarray(part, dtype=int)
        A[np.ix_(idx, idx)] = fit_var1(x[:, idx]).A
    past = x[:-1] - mean
    resid = (x[1:] - mean) - past @ A.T
    sigma = resid.T @ resid / max(len(resid) - n, 1)
    return Var1(mean=mean, A=A, sigma=sigma)


def full_cut(n: int) -> list[list[int]]:
    """Every organ alone: the bottom element of the partition lattice."""
    return [[i] for i in range(n)]


def lag_k_model(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Direct lag-k model from empirical covariances: (A_k, sigma_k, gamma_0).

    A_k = Γ_k Γ_0⁻¹ and Σ_k = Γ_0 - A_k Γ_0 A_kᵀ, where Γ_k = Cov(x_{t+k}, x_t). This does not
    assume VAR(1) holds at lag k. It is the input #3296's Φ_G / Φ* take at each lag.
    """
    x = np.asarray(x, dtype=float)
    if k < 1 or x.shape[0] <= k + x.shape[1]:
        raise ValueError(f"lag_k_model: k={k} too large for {x.shape[0]} rows")
    xc = x - x.mean(axis=0)
    # Both divided by T (the biased, block-Toeplitz estimator). Dividing Γ_k by T-k instead breaks
    # positive-definiteness of Σ_k: review 2026-11-18 found 188/1200 slow-organ windows (8 organs,
    # A = 0.995 I, 288 rows) with a non-PSD Σ_k, so Φ would read NaN.
    gamma0 = xc.T @ xc / len(xc)
    gammak = xc[k:].T @ xc[:-k] / len(xc)
    Ak = gammak @ np.linalg.pinv(gamma0)
    sigmak = gamma0 - Ak @ gamma0 @ Ak.T
    sigmak = (sigmak + sigmak.T) / 2
    return Ak, sigmak, gamma0


def predictive_information(gamma0: np.ndarray, sigma_k: np.ndarray) -> float:
    """I(x_{t+k}; x_t) in nats for a Gaussian: ½ log det Γ_0 / det Σ_k.

    NaN unless both matrices are positive definite. A positive slogdet sign alone would pass a
    matrix with an even number of negative eigenvalues.
    """
    if np.linalg.eigvalsh(gamma0).min() <= 0 or np.linalg.eigvalsh(sigma_k).min() <= 0:
        return float("nan")
    return 0.5 * (np.linalg.slogdet(gamma0)[1] - np.linalg.slogdet(sigma_k)[1])


def iterated_predictive_information(model: Var1, gamma0: np.ndarray, k: int) -> float:
    """The same quantity implied by iterating the lag-1 model k times.

    Comparing this with the direct lag-k value is the misfit check. If the direct value carries
    much more, VAR(1) is missing slow structure.
    """
    Ak = np.linalg.matrix_power(model.A, k)
    sigma_k = gamma0 - Ak @ gamma0 @ Ak.T
    return predictive_information(gamma0, (sigma_k + sigma_k.T) / 2)


@dataclass(frozen=True)
class RollingErrors:
    """Squared forecast errors, shape (origins, K), summed over organs in standardized units."""

    whole: np.ndarray
    clock: np.ndarray
    persistence: np.ndarray
    cuts: Mapping[str, np.ndarray]
    block: int  # origins per refit (bootstrap block length)
    skipped: tuple[tuple[int, str], ...] = ()  # (refit end row, reason), never scored

    @property
    def baseline(self) -> np.ndarray:
        """Per-origin errors of the baseline that wins (lower total MSE) at each k."""
        use_clock = self.clock.sum(axis=0) <= self.persistence.sum(axis=0)
        return np.where(use_clock[None, :], self.clock, self.persistence)


def rolling_errors(
    x: np.ndarray,
    *,
    partitions: Mapping[str, Partition] | None = None,
    window: int = DEFAULT_WINDOW,
    step: int = DEFAULT_STEP,
    horizon: int = DEFAULT_HORIZON,
) -> RollingErrors:
    """Fit on x[e-window:e], forecast from origins e..e+step-1 at k = 1..horizon.

    The origin state x[t] is observed at forecast time but is never in the fit. Each organ is
    standardized with the training window's mean and std, so organs count equally in the summed
    error. Origins whose target falls past the end of the series are dropped.

    A refit is skipped, and recorded in ``skipped`` with a reason, when any organ is missing
    anywhere in its training or scoring rows, or is (near-)constant over its training window.
    Standardizing a column whose std is float noise (a repeated 0.3 gives ~5e-17) would blow its
    errors up to ~1e29 and swamp every other organ.
    """
    x = np.asarray(x, dtype=float)
    T, n = x.shape
    partitions = dict(partitions or {})
    for part in partitions.values():
        _check_partition(part, n)
    whole, clock, persist = [], [], []
    cuts: dict[str, list[np.ndarray]] = {name: [] for name in partitions}
    skipped: list[tuple[int, str]] = []
    for e in range(window, T - horizon, step):
        train = x[e - window:e]
        origins = np.arange(e, min(e + step, T - horizon))
        if not np.all(np.isfinite(x[e - window:origins[-1] + horizon + 1])):
            skipped.append((e, "missing"))
            continue
        mu = train.mean(axis=0)
        sd = train.std(axis=0)
        flat = np.flatnonzero(sd < NEAR_CONSTANT_REL_STD * np.maximum(1.0, np.abs(mu)))
        if flat.size:
            skipped.append((e, f"constant:{int(flat[0])}"))
            continue
        z_train = (train - mu) / sd
        z = (x - mu) / sd
        z0 = z[origins]
        targets = np.stack([z[origins + k] for k in range(1, horizon + 1)], axis=1)  # (m, K, n)
        model = fit_var1(z_train)
        cut_models = {name: fit_cut(z_train, part) for name, part in partitions.items()}
        zero = -mu / sd  # a raw residual of 0, in standardized units

        def _err(fc: np.ndarray) -> np.ndarray:
            return ((targets - fc) ** 2).sum(axis=2)

        whole.append(_err(_rollout(model, z0, horizon)))
        clock.append(_err(np.broadcast_to(zero, targets.shape)))
        persist.append(_err(np.broadcast_to(z0[:, None, :], targets.shape)))
        for name, cm in cut_models.items():
            cuts[name].append(_err(_rollout(cm, z0, horizon)))
    if not whole:
        why = f"every refit was skipped: {skipped[:3]}" if skipped else "too short"
        raise ValueError(f"rolling_errors: {T} rows, window={window}, horizon={horizon}: {why}")
    return RollingErrors(
        whole=np.concatenate(whole),
        clock=np.concatenate(clock),
        persistence=np.concatenate(persist),
        cuts={name: np.concatenate(v) for name, v in cuts.items()},
        block=step,
        skipped=tuple(skipped),
    )


def _rollout(model: Var1, z0: np.ndarray, horizon: int) -> np.ndarray:
    return np.stack([model.forecast(z0, k) for k in range(1, horizon + 1)], axis=1)


def skill_curve(model_err: np.ndarray, baseline_err: np.ndarray) -> np.ndarray:
    """S(k) = 1 - MSE_model(k) / MSE_baseline(k). 0 = no better than the baseline."""
    base = baseline_err.sum(axis=0)
    return 1.0 - model_err.sum(axis=0) / np.where(base > 0, base, np.nan)


def horizon_steps(skill: np.ndarray) -> int:
    """Steps ahead before skill first drops to 0 or below (0 = no horizon at all)."""
    bad = np.flatnonzero(~(np.asarray(skill) > 0))
    return int(bad[0]) if bad.size else int(len(skill))


def cut_cost_curve(errors: RollingErrors, cut: str) -> np.ndarray:
    """C(k) = S_whole(k) - S_cut(k): the foresight the cut destroys at each step."""
    base = errors.baseline
    return skill_curve(errors.whole, base) - skill_curve(errors.cuts[cut], base)


@dataclass(frozen=True)
class HorizonEstimate:
    steps: int
    median_steps: float
    p10_steps: float
    p90_steps: float
    skill: np.ndarray
    p_positive: np.ndarray  # P(S(k) > 0) per k, from the bootstrap

    @property
    def minutes(self) -> int:
        return self.steps * BUCKET_MINUTES


def bootstrap_horizon(
    errors: RollingErrors,
    *,
    n_boot: int = 1000,
    rng: np.random.Generator | None = None,
) -> HorizonEstimate:
    """Horizon with a moving-block bootstrap over origins (block = one refit's origins)."""
    rng = rng if rng is not None else np.random.default_rng()
    stacked = np.stack([errors.whole, errors.clock, errors.persistence])  # (3, m, K)
    m = stacked.shape[1]
    block = max(1, min(errors.block, m))
    n_blocks = int(np.ceil(m / block))
    steps = np.empty(n_boot)
    positive = np.zeros(stacked.shape[2])
    for b in range(n_boot):
        starts = rng.integers(0, m - block + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:m]
        w, c, p = stacked[:, idx]
        use_clock = c.sum(axis=0) <= p.sum(axis=0)
        s = skill_curve(w, np.where(use_clock[None, :], c, p))
        steps[b] = horizon_steps(s)
        positive += s > 0
    point = skill_curve(errors.whole, errors.baseline)
    return HorizonEstimate(
        steps=horizon_steps(point),
        median_steps=float(np.median(steps)),
        p10_steps=float(np.percentile(steps, 10)),
        p90_steps=float(np.percentile(steps, 90)),
        skill=point,
        p_positive=positive / n_boot,
    )


def time_shift_null(
    x: np.ndarray,
    *,
    guard: int = DEFAULT_NULL_GUARD,
    span_fraction: float = DEFAULT_NULL_SPAN_FRACTION,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Shift each organ by its own random offset and keep the overlap. No wrap-around.

    Every pair of organs ends up at least ``guard`` steps apart, which breaks coupling at every
    scored lag and keeps each organ's own rhythm and memory. Offsets are drawn at random with a
    fixed total span of ``span_fraction * T``, so draws differ from each other and every draw keeps
    the same number of rows. Same reasoning as causal geometry's overlap null
    (``causal_geometry_engine.time_shift_surrogate_rs``): a wrap-around seam distorts the null.
    """
    x = np.asarray(x, dtype=float)
    T, n = x.shape
    rng = rng if rng is not None else np.random.default_rng()
    span = int(span_fraction * T)
    if n < 2 or span < (n - 1) * guard or T - span < 2:
        raise ValueError(
            f"time_shift_null: {T} rows cannot hold {n} organs {guard} apart within "
            f"{span_fraction:.0%} of the series"
        )
    extra = span - (n - 1) * guard
    gaps = guard + rng.multinomial(extra, rng.dirichlet(np.ones(n - 1)))
    offsets = rng.permutation(np.concatenate([[0], np.cumsum(gaps)]))
    length = T - span
    return np.stack([x[o:o + length, i] for i, o in enumerate(offsets)], axis=1)


@dataclass(frozen=True)
class CutCost:
    """Foresight a cut destroys, against the time-shift null (no alpha line)."""

    cost: np.ndarray  # C(k), observed
    null_mean: np.ndarray
    null_sd: np.ndarray
    p_real: np.ndarray  # share of null draws whose C(k) is below the observed C(k)
    horizon_cost_steps: int  # H_whole - H_cut, observed

    @property
    def excess(self) -> np.ndarray:
        return self.cost - self.null_mean

    @property
    def horizon_cost_minutes(self) -> int:
        return self.horizon_cost_steps * BUCKET_MINUTES


def cut_cost_vs_null(
    x: np.ndarray,
    partition: Partition,
    *,
    n_null: int = 20,
    rng: np.random.Generator | None = None,
    window: int = DEFAULT_WINDOW,
    step: int = DEFAULT_STEP,
    horizon: int = DEFAULT_HORIZON,
    guard: int = DEFAULT_NULL_GUARD,
) -> CutCost:
    """Observed cut cost over the whole series, and the same cost on ``n_null`` shifted copies.

    Each null copy keeps at least 75% of the rows (``DEFAULT_NULL_SPAN_FRACTION``), so the null
    is scored on most of the same hours as the observed cost.
    """
    rng = rng if rng is not None else np.random.default_rng()
    kw = dict(window=window, step=step, horizon=horizon)
    errs = rolling_errors(x, partitions={"cut": partition}, **kw)
    observed = cut_cost_curve(errs, "cut")
    base = errs.baseline
    h_whole = horizon_steps(skill_curve(errs.whole, base))
    h_cut = horizon_steps(skill_curve(errs.cuts["cut"], base))
    null = np.stack([
        cut_cost_curve(
            rolling_errors(time_shift_null(x, guard=guard, rng=rng), partitions={"cut": partition}, **kw),
            "cut",
        )
        for _ in range(n_null)
    ])
    return CutCost(
        cost=observed,
        null_mean=null.mean(axis=0),
        null_sd=null.std(axis=0, ddof=1) if n_null > 1 else np.zeros_like(observed),
        p_real=(null < observed[None, :]).mean(axis=0),
        horizon_cost_steps=h_whole - h_cut,
    )


def refusal_reason(x: np.ndarray, names: Sequence[str]) -> str | None:
    """Hygiene gate: why this window must not be scored, or None when it is clean.

    Takes the **raw node values**, before the time-of-day residual: subtracting a profile destroys
    the constant ratio a decay loop leaves behind. Refuses a missing value, a constant input, or a
    geometric-decay input, and names the input.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(names):
        raise ValueError(f"refusal_reason: {x.shape} does not match {len(names)} names")
    for i, name in enumerate(names):
        col = x[:, i]
        if not np.all(np.isfinite(col)):
            return f"missing:{name}"
        if np.ptp(col) == 0:
            return f"constant:{name}"
        if _has_decay_run(col):
            return f"decaying:{name}"
    return None


def _has_decay_run(col: np.ndarray) -> bool:
    """True when some 2 h run shrinks toward 0 by the same fraction every step.

    A decay loop gives x_{t+1} = r x_t, so the per-step shrink 1 - r is constant and positive.
    A relaxation toward a nonzero level has a shrink that itself fades (not constant). Growth has a
    negative shrink, and a flat stretch has none, so neither is flagged.
    """
    n_ratios = DECAY_MIN_RUN - 1
    nonzero = col != 0
    run_start = None
    for t in range(len(col) + 1):
        if t < len(col) and nonzero[t]:
            if run_start is None:
                run_start = t
            continue
        if run_start is not None and t - run_start >= DECAY_MIN_RUN:
            seg = col[run_start:t]
            shrink = 1.0 - seg[1:] / seg[:-1]
            windows = np.lib.stride_tricks.sliding_window_view(shrink, n_ratios)
            mean = windows.mean(axis=1)
            sd = windows.std(axis=1)
            if np.any((mean > DECAY_SHRINK_MIN) & (sd < DECAY_SHRINK_CV_MAX * mean)):
                return True
        run_start = None
    return False


def _check_partition(partition: Partition, n: int) -> None:
    flat = sorted(int(i) for part in partition for i in part)
    if flat != list(range(n)) or any(len(part) == 0 for part in partition):
        raise ValueError(f"partition {partition!r} does not cover organs 0..{n - 1} exactly once")
