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
DECAY_RATIO_STD_MAX = 0.01
DECAY_MIN_RUN = 24  # 2 h of nonzero values

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
    m = len(xc) - k
    gamma0 = xc.T @ xc / len(xc)
    gammak = xc[k:].T @ xc[:-k] / m
    Ak = gammak @ np.linalg.pinv(gamma0)
    sigmak = gamma0 - Ak @ gamma0 @ Ak.T
    sigmak = (sigmak + sigmak.T) / 2
    return Ak, sigmak, gamma0


def predictive_information(gamma0: np.ndarray, sigma_k: np.ndarray) -> float:
    """I(x_{t+k}; x_t) in nats for a Gaussian: ½ log det Γ_0 / det Σ_k."""
    s0, ld0 = np.linalg.slogdet(gamma0)
    sk, ldk = np.linalg.slogdet(sigma_k)
    if s0 <= 0 or sk <= 0:
        return float("nan")
    return 0.5 * (ld0 - ldk)


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
    """
    x = np.asarray(x, dtype=float)
    T, n = x.shape
    partitions = dict(partitions or {})
    for part in partitions.values():
        _check_partition(part, n)
    whole, clock, persist = [], [], []
    cuts: dict[str, list[np.ndarray]] = {name: [] for name in partitions}
    for e in range(window, T - horizon, step):
        train = x[e - window:e]
        mu = train.mean(axis=0)
        sd = train.std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        z_train = (train - mu) / sd
        origins = np.arange(e, min(e + step, T - horizon))
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
        raise ValueError(f"rolling_errors: {T} rows is too short for window={window}, horizon={horizon}")
    return RollingErrors(
        whole=np.concatenate(whole),
        clock=np.concatenate(clock),
        persistence=np.concatenate(persist),
        cuts={name: np.concatenate(v) for name, v in cuts.items()},
        block=step,
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
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Shift each organ by its own offset and keep the overlap. No wrap-around.

    Offsets are a random order of 0, guard, 2*guard, ..., so every pair of organs is at least
    ``guard`` steps apart. That breaks coupling at every scored lag and keeps each organ's own
    rhythm and memory. Same reasoning as causal geometry's overlap null
    (``causal_geometry_engine.time_shift_surrogate_rs``): a wrap-around seam distorts the null.
    """
    x = np.asarray(x, dtype=float)
    T, n = x.shape
    rng = rng if rng is not None else np.random.default_rng()
    offsets = rng.permutation(n) * guard
    span = int(offsets.max())
    if T - span < 2:
        raise ValueError(f"time_shift_null: {T} rows cannot hold {n} organs {guard} apart")
    length = T - span
    return np.stack([x[o:o + length, i] for i, o in enumerate(offsets)], axis=1)


def refusal_reason(x: np.ndarray, names: Sequence[str]) -> str | None:
    """Hygiene gate: why this window must not be scored, or None when it is clean.

    Refuses a missing value, a constant input, or a geometric-decay input, and names the input.
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
    nonzero = col != 0
    run_start = None
    for t in range(len(col) + 1):
        if t < len(col) and nonzero[t]:
            if run_start is None:
                run_start = t
            continue
        if run_start is not None and t - run_start >= DECAY_MIN_RUN:
            seg = col[run_start:t]
            ratios = seg[1:] / seg[:-1]
            for s in range(0, len(ratios) - DECAY_MIN_RUN + 2):
                win = ratios[s:s + DECAY_MIN_RUN - 1]
                if np.std(win) < DECAY_RATIO_STD_MAX and abs(np.mean(win) - 1.0) > DECAY_RATIO_STD_MAX:
                    return True
        run_start = None
    return False


def _check_partition(partition: Partition, n: int) -> None:
    flat = sorted(int(i) for part in partition for i in part)
    if flat != list(range(n)) or any(len(part) == 0 for part in partition):
        raise ValueError(f"partition {partition!r} does not cover organs 0..{n - 1} exactly once")
