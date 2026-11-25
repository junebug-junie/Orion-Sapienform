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

Baselines: ``clock`` forecasts each organ's mean over the training window: "the usual, at
today's level". (Until 2026-11-24 it forecast a raw residual of 0, the 14-day time-of-day profile.
The model's own long-range forecast converges to the training mean, so whenever today's level sat
off the 14-day profile, the model beat that baseline at every horizon without any dynamics: the
first live replay read a flat ~6% skill out to 2 h.) ``persistence``
forecasts that the residual stays where it is. A model's skill at k is measured against the better
(lower-MSE) of the two at that k.

No alpha line anywhere (Juniper's rule): every readout is a point estimate with its spread.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

from orion.substrate import born_machine

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
    # Set only by ``network_rolling_errors``: the VAR(1) forecast scored the same way as the
    # network (log-loss, bits), and the lag of each column. Var1 errors leave both as None and
    # their column j is lag j + 1.
    linear: np.ndarray | None = None
    lags: tuple[int, ...] | None = None

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

        def _err(fc: np.ndarray) -> np.ndarray:
            return ((targets - fc) ** 2).sum(axis=2)

        whole.append(_err(_rollout(model, z0, horizon)))
        clock.append(_err(np.zeros_like(targets)))  # the training mean, standardized
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
    errors_fn: Callable[..., RollingErrors] | None = None,
) -> CutCost:
    """Observed cut cost over the whole series, and the same cost on ``n_null`` shifted copies.

    Each null copy keeps at least 75% of the rows (``DEFAULT_NULL_SPAN_FRACTION``), so the null
    is scored on most of the same hours as the observed cost. ``errors_fn`` defaults to the
    VAR(1) ``rolling_errors``; pass ``network_rolling_errors`` to score the trained network, which
    then refits its chain on every shifted copy.
    """
    rng = rng if rng is not None else np.random.default_rng()
    rolling_errors_fn = errors_fn if errors_fn is not None else rolling_errors
    kw = dict(window=window, step=step, horizon=horizon)
    errs = rolling_errors_fn(x, partitions={"cut": partition}, **kw)
    observed = cut_cost_curve(errs, "cut")
    base = errs.baseline
    h_whole = horizon_steps(skill_curve(errs.whole, base))
    h_cut = horizon_steps(skill_curve(errs.cuts["cut"], base))
    null = np.stack([
        cut_cost_curve(
            rolling_errors_fn(time_shift_null(x, guard=guard, rng=rng), partitions={"cut": partition}, **kw),
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


# --- Network forecaster (spec 2026-11-24 rev 3: the demon's model is a trained tensor network) ---
#
# Each organ is cut into 3 levels (low, usual, high) at the training window's terciles. The input
# is the same as for Var1: residuals against each organ's 14-day time-of-day profile
# (``whole_body_structure.tod_residuals``, which only looks at earlier days). So "usual" means
# "usual for this time of day", and the tercile edges never see a scored row.
#
# One Born machine per lag k is trained on (levels at t, levels at t+k), with each organ's now and
# later sites next to each other: o1(t), o1(t+k), o2(t), o2(t+k), ... The forecast for organ i is
# the exact conditional p(later_i | now_A), with A every organ (the whole model) or only the part
# that holds i (a cut).
#
# Score: log-loss in bits, summed over organs. This is the log-loss twin of the summed squared
# error Var1 uses. Scoring each organ's own forecast (not the joint "later" vector) keeps a cut
# from being charged for noise the parts share at the same instant: no part's past can carry that,
# with or without coupling.

N_LEVELS = 3
# Every forecast (network, clock, persistence, linear) is mixed with this much of the uniform
# 3-level guess, so a level a model rules out costs a finite amount. Same floor for all models.
PROB_FLOOR = 1e-3


def tercile_edges(z: np.ndarray) -> np.ndarray:
    """(n, 2) level edges: the 1/3 and 2/3 quantiles of each column. Training rows only."""
    return np.quantile(np.asarray(z, dtype=float), [1 / 3, 2 / 3], axis=0).T


def to_levels(z: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """0 = low, 1 = usual, 2 = high, per value."""
    z = np.asarray(z, dtype=float)
    return ((z > edges[:, 0]).astype(np.int64) + (z > edges[:, 1]).astype(np.int64))


def _floored(p: np.ndarray) -> np.ndarray:
    return (1.0 - PROB_FLOOR) * p + PROB_FLOOR / N_LEVELS


def _bits(probs: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """-log2 p(observed level), summed over organs. probs (m, n, 3), levels (m, n) -> (m,)."""
    p = np.take_along_axis(_floored(probs), levels[:, :, None], axis=2)[:, :, 0]
    return -np.log2(p).sum(axis=1)


_erf = np.vectorize(math.erf, otypes=[float])


def _normal_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + _erf(np.asarray(z, dtype=float) / np.sqrt(2.0)))


def gaussian_level_probs(mean: np.ndarray, var: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """A Gaussian forecast integrated into the 3 levels. mean (m, n), var (n,) -> (m, n, 3)."""
    sd = np.sqrt(np.maximum(np.asarray(var, dtype=float), 1e-12))
    c1 = _normal_cdf((edges[:, 0] - mean) / sd)
    c2 = _normal_cdf((edges[:, 1] - mean) / sd)
    return np.stack([c1, np.clip(c2 - c1, 0.0, None), 1.0 - c2], axis=2)


def _var1_forecast_var(model: Var1, k: int) -> np.ndarray:
    """Per-organ variance of the k-step VAR(1) forecast: diag(sum_j A^j Σ A^jᵀ), j < k."""
    n = model.A.shape[0]
    cov = np.zeros((n, n))
    Aj = np.eye(n)
    for _ in range(k):
        cov += Aj @ model.sigma @ Aj.T
        Aj = model.A @ Aj
    return np.diag(cov).copy()


def _pairs(levels: np.ndarray, k: int) -> np.ndarray:
    """(T-k, 2n) training samples, sites interleaved: o1(t), o1(t+k), o2(t), o2(t+k), ..."""
    now, later = levels[:-k], levels[k:]
    out = np.empty((len(now), 2 * levels.shape[1]), dtype=np.int64)
    out[:, 0::2] = now
    out[:, 1::2] = later
    return out


@dataclass
class NetworkForecaster:
    """Trained chains (one per lag) plus the three baselines, all on one training window."""

    lags: tuple[int, ...]
    edges: np.ndarray  # (n, 2), on the standardized training window
    machines: dict[int, born_machine.BornMachine]
    selections: dict[int, born_machine.Selection]
    clock: np.ndarray  # (n, 3) level shares over the training window
    stay: dict[int, np.ndarray]  # lag -> (n,) share of training pairs whose level did not change
    var1: Var1  # fitted on the level-coded training window
    var1_continuous: Var1  # fitted on the standardized values themselves
    level_means: np.ndarray  # (n, 3): mean standardized value of each level, training window
    _tables: dict = field(default_factory=dict, repr=False)  # (k, part, organ) -> table, on use

    @property
    def n(self) -> int:
        return int(self.edges.shape[0])

    def _table(self, k: int, part: tuple[int, ...], organ: int) -> np.ndarray:
        key = (k, part, organ)
        if key not in self._tables:
            self._tables[key] = born_machine.conditional_table(
                self.machines[k], [2 * a for a in part], [2 * organ + 1]
            )
        return self._tables[key]

    def network_probs(self, now: np.ndarray, k: int, partition: Partition | None = None) -> np.ndarray:
        """p(later_i | now_A) for every organ i, with A the part that holds i. (m, n) -> (m, n, 3).

        Every factor comes from the one trained chain for lag k. A cut does not refit anything.
        """
        now = np.asarray(now, dtype=np.int64)
        partition = partition if partition is not None else [list(range(self.n))]
        _check_partition(partition, self.n)
        out = np.empty((len(now), self.n, N_LEVELS))
        for part in partition:
            part_t = tuple(sorted(int(a) for a in part))
            idx = tuple(now[:, a] for a in part_t)
            for i in part_t:
                out[:, i, :] = self._table(k, part_t, i)[idx]
        return out

    def clock_probs(self, m: int) -> np.ndarray:
        return np.broadcast_to(self.clock[None], (m, self.n, N_LEVELS)).copy()

    def persistence_probs(self, now: np.ndarray, k: int) -> np.ndarray:
        """The level stays with the training stay share; otherwise either other level, equally."""
        stay = self.stay[k]
        out = np.broadcast_to(((1.0 - stay) / 2.0)[None, :, None], (len(now), self.n, N_LEVELS)).copy()
        np.put_along_axis(out, np.asarray(now)[:, :, None], np.broadcast_to(stay[None, :, None], (len(now), self.n, 1)), axis=2)
        return out

    def linear_probs(self, now: np.ndarray, k: int) -> np.ndarray:
        """The VAR(1) k-step Gaussian forecast integrated into the same 3 levels, from the same
        3-level "now" the network sees (each level coded as its training-window mean).

        Same input for both models, so "network beats linear" measures the model class, not the
        input resolution.
        """
        coded = np.take_along_axis(self.level_means[None], np.asarray(now)[:, :, None], axis=2)[:, :, 0]
        return gaussian_level_probs(
            self.var1.forecast(coded, k), _var1_forecast_var(self.var1, k), self.edges
        )

    def linear_continuous_probs(self, z0: np.ndarray, k: int) -> np.ndarray:
        """The same VAR(1) forecast fitted on, and fed, the full-resolution values. Context only:
        it knows more about "now" than the network does."""
        m = self.var1_continuous
        return gaussian_level_probs(m.forecast(z0, k), _var1_forecast_var(m, k), self.edges)


def fit_network(
    z_train: np.ndarray,
    *,
    lags: Sequence[int],
    bond_dims: Sequence[int] = born_machine.BOND_DIMS,
    max_sweeps: int = born_machine.DEFAULT_SWEEPS,
) -> NetworkForecaster:
    """Fit the chains and baselines on a standardized (T, n) training window."""
    z_train = np.asarray(z_train, dtype=float)
    lags = tuple(int(k) for k in lags)
    if min(lags) < 1 or z_train.shape[0] <= max(lags) + 4:
        raise ValueError(f"fit_network: {z_train.shape[0]} rows is too few for lags {lags}")
    edges = tercile_edges(z_train)
    levels = to_levels(z_train, edges)
    n = z_train.shape[1]
    clock = np.stack([np.bincount(levels[:, i], minlength=N_LEVELS) for i in range(n)]).astype(float)
    clock /= clock.sum(axis=1, keepdims=True)
    level_means = np.zeros((n, N_LEVELS))
    for i in range(n):
        for lvl in range(N_LEVELS):
            hit = levels[:, i] == lvl
            # An empty level (tied edges) gets its nearest edge as its code.
            level_means[i, lvl] = z_train[hit, i].mean() if hit.any() else edges[i, min(lvl, 1)]
    coded = np.take_along_axis(level_means, levels.T, axis=1).T
    machines, selections, stay = {}, {}, {}
    for k in lags:
        machines[k], selections[k] = born_machine.select_and_fit(
            _pairs(levels, k), d=N_LEVELS, bond_dims=bond_dims, max_sweeps=max_sweeps
        )
        stay[k] = (levels[:-k] == levels[k:]).mean(axis=0)
    return NetworkForecaster(
        lags=lags, edges=edges, machines=machines, selections=selections, clock=clock, stay=stay,
        var1=fit_var1(coded),
        var1_continuous=fit_var1(z_train),
        level_means=level_means,
    )


def _refit_skip_reason(x: np.ndarray, e: int, window: int, last_scored_row: int) -> str | None:
    """Same refusals as ``rolling_errors``: a missing value, or a (near-)constant training column."""
    if not np.all(np.isfinite(x[e - window:last_scored_row + 1])):
        return "missing"
    train = x[e - window:e]
    mu, sd = train.mean(axis=0), train.std(axis=0)
    flat = np.flatnonzero(sd < NEAR_CONSTANT_REL_STD * np.maximum(1.0, np.abs(mu)))
    return f"constant:{int(flat[0])}" if flat.size else None


def network_rolling_errors(
    x: np.ndarray,
    *,
    partitions: Mapping[str, Partition] | None = None,
    window: int = DEFAULT_WINDOW,
    step: int = DEFAULT_STEP,
    horizon: int = DEFAULT_HORIZON,
    lags: Sequence[int] | None = None,
    bond_dims: Sequence[int] = born_machine.BOND_DIMS,
    max_sweeps: int = born_machine.DEFAULT_SWEEPS,
) -> RollingErrors:
    """The ``rolling_errors`` protocol, scored by log-loss in bits on 3-level organs.

    ``whole`` is the trained network, ``clock`` and ``persistence`` are the level baselines,
    ``linear`` is the VAR(1) forecast in the same 3 levels, and each cut is the product over
    parts of p(later_A | now_A) from the same chain. Columns follow ``lags`` (default 1..horizon),
    so ``skill_curve``, ``bootstrap_horizon``, ``cut_cost_curve`` and ``cut_cost_vs_null`` read it
    unchanged. A refit trains one chain per lag; keep ``step`` large when lags are many.
    """
    x = np.asarray(x, dtype=float)
    T, n = x.shape
    lags = tuple(int(k) for k in (lags if lags is not None else range(1, horizon + 1)))
    max_lag = max(lags)
    partitions = dict(partitions or {})
    for part in partitions.values():
        _check_partition(part, n)
    whole, clock, persist, linear = [], [], [], []
    cuts: dict[str, list[np.ndarray]] = {name: [] for name in partitions}
    skipped: list[tuple[int, str]] = []
    for e in range(window, T - max_lag, step):
        origins = np.arange(e, min(e + step, T - max_lag))
        why = _refit_skip_reason(x, e, window, int(origins[-1]) + max_lag)
        if why:
            skipped.append((e, why))
            continue
        train = x[e - window:e]
        mu, sd = train.mean(axis=0), train.std(axis=0)
        z = (x - mu) / sd
        fc = fit_network(z[e - window:e], lags=lags, bond_dims=bond_dims, max_sweeps=max_sweeps)
        lv = to_levels(z, fc.edges)
        now = lv[origins]
        cols = {key: [] for key in ("whole", "clock", "persist", "linear", *(f"cut:{c}" for c in partitions))}
        for k in lags:
            target = lv[origins + k]
            cols["whole"].append(_bits(fc.network_probs(now, k), target))
            cols["clock"].append(_bits(fc.clock_probs(len(origins)), target))
            cols["persist"].append(_bits(fc.persistence_probs(now, k), target))
            cols["linear"].append(_bits(fc.linear_probs(now, k), target))
            for name, part in partitions.items():
                cols[f"cut:{name}"].append(_bits(fc.network_probs(now, k, part), target))
        whole.append(np.stack(cols["whole"], axis=1))
        clock.append(np.stack(cols["clock"], axis=1))
        persist.append(np.stack(cols["persist"], axis=1))
        linear.append(np.stack(cols["linear"], axis=1))
        for name in partitions:
            cuts[name].append(np.stack(cols[f"cut:{name}"], axis=1))
    if not whole:
        why = f"every refit was skipped: {skipped[:3]}" if skipped else "too short"
        raise ValueError(f"network_rolling_errors: {T} rows, window={window}, lags={lags}: {why}")
    return RollingErrors(
        whole=np.concatenate(whole),
        clock=np.concatenate(clock),
        persistence=np.concatenate(persist),
        cuts={name: np.concatenate(v) for name, v in cuts.items()},
        block=step,
        skipped=tuple(skipped),
        linear=np.concatenate(linear),
        lags=lags,
    )


@dataclass(frozen=True)
class SkillSpread:
    """Skill of one forecast against another per column, with a moving-block bootstrap."""

    skill: np.ndarray  # 1 - loss_model / loss_base
    bits: np.ndarray  # mean loss_base - loss_model per row (bits; > 0 = model better)
    p10: np.ndarray  # skill, 10th percentile over resamples
    p90: np.ndarray
    p_real: np.ndarray  # share of resamples in which the model beats the base (skill > 0)


def skill_spread(
    model_err: np.ndarray,
    base_err: np.ndarray,
    *,
    block: int,
    n_boot: int = 1000,
    rng: np.random.Generator | None = None,
) -> SkillSpread:
    """Paired moving-block bootstrap over origins: both forecasts resampled on the same rows."""
    rng = rng if rng is not None else np.random.default_rng()
    m = model_err.shape[0]
    block = max(1, min(block, m))
    n_blocks = int(np.ceil(m / block))
    boots = np.empty((n_boot, model_err.shape[1]))
    for b in range(n_boot):
        starts = rng.integers(0, m - block + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:m]
        boots[b] = skill_curve(model_err[idx], base_err[idx])
    return SkillSpread(
        skill=skill_curve(model_err, base_err),
        bits=(base_err - model_err).mean(axis=0),
        p10=np.percentile(boots, 10, axis=0),
        p90=np.percentile(boots, 90, axis=0),
        p_real=(boots > 0).mean(axis=0),
    )


# --- Cut cost on the partition lattice ----------------------------------------------------------


def set_partitions(n: int) -> list[list[list[int]]]:
    """Every partition of 0..n-1 (Bell(n) of them), the whole system first."""
    if n < 1:
        return []
    out: list[list[list[int]]] = []

    def grow(i: int, parts: list[list[int]]) -> None:
        if i == n:
            out.append([list(p) for p in parts])
            return
        for p in parts:
            p.append(i)
            grow(i + 1, parts)
            p.pop()
        parts.append([i])
        grow(i + 1, parts)
        parts.pop()

    grow(0, [])
    return out


def partition_normalization(partition: Partition, n_levels: int = N_LEVELS) -> float:
    """Balduzzi-Tononi normalization: (parts - 1) times the smallest part's maximum entropy (bits).

    The minimum-information partition minimizes cost / normalization, so a cut that only peels off
    one organ is not favoured just because it is small. 0 for the whole system (never a cut).
    """
    sizes = [len(p) for p in partition]
    return (len(sizes) - 1) * min(sizes) * float(np.log2(n_levels))


def linear_cut_bits(z: np.ndarray, partition: Partition, k: int) -> float:
    """Gaussian twin of the network cut cost, closed form from the lag-k model on ``z``.

    Sum over organs of ½ log2(var(organ | its part's past) / var(organ | everyone's past)), from
    ``lag_k_model``. Same per-organ score as the network, so the two minimum-information partitions
    can be compared. This is not Oizumi's Φ_G (which needs an iterative fit); it is the
    "each part forecasts from its own past" member of the family that Φ_G belongs to.
    """
    z = np.asarray(z, dtype=float)
    _check_partition(partition, z.shape[1])
    whole = np.diag(lag_k_model(z, k)[1])
    total = 0.0
    for part in partition:
        idx = np.asarray(sorted(part), dtype=int)
        cut = np.diag(lag_k_model(z[:, idx], k)[1])
        total += 0.5 * float(np.sum(np.log2(np.maximum(cut, 1e-12) / np.maximum(whole[idx], 1e-12))))
    return total


@dataclass(frozen=True)
class LatticeCosts:
    """Held-out network cut cost and closed-form linear cut cost on every partition, per lag."""

    partitions: tuple[tuple[tuple[int, ...], ...], ...]
    lags: tuple[int, ...]
    network_bits: np.ndarray  # (P, K): mean held-out log-loss(cut) - log-loss(whole), bits per row
    linear_bits: np.ndarray  # (P, K): closed form on the training rows, bits
    normalization: np.ndarray  # (P,)
    whole_bits: np.ndarray  # (K,) network held-out log-loss per row (bits, summed over organs)
    linear_whole_bits: np.ndarray  # (K,) VAR(1) in 3 levels, same rows
    n_test: int

    def mip(self, which: str, col: int) -> int:
        """Index of the minimum-information partition (the whole system is never a candidate)."""
        cost = self.network_bits if which == "network" else self.linear_bits
        ok = self.normalization > 0
        score = np.where(ok, cost[:, col] / np.where(ok, self.normalization, 1.0), np.inf)
        return int(np.argmin(score))


def held_out_lattice(
    x: np.ndarray,
    *,
    lags: Sequence[int],
    train_rows: int,
    partitions: Sequence[Partition] | None = None,
    bond_dims: Sequence[int] = born_machine.BOND_DIMS,
    max_sweeps: int = born_machine.DEFAULT_SWEEPS,
) -> tuple[LatticeCosts, NetworkForecaster]:
    """Train once on rows [0, train_rows), score every partition on the rows after.

    The network's cut forecasts all come from the one chain per lag; no partition is refit.
    """
    x = np.asarray(x, dtype=float)
    T, n = x.shape
    lags = tuple(int(k) for k in lags)
    if not np.all(np.isfinite(x)):
        raise ValueError("held_out_lattice: missing values")
    if n > 8:
        raise ValueError(f"held_out_lattice: {n} organs; the full search is for n <= 8")
    parts = [list(map(list, p)) for p in (partitions if partitions is not None else set_partitions(n))]
    train = x[:train_rows]
    mu, sd = train.mean(axis=0), train.std(axis=0)
    if np.any(sd <= 0):
        raise ValueError("held_out_lattice: a constant training column")
    z = (x - mu) / sd
    fc = fit_network(z[:train_rows], lags=lags, bond_dims=bond_dims, max_sweeps=max_sweeps)
    lv = to_levels(z, fc.edges)
    origins = np.arange(train_rows, T - max(lags))
    if len(origins) < 2:
        raise ValueError("held_out_lattice: no held-out rows")
    now = lv[origins]
    net = np.zeros((len(parts), len(lags)))
    lin = np.zeros((len(parts), len(lags)))
    whole_bits = np.zeros(len(lags))
    lin_whole = np.zeros(len(lags))
    for c, k in enumerate(lags):
        target = lv[origins + k]
        w = _bits(fc.network_probs(now, k), target)
        whole_bits[c] = w.mean()
        lin_whole[c] = _bits(fc.linear_probs(now, k), target).mean()
        for p, part in enumerate(parts):
            net[p, c] = (_bits(fc.network_probs(now, k, part), target) - w).mean()
            lin[p, c] = linear_cut_bits(z[:train_rows], part, k)
    return LatticeCosts(
        partitions=tuple(tuple(tuple(sorted(a)) for a in p) for p in parts),
        lags=lags,
        network_bits=net,
        linear_bits=lin,
        normalization=np.array([partition_normalization(p) for p in parts]),
        whole_bits=whole_bits,
        linear_whole_bits=lin_whole,
        n_test=len(origins),
    ), fc


@dataclass(frozen=True)
class LatticeVsNull:
    observed: LatticeCosts
    null_network_bits: np.ndarray  # (n_null, P, K), each draw a chain refit on shifted data
    null_linear_bits: np.ndarray  # (n_null, P, K)

    @property
    def p_real(self) -> np.ndarray:
        """(P, K): share of null draws whose network cut cost is below the observed one."""
        return (self.null_network_bits < self.observed.network_bits[None]).mean(axis=0)

    @property
    def p_real_linear(self) -> np.ndarray:
        return (self.null_linear_bits < self.observed.linear_bits[None]).mean(axis=0)


def lattice_vs_null(
    x: np.ndarray,
    *,
    lags: Sequence[int],
    train_fraction: float = 2 / 3,
    partitions: Sequence[Partition] | None = None,
    n_null: int = 20,
    guard: int = DEFAULT_NULL_GUARD,
    rng: np.random.Generator | None = None,
    bond_dims: Sequence[int] = born_machine.BOND_DIMS,
    max_sweeps: int = born_machine.DEFAULT_SWEEPS,
) -> LatticeVsNull:
    """``held_out_lattice`` on the data and on ``n_null`` time-shifted copies (``time_shift_null``).

    Each copy refits its chains, so the null carries the same fitting noise as the observed cost.
    """
    rng = rng if rng is not None else np.random.default_rng()
    x = np.asarray(x, dtype=float)
    kw = dict(lags=lags, partitions=partitions, bond_dims=bond_dims, max_sweeps=max_sweeps)
    observed, _ = held_out_lattice(x, train_rows=int(train_fraction * len(x)), **kw)
    null_net, null_lin = [], []
    for _ in range(n_null):
        xs = time_shift_null(x, guard=guard, rng=rng)
        res, _ = held_out_lattice(xs, train_rows=int(train_fraction * len(xs)), **kw)
        null_net.append(res.network_bits)
        null_lin.append(res.linear_bits)
    return LatticeVsNull(observed=observed, null_network_bits=np.stack(null_net), null_linear_bits=np.stack(null_lin))


@dataclass(frozen=True)
class CutInformation:
    """Per contiguous cut of a trained chain: entanglement against classical mutual information."""

    entanglement_bits: np.ndarray  # (L-1,)
    mutual_information_bits: np.ndarray  # (L-1,)
    mi_standard_error: np.ndarray  # 0 when exact

    @property
    def gap_bits(self) -> np.ndarray:
        """Capacity the chain uses beyond the classical shared information (>= 0 up to MC error)."""
        return self.entanglement_bits - self.mutual_information_bits


def cut_information(bm: born_machine.BornMachine, *, rng: np.random.Generator | None = None) -> CutInformation:
    mi, se = born_machine.mutual_information(bm, rng=rng)
    return CutInformation(
        entanglement_bits=born_machine.entanglement_entropies(bm),
        mutual_information_bits=mi,
        mi_standard_error=se,
    )
