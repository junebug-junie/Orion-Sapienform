"""Whole-body structure (causal geometry Layer 2): pure numpy, no I/O.

Spec: docs/superpowers/specs/2026-11-11-causal-geometry-redesign.md, "Layer 2". Issue #3290.

Input is a (T, n) array of channels in 5-minute buckets (body pressures, bus channel counts,
cognition and world counts). NaN marks a bucket the source did not cover in full.

* ``tod_residuals`` turns levels into **surprise**: each value minus the same channel's mean for
  that hour of day over the previous up-to-14 days. Only earlier days enter the profile, so
  no value is compared with a profile that already contains it. This keeps "both follow the
  clock" from posing as coupling.
* ``window_features`` computes, for one window: total correlation (coupling), effective number
  of modes, the leading mode, O-information (redundancy vs synergy), and cross-group total
  correlation (body vs mind).
* ``shifted_null_window`` draws each channel's window from a different time, at least a guard
  apart from every other channel. That breaks coupling and keeps each channel's own rhythm.
* ``match_modes`` checks whether modes stay recognisable from one day to the next.

Everything is Gaussian (correlation-matrix) information theory. Correlations are shrunk toward
the identity by a fixed amount before any log-determinant, so a window with fewer rows than
channels still has a finite value. The null gets the same shrinkage, so the bias cancels in
"observed minus null".

No alpha line (Juniper's rule): every comparison reports effect, null spread and P(real).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

BUCKETS_PER_HOUR = 12
BUCKETS_PER_DAY = 288
PROFILE_DAYS = 14
PROFILE_MIN_DAYS = 3
MIN_PRESENCE = 0.9
SHRINKAGE = 0.1


def tod_residuals(
    values: np.ndarray,
    bucket_index: np.ndarray,
    *,
    profile_days: int = PROFILE_DAYS,
    min_days: int = PROFILE_MIN_DAYS,
) -> np.ndarray:
    """Value minus that channel's hour-of-day mean over the previous ``profile_days`` days.

    ``bucket_index`` is the absolute 5-minute bucket number (epoch seconds // 300) of each row.
    A row whose profile rests on fewer than ``min_days`` earlier days is NaN, as is a NaN input.
    """
    values = np.asarray(values, dtype=float)
    idx = np.asarray(bucket_index, dtype=np.int64)
    if values.ndim != 2 or len(idx) != len(values):
        raise ValueError(f"tod_residuals: values {values.shape} vs index {idx.shape}")
    day = idx // BUCKETS_PER_DAY
    hour = (idx // BUCKETS_PER_HOUR) % 24
    d0 = int(day.min())
    n_days = int(day.max()) - d0 + 1
    n = values.shape[1]
    sums = np.zeros((n_days, 24, n))
    counts = np.zeros((n_days, 24, n))
    finite = np.isfinite(values)
    np.add.at(sums, (day - d0, hour), np.where(finite, values, 0.0))
    np.add.at(counts, (day - d0, hour), finite.astype(float))
    with np.errstate(invalid="ignore"):
        hour_mean = sums / counts  # (day, hour, ch), NaN where the hour had no data
    present = np.isfinite(hour_mean)
    cs = np.concatenate([np.zeros((1, 24, n)), np.cumsum(np.where(present, hour_mean, 0.0), axis=0)])
    cc = np.concatenate([np.zeros((1, 24, n)), np.cumsum(present.astype(float), axis=0)])
    out = np.full_like(values, np.nan)
    rel = day - d0
    lo = np.maximum(rel - profile_days, 0)
    s = cs[rel, hour] - cs[lo, hour]
    c = cc[rel, hour] - cc[lo, hour]
    ok = c >= min_days
    with np.errstate(invalid="ignore", divide="ignore"):
        profile = s / c
    out[ok] = values[ok] - profile[ok]
    return out


def select_channels(x: np.ndarray, *, min_presence: float = MIN_PRESENCE) -> np.ndarray:
    """Indices of channels present in at least ``min_presence`` of rows and not constant."""
    x = np.asarray(x, dtype=float)
    present = np.isfinite(x).mean(axis=0) >= min_presence
    with np.errstate(invalid="ignore"):
        sd = np.nanstd(x, axis=0)
    return np.flatnonzero(present & (sd > 0))


def correlation(x: np.ndarray) -> np.ndarray:
    """Correlation of a (T, n) window. A missing value is set to the column mean.

    That shrinks a correlation toward 0 (conservative) and never invents one. A column that is
    constant in this window gets zero correlation with everything (and 1 on the diagonal).
    """
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x, axis=0)
    xc = np.where(np.isfinite(x), x - mu, 0.0)
    sd = np.sqrt((xc ** 2).mean(axis=0))
    z = np.divide(xc, sd, out=np.zeros_like(xc), where=sd > 0)
    r = z.T @ z / len(z)
    np.fill_diagonal(r, 1.0)
    return np.clip(r, -1.0, 1.0)


def _shrink(r: np.ndarray, lam: float = SHRINKAGE) -> np.ndarray:
    return (1.0 - lam) * r + lam * np.eye(len(r))


def total_correlation(r: np.ndarray, *, lam: float = SHRINKAGE) -> float:
    """Gaussian multi-information in nats: -½ log det R (Watanabe 1960)."""
    return float(-0.5 * np.linalg.slogdet(_shrink(r, lam))[1])


def o_information(r: np.ndarray, *, lam: float = SHRINKAGE) -> float:
    """Gaussian O-information in nats (Rosas et al. 2019): TC minus dual TC.

    Positive: the parts repeat each other (redundancy). Negative: information held only by the
    parts together (synergy). For a correlation matrix R with precision P = R⁻¹:
    Ω = -log det R - ½ Σ log P_ii.
    """
    rs = _shrink(r, lam)
    p = np.linalg.inv(rs)
    return float(-np.linalg.slogdet(rs)[1] - 0.5 * np.log(np.diag(p)).sum())


def participation_ratio(r: np.ndarray) -> float:
    """Effective number of modes: (Σλ)² / Σλ² = n² / ‖R‖²_F. 1 = one mode, n = independent."""
    return float(len(r) ** 2 / (r ** 2).sum())


def cross_total_correlation(r: np.ndarray, a: Sequence[int], b: Sequence[int], *, lam: float = SHRINKAGE) -> float:
    """Gaussian mutual information between channel groups a and b: TC(a∪b) - TC(a) - TC(b)."""
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    ab = np.concatenate([a, b])
    return (
        total_correlation(r[np.ix_(ab, ab)], lam=lam)
        - total_correlation(r[np.ix_(a, a)], lam=lam)
        - total_correlation(r[np.ix_(b, b)], lam=lam)
    )


def top_modes(r: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Largest k eigenvalues (as shares of n) and their eigenvectors (columns), sign-fixed.

    Sign rule: the largest-magnitude loading is positive, so the same mode on two days gets the
    same sign.
    """
    w, v = np.linalg.eigh(r)
    order = np.argsort(w)[::-1][:k]
    w, v = w[order], v[:, order]
    flip = np.sign(v[np.argmax(np.abs(v), axis=0), np.arange(v.shape[1])])
    return w / len(r), v * flip


def match_modes(ref: np.ndarray, cur: np.ndarray) -> np.ndarray:
    """For each reference mode (column of ``ref``), the best |cosine| with any mode in ``cur``.

    Greedy one-to-one matching, strongest pair first.
    """
    sim = np.abs(ref.T @ cur)
    out = np.zeros(ref.shape[1])
    used_r, used_c = set(), set()
    for flat in np.argsort(sim, axis=None)[::-1]:
        i, j = np.unravel_index(flat, sim.shape)
        if i in used_r or j in used_c:
            continue
        out[i] = sim[i, j]
        used_r.add(i)
        used_c.add(j)
        if len(used_r) == ref.shape[1] or len(used_c) == cur.shape[1]:
            break
    return out


@dataclass(frozen=True)
class WindowFeatures:
    coupling: float  # total correlation, nats
    modes_effective: float
    leading_share: float  # top eigenvalue / n
    leading_vector: np.ndarray
    redundancy_synergy: float  # O-information, nats
    body_mind: float | None  # cross-group TC, nats; None without both groups


def window_features(
    x: np.ndarray,
    *,
    groups: Mapping[str, Sequence[int]] | None = None,
    lam: float = SHRINKAGE,
) -> WindowFeatures:
    """All Layer 2 features of one (T, n) window except drift (which needs two windows)."""
    r = correlation(x)
    shares, vecs = top_modes(r, 1)
    body_mind = None
    if groups and groups.get("body") and groups.get("mind"):
        body_mind = cross_total_correlation(r, groups["body"], groups["mind"], lam=lam)
    return WindowFeatures(
        coupling=total_correlation(r, lam=lam),
        modes_effective=participation_ratio(r),
        leading_share=float(shares[0]),
        leading_vector=vecs[:, 0],
        redundancy_synergy=o_information(r, lam=lam),
        body_mind=body_mind,
    )


def structure_drift(r_prev: np.ndarray, r_cur: np.ndarray) -> float:
    """How fast the pattern changes: RMS difference between two correlation matrices."""
    n = len(r_cur)
    return float(np.sqrt(((r_cur - r_prev) ** 2).sum() / (n * (n - 1))))


def null_starts(
    n_rows: int,
    length: int,
    n_channels: int,
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Random window starts, one per channel, spread so every pair is at least ``gap`` apart.

    Starts are a random permutation of an even grid over the series with a small jitter. Returns
    (starts, gap). The caller reports ``gap``, because a pair of channels closer than their own
    lagged coupling keeps some of it, and that makes the null conservative.
    """
    room = n_rows - length
    if room < n_channels:
        raise ValueError(f"null_starts: {n_rows} rows leave no room for {n_channels} shifted windows of {length}")
    step = room // n_channels
    jitter = rng.integers(0, max(1, step // 2), size=n_channels)
    starts = rng.permutation(n_channels) * step + jitter
    gap = step - (max(1, step // 2) - 1)
    return starts, int(gap)


def shifted_null_window(x: np.ndarray, length: int, starts: np.ndarray) -> np.ndarray:
    """A (length, n) window in which channel i comes from rows starts[i]..starts[i]+length."""
    return np.stack([x[s:s + length, i] for i, s in enumerate(starts)], axis=1)


def z_and_p_real(observed: float, null: np.ndarray, *, higher_is_structure: bool = True) -> tuple[float, float]:
    """(observed - null mean) / null sd, and the share of null draws on the "less structure" side.

    ``higher_is_structure`` is False for a feature where structure reads low (effective modes).
    """
    null = np.asarray(null, dtype=float)
    sd = float(null.std(ddof=1)) if len(null) > 1 else 0.0
    z = (observed - float(null.mean())) / sd if sd > 0 else float("nan")
    beaten = null < observed if higher_is_structure else null > observed
    return z, float(beaten.mean())
