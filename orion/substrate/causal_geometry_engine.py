"""Causal Geometry v1, Phase A core engine: observed-geometry measurement.

Extracted from `scripts/causal_geometry_report.py` (which now imports from here
and keeps only its CLI wrapper) so the new Phase B producer
(`orion/substrate/causal_geometry_producer.py`, called periodically from
`services/orion-field-digester/app/worker.py`) can call the exact same
measurement logic the standalone report script uses, instead of duplicating it.

See docs/superpowers/specs/2026-07-16-causal-geometry-v1-design.md for the full
three-phase design, and `scripts/causal_geometry_report.py`'s own (still
present) module docstring for the detailed rationale behind the fixed
constants, the source tables, resampling, lagged cross-correlation, the
Bonferroni-corrected-per-pair + Benjamini-Hochberg-corrected-across-pairs
surrogate significance test (see that docstring's "p-value resolution" note
for two live incidents where a floored p starved BH), and the
designed-vs-observed
divergence matching rule (including the `#<capability_channel>` target_id
disambiguation suffix -- see `orion/substrate/field_topology_plasticity.py`'s
`_base_target_id()` for the corresponding join-side fix).
"""
from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from orion.schemas.causal_geometry import (
    CausalGeometryDivergenceEntryV1,
    CausalGeometryEdgeV1,
    CausalGeometrySnapshotV1,
)

_REPO_ROOT = str(Path(__file__).resolve().parents[2])

# --- documented, fixed knobs (see scripts/causal_geometry_report.py's module
# docstring for the full rationale behind each of these) -------------------

BUCKET_SECONDS = 300
LAG_GRID_SECONDS: Tuple[int, ...] = (0, 300, 600, 900, 1800)

DEFAULT_WINDOW_HOURS = 168.0
DEFAULT_MIN_SAMPLES = 30
# Cap on time-shift surrogates per test (see time_shift_surrogate_rs: every
# eligible shift is used when there are at most this many -- 1818 at n=2017).
# Since 2026-11-10 it no longer sets the smallest reportable p: past the last
# 10 surrogates the p can come from a tail fit (see surrogate_tail_pvalue).
DEFAULT_N_SURROGATES = 2000
DEFAULT_ALPHA = 0.05
DEFAULT_BIOMETRICS_NODE = "athena"
DEFAULT_SEED = 1337

DEFAULT_POSTGRES_URI = "postgresql://postgres:postgres@127.0.0.1:55432/conjourney"

DEFAULT_FIELD_TOPOLOGY_PATH = str(
    Path(_REPO_ROOT) / "config" / "field" / "orion_field_topology.v1.yaml"
)

ChannelPoints = List[Tuple[datetime, float]]
ChannelSeries = Dict[str, ChannelPoints]


# --- data fetch (real DB) ---------------------------------------------------


def fetch_channels(
    postgres_uri: str,
    window_start: datetime,
    *,
    biometrics_node: str = DEFAULT_BIOMETRICS_NODE,
) -> Tuple[ChannelSeries, Dict[str, int]]:
    """Pull and flatten organ signal tables into named channels.

    Uses psycopg2 (already a dependency of several services; no new dependency
    added when called from a service that already has it, e.g. orion-hub,
    orion-field-digester). Each row's jsonb column is flattened into one
    channel per key (`biometrics:<key>`); `attention_salience_trace.salience` has no
    sub-keys and becomes the single scalar channel `attention:salience`.

    Returns `(channels, table_row_counts)`. `table_row_counts` maps each source
    table name to the raw row count pulled in the window -- a table with 0 rows
    in the window contributes zero channels and would otherwise be silently
    invisible from the rest of the report.
    """
    import psycopg2

    channels: ChannelSeries = {}
    table_row_counts: Dict[str, int] = {}

    def _append(channel_id: str, ts: datetime, value: Any) -> None:
        if value is None or isinstance(value, bool):
            return
        try:
            fvalue = float(value)
        except (TypeError, ValueError):
            return
        channels.setdefault(channel_id, []).append((ts, fvalue))

    conn = psycopg2.connect(postgres_uri)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT created_at AS ts, salience
                FROM attention_salience_trace
                WHERE created_at >= %s
                ORDER BY ts
                """,
                (window_start,),
            )
            rows = cur.fetchall()
            table_row_counts["attention_salience_trace"] = len(rows)
            for ts, salience in rows:
                _append("attention:salience", ts, salience)

            cur.execute(
                """
                SELECT timestamp::timestamptz AS ts, pressures
                FROM orion_biometrics_summary
                WHERE node = %s AND timestamp::timestamptz >= %s
                ORDER BY ts
                """,
                (biometrics_node, window_start),
            )
            rows = cur.fetchall()
            table_row_counts["orion_biometrics_summary"] = len(rows)
            for ts, pressures in rows:
                for key, value in (pressures or {}).items():
                    _append(f"biometrics:{key}", ts, value)

            # self_state_predictions source removed 2026-10-07: the table has
            # had 0 rows since its producer was deleted in the 2026-07-22
            # SelfStateV1 burn, so it contributed no channel to any snapshot
            # (live snapshots: 19 channels, every one from the other sources).

            # substrate_field_state's node:substrate.bus_synaptic source removed
            # 2026-11: the node was retired (PE suite rebuild step 5) and the
            # field digester prunes it, so it contributed no channel any more.
    finally:
        conn.close()

    for points in channels.values():
        points.sort(key=lambda p: p[0])
    return channels, table_row_counts


# --- resampling + lagged correlation ----------------------------------------


def bucketize(points: ChannelPoints, bucket_seconds: int = BUCKET_SECONDS) -> Dict[int, float]:
    """Mean-aggregate raw (timestamp, value) points onto a bucket-index grid.

    Bucket index is `int(ts.timestamp() // bucket_seconds)`. Empty buckets are
    simply absent from the returned dict -- this never interpolates or
    forward-fills, since a fabricated value in a gap could manufacture a
    spurious correlation exactly opposite of this feature's purpose.
    """
    grouped: Dict[int, List[float]] = {}
    for ts, value in points:
        idx = int(ts.timestamp() // bucket_seconds)
        grouped.setdefault(idx, []).append(value)
    return {idx: float(np.mean(vals)) for idx, vals in grouped.items()}


def _aligned_arrays(
    source_buckets: Dict[int, float], target_buckets: Dict[int, float], lag_buckets: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Pair up source[idx] with target[idx + lag_buckets] for every idx present
    in both, i.e. target observed `lag_buckets` buckets after source."""
    xs: List[float] = []
    ys: List[float] = []
    for idx in sorted(source_buckets.keys()):
        j = idx + lag_buckets
        if j in target_buckets:
            xs.append(source_buckets[idx])
            ys.append(target_buckets[j])
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def _pearson(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if len(x) < 2 or len(y) < 2:
        return None
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    r = float(np.corrcoef(x, y)[0, 1])
    if np.isnan(r):
        return 0.0
    return r


class DirectionResult:
    __slots__ = ("source_id", "target_id", "lag_sec", "r", "n", "xs", "ys")

    def __init__(self, source_id: str, target_id: str, lag_sec: int, r: float, n: int, xs: np.ndarray, ys: np.ndarray):
        self.source_id = source_id
        self.target_id = target_id
        self.lag_sec = lag_sec
        self.r = r
        self.n = n
        self.xs = xs
        self.ys = ys


def _best_direction(
    source_id: str,
    target_id: str,
    source_buckets: Dict[int, float],
    target_buckets: Dict[int, float],
    lag_grid_seconds: Tuple[int, ...],
    bucket_seconds: int,
) -> Optional[DirectionResult]:
    best: Optional[DirectionResult] = None
    for lag_sec in lag_grid_seconds:
        lag_buckets = lag_sec // bucket_seconds
        xs, ys = _aligned_arrays(source_buckets, target_buckets, lag_buckets)
        if len(xs) < 2:
            continue
        r = _pearson(xs, ys)
        if r is None:
            continue
        if best is None or abs(r) > abs(best.r):
            best = DirectionResult(source_id, target_id, lag_sec, r, len(xs), xs, ys)
    return best


# Smallest p-value time_shift_pvalue reports. A tail fit can underflow to an
# exact 0.0, which would read as "certain"; no finite sample supports that.
# Far below any BH threshold, so it never decides an edge -- it only keeps the
# stored `significance` honest.
MIN_REPORTED_PVALUE = 1e-300

# Fisher z of |r| -> 1 is infinite; clip just inside so a perfectly
# collinear pair gets a huge-but-finite z instead of inf/nan.
_R_CLIP = 1.0 - 1e-12

# Null shifts: |s| from the guard up to MAX_SHIFT_FRACTION * n. The guard
# keeps a real effect out of its own null (see time_shift_surrogate_rs).
GUARD_FRACTION = 0.05
GUARD_MIN = 12
MAX_SHIFT_FRACTION = 0.5

# Tail-fit knobs (Knijnenburg et al. 2009, "Fewer permutations, more accurate
# P-values", Bioinformatics 25:i161): plain rank p when at least
# MIN_EXCEEDANCES surrogates reach the observed statistic; otherwise fit a
# generalized Pareto distribution (GPD) to the top TAIL_FIT_SIZE surrogates.
MIN_EXCEEDANCES = 10
TAIL_FIT_SIZE = 250
_MIN_TAIL_FIT_SIZE = 20

# Extrapolating past the surrogates needs a tail built from many roughly
# independent values: Bartlett effective sample size >= N_EFF_MIN, and each
# series' participation size >= PARTICIPATION_MIN. Otherwise there is no tail
# fit and p >= 1/(units + 1) (see time_shift_pvalue).
N_EFF_MIN = 100.0
PARTICIPATION_MIN = 50.0


def time_shift_surrogate_rs(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    n_surrogates: int = DEFAULT_N_SURROGATES,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Pearson r of `xs` against `ys` shifted in time, over the overlap only.

    For every shift s with guard <= |s| <= MAX_SHIFT_FRACTION * n, in both
    directions: r(xs[s:], ys[:n-s]) and r(xs[:n-s], ys[s:]). No wrap-around:
    each surrogate is two real, contiguous stretches of the two series, so a
    daily cycle, a regime change or a burst is never cut and re-glued.
    guard = max(GUARD_MIN, GUARD_FRACTION * n).

    Why not circular shifts (the pre-2026-11-10 null): (1) shifts near 0 keep
    a real coupling (live power/thermal: the 15 largest surrogates were all
    within 25 buckets of 0, r up to 0.60 against an observed 0.68), so the
    effect sits in its own null; (2) leaving those shifts out of a circular
    null made persistent and periodic nulls badly liberal, because the
    wrap-around seam shrinks r at every far shift (daily-cycle null: 357
    false edges over 5 seeds). Overlap shifts have no seam, so the guard is
    safe. Their r has a little more spread (overlap n/2..n), which only makes
    the test conservative for short-memory series.

    All shifts at once: overlap sums via cumulative sums, the cross term via
    one zero-padded FFT, O(n log n). When there are more eligible shifts than
    `n_surrogates`, that many are drawn without replacement from `rng`.

    Returns an empty array when either series is constant or there are no
    eligible shifts. Raises ValueError on a length mismatch (a caller bug).
    """
    n = len(ys)
    if len(xs) != n:
        raise ValueError(f"time_shift_surrogate_rs: len(xs)={len(xs)} != len(ys)={n}")
    guard = max(GUARD_MIN, int(n * GUARD_FRACTION))
    max_shift = int(n * MAX_SHIFT_FRACTION)
    if max_shift < guard:
        return np.array([], dtype=float)
    sx = float(np.std(xs))
    sy = float(np.std(ys))
    if sx == 0.0 or sy == 0.0:
        return np.array([], dtype=float)
    x = (np.asarray(xs, dtype=float) - np.mean(xs)) / sx
    y = (np.asarray(ys, dtype=float) - np.mean(ys)) / sy
    size = 2 * n
    # lin[s] = sum_t x[t + s] * y[t]; lin[size - s] = sum_t x[t] * y[t + s].
    lin = np.fft.irfft(np.fft.rfft(x, size) * np.conj(np.fft.rfft(y, size)), size)
    cx = np.concatenate([[0.0], np.cumsum(x)])
    cxx = np.concatenate([[0.0], np.cumsum(x * x)])
    cy = np.concatenate([[0.0], np.cumsum(y)])
    cyy = np.concatenate([[0.0], np.cumsum(y * y)])
    shifts = np.arange(guard, max_shift + 1)
    m = (n - shifts).astype(float)

    def _r(sum_xy, a_lo, a_hi, b_lo, b_hi):
        sa = cx[a_hi] - cx[a_lo]
        saa = cxx[a_hi] - cxx[a_lo]
        sb = cy[b_hi] - cy[b_lo]
        sbb = cyy[b_hi] - cyy[b_lo]
        cov = sum_xy - sa * sb / m
        var_a = saa - sa * sa / m
        var_b = sbb - sb * sb / m
        # One-pass variances lose precision when an overlap stretch is nearly
        # constant; the series are standardized (variance 1 per point), so a
        # stretch below 1e-10 per point is treated as constant -> r = 0, the
        # same rule as _pearson's std == 0.
        ok = (var_a > 1e-10 * m) & (var_b > 1e-10 * m)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(ok, cov / np.sqrt(np.maximum(var_a * var_b, 1e-300)), 0.0)
        return r

    # x[s:] against y[:n-s]
    r_fwd = _r(lin[shifts], shifts, np.full_like(shifts, n), np.zeros_like(shifts), n - shifts)
    # x[:n-s] against y[s:]
    r_bwd = _r(lin[size - shifts], np.zeros_like(shifts), n - shifts, shifts, np.full_like(shifts, n))
    rs = np.clip(np.concatenate([r_fwd, r_bwd]), -1.0, 1.0)
    if rs.size > n_surrogates:
        if rng is None:
            rng = np.random.default_rng()
        rs = rng.choice(rs, size=n_surrogates, replace=False)
    return rs


def _acf(values: np.ndarray) -> np.ndarray:
    """Sample autocorrelation at lags 0..n-1 (FFT, zero-padded, biased)."""
    n = len(values)
    centered = np.asarray(values, dtype=float) - np.mean(values)
    spec = np.fft.rfft(centered, 2 * n)
    acov = np.fft.irfft(spec * np.conj(spec), 2 * n)[:n]
    if acov[0] <= 0.0:
        return np.zeros(n)
    return acov / acov[0]


def effective_sample_size(xs: np.ndarray, ys: np.ndarray) -> float:
    """Bartlett (1946) effective number of independent pairs for a
    cross-correlation: n / (1 + 2 * sum_{k=1}^{n/4} rho_x(k) rho_y(k)).

    Used only as a gate on tail extrapolation, not as the test itself (as
    the test it is liberal for persistent series: the truncated sum is noisy).
    Live 2026-11-10: power/thermal 208, cpu/power 621; a near-trending pair
    disk_capacity/mem 8 (r=-0.76, the classic spurious-trend case).
    """
    n = len(xs)
    if n < 8:
        return float(n)
    k = max(2, n // 4 + 1)  # lags 1 .. n // 4
    product_sum = float(np.sum(_acf(xs)[1:k] * _acf(ys)[1:k]))
    return float(n / max(1.0, 1.0 + 2.0 * product_sum))


def participation_size(values: np.ndarray) -> float:
    """Roughly how many points carry a series' variance: n / mean(z^4)
    (n over the kurtosis). Gaussian noise: n/3. A channel that is flat
    except for 3 spikes: ~3, however long it is.

    Bartlett's n_eff only sees autocorrelation, so a rare-event channel
    (isolated spikes on a flat floor) reads as n_eff ~ n. Its null tail is
    then pure floor noise, and a GPD fitted to it extrapolates a single
    coincidence of two spikes to p ~ 1e-30 (review round 2: false edges in
    2-3 of 10 seeds of an m=171 null family).
    """
    v = np.asarray(values, dtype=float)
    sd = float(np.std(v))
    if sd == 0.0:
        return 0.0
    z = (v - float(np.mean(v))) / sd
    return float(len(v) / max(1.0, float(np.mean(z ** 4))))


def _gpd_tail_survival(exceedances: np.ndarray, y: float) -> Optional[float]:
    """Survival P(Y > y) of a GPD fitted to `exceedances` (all >= 0).

    Shape xi by probability-weighted moments (Hosking & Wallis 1987),
    clamped at xi >= 0 (exponential is the lightest tail allowed): a bounded
    fitted tail would declare values just past the largest surrogate
    impossible, which a finite set of shifts cannot show. When clamped, sigma
    is the exponential MLE (the mean). Returns None when no usable fit exists.
    """
    ys = np.sort(exceedances)
    m = len(ys)
    a0 = float(np.mean(ys))
    if not np.isfinite(a0) or a0 <= 0.0:
        return None
    plot_pos = (np.arange(1, m + 1) - 0.35) / m
    a1 = float(np.mean((1.0 - plot_pos) * ys))
    denom = a0 - 2.0 * a1
    if denom <= 0.0:
        # denom = mean(y * (2p - 1)) is >= 0 for sorted data, so this is a
        # degenerate input, not a shape estimate. No fit -> the caller uses
        # the rank p; never fall back to an exponential that may be lighter
        # than the data.
        return None
    xi, sigma = 0.0, a0
    k = a0 / denom - 2.0  # Hosking's k = -xi
    if -k > 0.0:
        xi, sigma = -k, 2.0 * a0 * a1 / denom
    if not np.isfinite(sigma) or sigma <= 0.0:
        return None
    if xi < 1e-9:
        return math.exp(-y / sigma)
    return (1.0 + xi * y / sigma) ** (-1.0 / xi)


def surrogate_tail_pvalue(null_stats: np.ndarray, observed: float, *, allow_tail_fit: bool = True) -> float:
    """One-sided p of `observed` against surrogate values `null_stats`.

    - At least MIN_EXCEEDANCES surrogates >= observed: the add-one rank p
      `(M + 1) / (N + 1)`. No shape assumption.
    - Fewer, and `allow_tail_fit`: a GPD fitted to the top TAIL_FIT_SIZE
      surrogates (Pickands-Balkema-de Haan: exceedances over a high threshold
      are GPD whatever the parent), `(N_tail / N) * S(observed - t)`, never
      above the rank p. A heavy-tailed null (bursty channels) gets a heavy
      fitted tail instead of being mistaken for normal.
    - Otherwise: the rank p, which bottoms out at 1/(N+1).
    """
    stats = np.asarray(null_stats, dtype=float)
    n = len(stats)
    if n == 0:
        return 1.0
    m_exceed = int(np.sum(stats >= observed))
    rank_p = (m_exceed + 1) / (n + 1)
    if m_exceed >= MIN_EXCEEDANCES or not allow_tail_fit:
        return float(rank_p)
    n_tail = min(TAIL_FIT_SIZE, n // 4)
    if n_tail < _MIN_TAIL_FIT_SIZE:
        return float(rank_p)
    top = np.sort(stats)[::-1]
    threshold = 0.5 * (top[n_tail - 1] + top[n_tail])
    survival = _gpd_tail_survival(top[:n_tail] - threshold, observed - threshold)
    if survival is None:
        return float(rank_p)
    return float(min(rank_p, max(MIN_REPORTED_PVALUE, (n_tail / n) * survival)))


def time_shift_pvalue(
    xs: np.ndarray,
    ys: np.ndarray,
    observed_r: float,
    *,
    n_surrogates: int = DEFAULT_N_SURROGATES,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Two-sided p-value for a Pearson r against a time-shift null.

    Null: `xs` and `ys` are independent, each keeping its own serial
    structure. Shifting one series against the other (time_shift_surrogate_rs)
    samples that null without modelling the autocorrelation. Statistic:
    |atanh(r)| (Fisher z; monotone in |r|). p is read by
    `surrogate_tail_pvalue`: rank p in the body; past the last
    MIN_EXCEEDANCES surrogates a GPD tail fit, but only when the pair's
    Bartlett effective sample size is at least N_EFF_MIN and each series'
    participation size is at least PARTICIPATION_MIN. Otherwise the p is the
    rank p, never finer than 1/(units + 1): such a pair cannot become an edge
    (near-trending pairs, slow on/off regimes, pure daily rhythms, rare-event
    channels).

    2026-11-10: replaced the add-one circular-shift rank p, whose floor
    (1/2001; 0.005 after the within-pair Bonferroni x10) sat above BH's
    first-rank threshold 0.05/171 = 0.00029. BH then passed nothing unless 18+
    pairs tied at the floor, and passed all of them when they did (0 edges
    since 2026-10-30, including power->thermal r=+0.61, n=2017). Rejected on
    the way, each by a synthetic null in
    `orion/substrate/tests/test_causal_geometry_pvalue_calibration.py`: a
    normal fit to circular surrogates (false edges in every seed on bursty
    channels, ~0 power on daily cycles), a GPD tail on unguarded circular
    surrogates (real links lost to their own near-0 shifts), and a guarded
    circular null (daily-cycle and regime nulls liberal).

    Assumes stationarity: trending pairs are caught only by the n_eff gate.
    Shifts count array positions, not wall-clock time: `_aligned_arrays`
    drops missing buckets, so on gappy channels (attention:salience,
    bus_synaptic:*) a shift spans gaps.
    Returns 1.0 when no null can be built (constant series, n too short).
    Never returns 0.0 -- floored at MIN_REPORTED_PVALUE.
    """
    if len(xs) != len(ys):
        raise ValueError(f"time_shift_pvalue: len(xs)={len(xs)} != len(ys)={len(ys)}")
    rs = time_shift_surrogate_rs(xs, ys, n_surrogates=n_surrogates, rng=rng)
    if rs.size == 0:
        return 1.0
    null_z = np.abs(np.arctanh(np.clip(rs, -_R_CLIP, _R_CLIP)))
    obs_z = abs(math.atanh(max(-_R_CLIP, min(_R_CLIP, float(observed_r)))))
    n_eff = effective_sample_size(xs, ys)
    participation = min(participation_size(xs), participation_size(ys))
    if n_eff >= N_EFF_MIN and participation >= PARTICIPATION_MIN:
        return surrogate_tail_pvalue(null_z, obs_z)
    # Too few independent or informative values: no extrapolation, and no p
    # finer than 1/(units + 1), the floor of a shift test over that many
    # exchangeable units. After Bonferroni x10 that is >= 10/101 ~= 0.099 >
    # alpha, so such a pair cannot become an edge at all -- it is excluded,
    # not ranked. That also means a pile of trending pairs can never tie at a
    # floor and pass BH together (the old cliff, in miniature).
    units = min(n_eff, participation)
    rank_p = surrogate_tail_pvalue(null_z, obs_z, allow_tail_fit=False)
    return float(max(rank_p, 1.0 / (units + 1.0)))


class PairResult:
    __slots__ = ("channel_a", "channel_b", "n_lag0", "winner", "significant", "p_value")

    def __init__(self, channel_a: str, channel_b: str, n_lag0: int):
        self.channel_a = channel_a
        self.channel_b = channel_b
        self.n_lag0 = n_lag0
        self.winner: Optional[DirectionResult] = None
        self.significant = False
        self.p_value: Optional[float] = None


def compute_pairwise_results(
    channel_buckets: Dict[str, Dict[int, float]],
    *,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    n_surrogates: int = DEFAULT_N_SURROGATES,
    alpha: float = DEFAULT_ALPHA,
    lag_grid_seconds: Tuple[int, ...] = LAG_GRID_SECONDS,
    bucket_seconds: int = BUCKET_SECONDS,
    seed: int = DEFAULT_SEED,
) -> List[PairResult]:
    """Compute lagged cross-correlation + surrogate significance for every
    unordered pair of channels.

    Eligibility gate: zero-lag co-sampled overlap size (the maximum possible n
    for a pair, since positive lags only shrink alignment further). Pairs
    below `min_samples` get a `PairResult` with `winner=None` and are never
    given a fabricated edge -- callers must check `n_lag0 < min_samples`
    before treating the absence of `winner` as "not correlated" instead of
    "not enough data to know".
    """
    channel_ids = sorted(channel_buckets.keys())
    results: List[PairResult] = []
    rng = np.random.default_rng(seed)

    for i, a in enumerate(channel_ids):
        for b in channel_ids[i + 1 :]:
            bucket_a = channel_buckets[a]
            bucket_b = channel_buckets[b]
            n_lag0 = len(set(bucket_a.keys()) & set(bucket_b.keys()))
            result = PairResult(a, b, n_lag0)
            if n_lag0 < min_samples:
                results.append(result)
                continue

            fwd = _best_direction(a, b, bucket_a, bucket_b, lag_grid_seconds, bucket_seconds)
            bwd = _best_direction(b, a, bucket_b, bucket_a, lag_grid_seconds, bucket_seconds)
            candidates = [c for c in (fwd, bwd) if c is not None]
            if not candidates:
                results.append(result)
                continue
            winner = max(candidates, key=lambda c: abs(c.r))
            p_raw = time_shift_pvalue(
                winner.xs, winner.ys, winner.r, n_surrogates=n_surrogates, rng=rng
            )
            # Bonferroni correction for the implicit multiple comparisons within
            # this one pair's own winner search -- both directions x the full lag
            # grid. Deliberately still per-pair and conservative for that narrow
            # search; it is NOT a correction across the whole channel x channel
            # matrix -- see _apply_fdr_correction below for that.
            #
            # p_raw is continuous past the surrogate rank floor for pairs with
            # enough independent samples (GPD tail fit, see time_shift_pvalue),
            # so this product has no resolution floor
            # that could starve _apply_fdr_correction's rank-dependent BH
            # threshold. Two earlier floors did exactly that: 10/201 (fixed
            # 2026-07-29 by raising n_surrogates) and 10/2001 = 0.005 vs
            # 0.05/171 (fixed 2026-11-10 by dropping the rank p) -- see
            # scripts/causal_geometry_report.py's "p-value resolution" note.
            n_comparisons = 2 * len(lag_grid_seconds)
            p_value = min(1.0, p_raw * n_comparisons)
            result.winner = winner
            result.p_value = p_value
            results.append(result)

    _apply_fdr_correction(results, alpha=alpha)
    return results


def _apply_fdr_correction(results: List["PairResult"], *, alpha: float) -> None:
    """Benjamini-Hochberg step-up FDR correction across every pair actually
    tested this tick (i.e. every `PairResult` with a non-None `p_value` --
    pairs skipped for insufficient data or no surrogate-eligible direction
    never reach here and stay `significant=False`).

    Family size is the number of pairs tested in this run, not a fixed
    constant -- so it moves whenever the channel set changes (e.g. removing
    `drive:*` channels or adding `bus_synaptic:*` shrinks/grows the pair
    count and therefore the correction). This replaces a plain per-pair
    `p_value < alpha` cutoff, which does not control the false discovery
    rate across the whole matrix: with N channels there are
    `N*(N-1)/2` pairs tested per tick, and a flat per-pair alpha lets the
    expected number of false positives grow with that count. BH controls
    the *expected proportion* of false discoveries among declared-significant
    edges instead, at the same target `alpha`.
    """
    tested = [r for r in results if r.p_value is not None]
    m = len(tested)
    if m == 0:
        return
    tested.sort(key=lambda r: r.p_value)
    largest_k = 0
    for i, result in enumerate(tested, start=1):
        if result.p_value <= (i / m) * alpha:
            largest_k = i
    for i, result in enumerate(tested, start=1):
        result.significant = i <= largest_k


def _bucket_idx_to_datetime(idx: int, bucket_seconds: int = BUCKET_SECONDS) -> datetime:
    return datetime.fromtimestamp(idx * bucket_seconds, tz=timezone.utc)


def build_edges(
    pair_results: List[PairResult],
    *,
    window_start: datetime,
    window_end: datetime,
    bucket_seconds: int = BUCKET_SECONDS,
) -> List[CausalGeometryEdgeV1]:
    edges: List[CausalGeometryEdgeV1] = []
    for result in pair_results:
        if not result.significant or result.winner is None or result.p_value is None:
            continue
        w = result.winner
        edges.append(
            CausalGeometryEdgeV1(
                source_id=w.source_id,
                target_id=w.target_id,
                lag_sec=w.lag_sec,
                strength=max(-1.0, min(1.0, w.r)),
                significance=result.p_value,
                n_samples=w.n,
                window_start=window_start,
                window_end=window_end,
            )
        )
    return edges


# --- designed topology loading + divergence ---------------------------------


def load_field_topology(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def build_capability_channel_index(topology: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Reverse index: capability channel name (e.g. "reasoning_pressure") ->
    list of YAML edges (source_id, target_id, weight, edge_type) whose
    channel_map maps some local channel onto that capability channel name."""
    index: Dict[str, List[Dict[str, Any]]] = {}
    for edge in topology.get("edges", []) or []:
        channel_map = edge.get("channel_map") or {}
        for _local_channel, cap_channel in channel_map.items():
            index.setdefault(cap_channel, []).append(
                {
                    "source_id": edge.get("source_id"),
                    "target_id": edge.get("target_id"),
                    "weight": edge.get("weight"),
                    "edge_type": edge.get("edge_type"),
                }
            )
    return index


def _best_strength_for_channel(channel_id: str, edges: List[CausalGeometryEdgeV1]) -> Optional[float]:
    best: Optional[float] = None
    for edge in edges:
        if edge.source_id == channel_id or edge.target_id == channel_id:
            if best is None or abs(edge.strength) > abs(best):
                best = edge.strength
    return best


def build_divergence(
    edges: List[CausalGeometryEdgeV1],
    pair_results: List[PairResult],
    topology: Dict[str, Any],
    *,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> List[CausalGeometryDivergenceEntryV1]:
    """Compare observed edges against `config/field/orion_field_topology.v1.yaml`.

    See `scripts/causal_geometry_report.py`'s module docstring's "Designed-vs-
    observed divergence" section for the full matching rule. Summary: an
    observed channel's suffix (after the first ':') is checked against every
    capability channel name used in the YAML's edge channel_maps; an exact
    string match is treated as a plausible alias. Divergence entries for YAML
    edges get `#<capability_channel>` appended to target_id to disambiguate the
    case where two different capability channels point at the same
    (source_id, target_id) YAML edge -- callers matching against the physical
    lattice edge (e.g. `orion/substrate/field_topology_plasticity.py`) must
    strip that suffix before joining.
    """
    cap_index = build_capability_channel_index(topology)
    entries: List[CausalGeometryDivergenceEntryV1] = []
    consumed_cap_edges: set = set()

    observed_channel_ids = {edge.source_id for edge in edges} | {edge.target_id for edge in edges}
    for channel_id in sorted(observed_channel_ids):
        suffix = channel_id.split(":", 1)[1] if ":" in channel_id else channel_id
        if suffix not in cap_index:
            continue
        observed_strength = _best_strength_for_channel(channel_id, edges)
        for designed_edge in cap_index[suffix]:
            key = (designed_edge["source_id"], designed_edge["target_id"], suffix)
            if key in consumed_cap_edges:
                continue
            consumed_cap_edges.add(key)
            designed_weight = designed_edge["weight"]
            entries.append(
                CausalGeometryDivergenceEntryV1(
                    source_id=designed_edge["source_id"],
                    target_id=f"{designed_edge['target_id']}#{suffix}",
                    observed_strength=observed_strength,
                    designed_weight=designed_weight,
                    delta=(observed_strength - designed_weight) if observed_strength is not None else None,
                    status="both" if observed_strength is not None else "designed_only",
                )
            )

    # Every remaining YAML edge/channel entry with no observed alias at all.
    for cap_channel, designed_edge_list in cap_index.items():
        for designed_edge in designed_edge_list:
            key = (designed_edge["source_id"], designed_edge["target_id"], cap_channel)
            if key in consumed_cap_edges:
                continue
            consumed_cap_edges.add(key)
            entries.append(
                CausalGeometryDivergenceEntryV1(
                    source_id=designed_edge["source_id"],
                    target_id=f"{designed_edge['target_id']}#{cap_channel}",
                    observed_strength=None,
                    designed_weight=designed_edge["weight"],
                    delta=None,
                    status="designed_only",
                )
            )

    # Significant observed edges whose endpoints don't alias to any capability
    # channel at all.
    aliased_channel_ids = {
        cid
        for cid in observed_channel_ids
        if (cid.split(":", 1)[1] if ":" in cid else cid) in cap_index
    }
    for edge in edges:
        if edge.source_id in aliased_channel_ids or edge.target_id in aliased_channel_ids:
            continue
        entries.append(
            CausalGeometryDivergenceEntryV1(
                source_id=edge.source_id,
                target_id=edge.target_id,
                observed_strength=edge.strength,
                designed_weight=None,
                delta=None,
                status="observed_only",
            )
        )

    # Pairs that never had enough co-sampled data to test at all.
    for result in pair_results:
        if result.n_lag0 < min_samples:
            entries.append(
                CausalGeometryDivergenceEntryV1(
                    source_id=result.channel_a,
                    target_id=result.channel_b,
                    observed_strength=None,
                    designed_weight=None,
                    delta=None,
                    status="insufficient_data",
                )
            )

    return entries


# --- snapshot assembly -------------------------------------------------------


def build_snapshot(
    channels: ChannelSeries,
    topology: Dict[str, Any],
    *,
    window_start: datetime,
    window_end: datetime,
    bucket_seconds: int = BUCKET_SECONDS,
    lag_grid_seconds: Tuple[int, ...] = LAG_GRID_SECONDS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    n_surrogates: int = DEFAULT_N_SURROGATES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
    table_row_counts: Optional[Dict[str, int]] = None,
) -> Tuple[CausalGeometrySnapshotV1, List[PairResult]]:
    channel_buckets = {
        channel_id: bucketize(points, bucket_seconds) for channel_id, points in channels.items()
    }

    pair_results = compute_pairwise_results(
        channel_buckets,
        min_samples=min_samples,
        n_surrogates=n_surrogates,
        alpha=alpha,
        lag_grid_seconds=lag_grid_seconds,
        bucket_seconds=bucket_seconds,
        seed=seed,
    )
    edges = build_edges(
        pair_results, window_start=window_start, window_end=window_end, bucket_seconds=bucket_seconds
    )
    divergence = build_divergence(edges, pair_results, topology, min_samples=min_samples)

    notes: List[str] = []
    if table_row_counts:
        for table_name, row_count in table_row_counts.items():
            if row_count == 0:
                notes.append(
                    f"source table {table_name!r} returned 0 rows in the requested window -- "
                    "every channel it would have contributed is absent from this snapshot, "
                    "not silently zero-valued."
                )
    n_pairs_tested = len(pair_results)
    n_insufficient = sum(1 for r in pair_results if r.n_lag0 < min_samples)
    n_pairs_with_data = n_pairs_tested - n_insufficient
    notes.append(
        f"{len(channels)} channels, {n_pairs_tested} channel pairs considered, "
        f"{n_insufficient} below --min-samples={min_samples}, "
        f"{n_pairs_with_data} tested for significance, {len(edges)} significant edges found."
    )
    insufficient_data = len(edges) == 0
    if insufficient_data:
        if n_pairs_with_data == 0:
            notes.append(
                "No channel pair had enough co-sampled data to test at all -- widen "
                "--window-hours, lower --min-samples, or check that the source tables "
                "actually have rows in the requested window."
            )
        else:
            notes.append(
                f"{n_pairs_with_data} pair(s) had enough data to test but none cleared "
                f"--alpha={alpha} (BH-FDR; up to {n_surrogates} time-shift "
                "surrogates each, GPD tail past the last 10) -- "
                "this is reported honestly as insufficient_data rather than a fabricated graph."
            )

    snapshot = CausalGeometrySnapshotV1(
        snapshot_id=str(uuid.uuid4()),
        generated_at=datetime.now(timezone.utc),
        window_start=window_start,
        window_end=window_end,
        edges=edges,
        designed_topology_version=topology.get("schema_version"),
        divergence=divergence,
        insufficient_data=insufficient_data,
        notes=notes,
    )
    return snapshot, pair_results
