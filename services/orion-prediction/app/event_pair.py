"""Does event A raise the rate of event B within a lag window? (observational, placebo-calibrated)

Statistic. Each A at time ``a`` opens a window ``[a + lag_min, a + lag_max)``, clipped to the time
range. Overlapping windows are merged first, so two A's close together do not count the same B
twice. Then:

    observed  = number of B inside the merged windows
    expected  = (all B in the range / range length) x merged window length
    rate_ratio = observed / expected

1.0 means B is no more common after A than anywhere else in the range.

p-value. Placebo by circular shift: move every A by the same random offset (wrapping round the end
of the range), keeping A's own spacing and B untouched, and recompute the ratio. Offsets are drawn
from ``[lag_max, span - lag_max]`` so no placebo reproduces the real alignment within the lag
window (circular distance). ``p = (1 + #placebo ratio >= real) / (n_placebo + 1)``, one-sided ("A raises B").

Two placebos. ``same_time_of_day`` (default) shifts A by whole days plus a jitter of up to
``TOD_JITTER_SEC``, so every placebo A keeps its time of day. ``circular`` shifts A anywhere. The
free shift moves A to other hours and then reads "both happen in the daytime" as an effect: with A
and B independent but both daytime-only, it gives p <= 0.05 in ~38% of seeds versus ~6% for the
time-of-day placebo (tests/test_event_pair.py). Street events follow the day, so the default keeps
the daily rhythm. ``same_time_of_day`` needs lag_max < 30 min; use ``circular`` for longer lags
on series with no daily rhythm. With ``same_time_of_day`` the range is cut to its last whole day
(events after it are dropped): a wrap round a part-day range would land at another time of day.

``claim_kind`` is always ``observational_effect``: a common cause of A and B (time of day, one
person who triggers both cameras) produces the same number. Rows are counted, not visits: a burst
of B rows inflates ``observed`` and ``expected`` alike, and the placebo keeps B's bursts, so the
p-value stays honest while the ratio reads larger than a per-visit ratio would.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CLAIM_KIND = "observational_effect"
MIN_A_EVENTS = 5
TOD_JITTER_SEC = 3600.0
PLACEBOS = ("circular", "same_time_of_day")
TOD_MIN_DAYS = 7          # fewer whole days give too few distinct day shifts for a fine p-value
MAX_PLACEBO_WORK = 50_000_000   # n_placebo x (n_a + n_b); above this the request is refused
MIN_SPAN_LAGS = 10   # the range must hold >= 10 lag windows, or the placebo shifts barely differ


class EventPairRefused(ValueError):
    """Not enough evidence to answer. Carries the counts so the caller sees why."""

    def __init__(self, reason: str, **detail):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


@dataclass
class PairStat:
    observed: int
    expected: float
    covered_sec: float
    rate_ratio: float


@dataclass
class EventPairResult:
    claim_kind: str
    n_a: int
    n_b: int
    span_sec: float
    lag_min_sec: float
    lag_max_sec: float
    observed_b_in_windows: int
    expected_b_in_windows: float
    covered_sec: float
    rate_in_windows_per_hour: float
    baseline_rate_per_hour: float
    rate_ratio: float
    p_value: float
    n_placebo: int
    placebo_ratio_quantiles: dict = field(default_factory=dict)
    seed: int = 0
    placebo: str = "same_time_of_day"


def _offsets(rng: np.random.Generator, n: int, span: float, lag_max: float, placebo: str) -> np.ndarray:
    """Placebo shifts whose circular distance from the real alignment is > lag_max."""
    if placebo == "circular":
        return rng.uniform(lag_max, span - lag_max, size=n)
    days = int(span // 86400.0)
    out = np.empty(0)
    for _ in range(50):   # rejection sampling; the accepted fraction is >= 1 - lag_max / TOD_JITTER_SEC
        d = np.mod(rng.integers(0, days, size=n) * 86400.0 + rng.uniform(-TOD_JITTER_SEC, TOD_JITTER_SEC, n), span)
        out = np.concatenate([out, d[np.minimum(d, span - d) > lag_max]])
        if len(out) >= n:
            return out[:n]
    return out


def _merged(starts: np.ndarray, width: float, span: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Sorted window starts of equal width -> (starts, running max end, union length), clipped to [0, span)."""
    s = np.clip(np.sort(starts), 0.0, span)
    e = np.clip(s + width, 0.0, span)
    e = np.maximum.accumulate(e)              # equal widths keep ends sorted; clipping keeps them monotone
    prev = np.concatenate(([-np.inf], e[:-1]))
    covered = float(np.sum(np.maximum(0.0, e - np.maximum(s, prev))))
    return s, e, covered


def pair_stat(a: np.ndarray, b_sorted: np.ndarray, span: float, lag_min: float, lag_max: float) -> PairStat:
    """``a``, ``b_sorted``: seconds since the range start, all inside [0, span)."""
    s, e, covered = _merged(a + lag_min, lag_max - lag_min, span)
    if len(s) == 0 or len(b_sorted) == 0:
        observed = 0
    else:
        # The last window starting at or before b has the furthest end of all windows that start
        # by then (ends are monotone), so b is covered iff it falls before that end.
        idx = np.searchsorted(s, b_sorted, side="right") - 1
        ok = idx >= 0
        observed = int(np.sum(b_sorted[ok] < e[idx[ok]]))
    expected = len(b_sorted) / span * covered if span > 0 else 0.0
    ratio = observed / expected if expected > 0 else float("nan")
    return PairStat(observed, expected, covered, ratio)


def event_pair_effect(a: np.ndarray, b: np.ndarray, span: float, lag_min: float, lag_max: float,
                      n_placebo: int = 999, seed: int = 0, min_a: int = MIN_A_EVENTS,
                      placebo: str = "same_time_of_day") -> EventPairResult:
    """Event times in seconds since the range start; anything outside [0, span) is dropped."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    days = int(span // 86400.0)
    if placebo == "same_time_of_day" and days >= TOD_MIN_DAYS:
        span = days * 86400.0
    a = a[(a >= 0) & (a < span)]
    b = np.sort(b[(b >= 0) & (b < span)])
    counts = {"n_a": int(len(a)), "n_b": int(len(b)), "span_sec": float(span)}
    if not (0 <= lag_min < lag_max):
        raise EventPairRefused("need 0 <= lag_min < lag_max", **counts)
    if span < MIN_SPAN_LAGS * lag_max:
        raise EventPairRefused(f"time range must be >= {MIN_SPAN_LAGS} x lag_max", **counts)
    if len(a) < min_a:
        raise EventPairRefused(f"event A has {len(a)} occurrence(s) in range; need >= {min_a}", **counts)
    if len(b) == 0:
        raise EventPairRefused("event B never occurs in range; there is no baseline rate", **counts)
    if placebo not in PLACEBOS:
        raise EventPairRefused(f"placebo must be one of {PLACEBOS}", **counts)
    if placebo == "same_time_of_day" and (days < TOD_MIN_DAYS or lag_max >= TOD_JITTER_SEC / 2):
        raise EventPairRefused(f"same_time_of_day needs >= {TOD_MIN_DAYS} days of range and lag_max < 30 min",
                               **counts)
    if n_placebo * (len(a) + len(b)) > MAX_PLACEBO_WORK:
        raise EventPairRefused(f"n_placebo x events exceeds {MAX_PLACEBO_WORK:,}; narrow the range or lower "
                               "n_placebo", **counts)
    if n_placebo < 1:
        raise EventPairRefused("n_placebo must be >= 1", **counts)
    real = pair_stat(a, b, span, lag_min, lag_max)
    if not real.covered_sec > 0:
        raise EventPairRefused("A's lag windows fall outside the time range", **counts)
    rng = np.random.default_rng(seed)
    offsets = _offsets(rng, n_placebo, span, lag_max, placebo)
    ratios = np.array([pair_stat(np.mod(a + d, span), b, span, lag_min, lag_max).rate_ratio for d in offsets])
    ratios = ratios[np.isfinite(ratios)]
    if len(ratios) == 0:
        raise EventPairRefused("no placebo shift produced a finite ratio", **counts)
    hits = int(np.sum(ratios >= real.rate_ratio - 1e-12))
    p = (1 + hits) / (1 + len(ratios))
    q = np.quantile(ratios, [0.05, 0.5, 0.95])
    hour = 3600.0
    return EventPairResult(
        claim_kind=CLAIM_KIND, n_a=len(a), n_b=len(b), span_sec=float(span), lag_min_sec=float(lag_min),
        lag_max_sec=float(lag_max), observed_b_in_windows=real.observed, expected_b_in_windows=real.expected,
        covered_sec=real.covered_sec, rate_in_windows_per_hour=real.observed / real.covered_sec * hour,
        baseline_rate_per_hour=len(b) / span * hour, rate_ratio=real.rate_ratio, p_value=float(p),
        n_placebo=int(len(ratios)), placebo_ratio_quantiles={"q05": float(q[0]), "q50": float(q[1]),
                                                               "q95": float(q[2])},
        seed=int(seed), placebo=placebo,
    )
