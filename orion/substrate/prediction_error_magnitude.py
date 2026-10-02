"""How big is a substrate node's prediction error, against its own past?

Pure function: a node's stored reading history -> ``PredictionErrorMagnitudeV1``.
No I/O, no clock reads (``now`` is passed in), deterministic for a given input.

Spec: docs/superpowers/specs/2026-10-02-reverie-prediction-error-magnitude-proposal.md
(approved 2026-10-02, step 1).

Rules, each pinned by a test in tests/test_prediction_error_magnitude.py:

- Percentiles, not mean/SD z-scores. Several domains are 75-93% exact zeros;
  mean/SD there describe a reading that almost never happens, and mean(|z|)
  has a nonzero rest floor (AGENTS.md metric gate, 2026-07-26).
- ``percentile_now`` = share of 7-day readings STRICTLY below the current
  value, so an all-zero history with current 0 reads 0.0 (rest).
- Below ``min_readings`` (default 200) 7-day readings, ``band`` and ``trend``
  are ``insufficient_history``.
- ``trend`` describes what HAS happened: median of the last hour minus the
  median of the 24 h before that. Rising/settling only when
  |delta| > max(trend_min_delta, 0.5 * (p90_7d - p50_7d)).

This module must NOT import or reuse
``orion.substrate.prediction_error_trend.compute_prediction_error_trend``:
that function deliberately flips the sign to FORECAST reversion
(prior-half minus recent-half), so reusing it here would tell Orion an error
is rising while it has been falling. A regression test asserts the import
stays absent.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Iterable, Sequence

from orion.schemas.attention_frame import PredictionErrorMagnitudeV1

WINDOW_7D = timedelta(days=7)
WINDOW_24H = timedelta(hours=24)
WINDOW_1H = timedelta(hours=1)

DEFAULT_MIN_READINGS = 200
DEFAULT_TREND_MIN_DELTA = 0.01
# Band cut points on percentile_now: < 0.5 quiet, < 0.9 usual, < 0.99 high,
# otherwise unusual. Knobs, not findings (spec, metric gate item 2).
DEFAULT_BAND_CUTS: tuple[float, float, float] = (0.5, 0.9, 0.99)


def _aware(ts: datetime) -> datetime:
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


def _quantile(sorted_values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated quantile (numpy's default) over pre-sorted values."""
    n = len(sorted_values)
    if n == 0:
        return None
    if n == 1:
        return float(sorted_values[0])
    pos = (n - 1) * q
    lo = math.floor(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac)


def _median(values: Iterable[float]) -> float | None:
    return _quantile(sorted(values), 0.5)


def _round(x: float | None) -> float | None:
    return None if x is None else round(float(x), 6)


def compute_prediction_error_magnitude(
    *,
    value: float,
    observed_at: datetime,
    history: Sequence[tuple[datetime, float]],
    now: datetime,
    trend_min_delta: float = DEFAULT_TREND_MIN_DELTA,
    min_readings: int = DEFAULT_MIN_READINGS,
    band_cuts: tuple[float, float, float] = DEFAULT_BAND_CUTS,
) -> PredictionErrorMagnitudeV1:
    """Summarize one node's current reading against its stored history.

    ``history`` is ``(observed_at, value)`` pairs for this node, any order,
    normally already including the current reading. Readings older than the
    7-day window ending at ``now`` are ignored.
    """
    now = _aware(now)
    observed_at = _aware(observed_at)
    value = float(value)
    age_sec = max(0.0, (now - observed_at).total_seconds())

    readings: list[tuple[datetime, float]] = []
    for ts, v in history:
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(fv):
            continue
        ts = _aware(ts)
        # No upper bound: a reading stamped slightly after `now` (producer /
        # runtime clock skew) is still a real, current reading.
        if ts >= now - WINDOW_7D:
            readings.append((ts, fv))

    values_7d = sorted(v for _, v in readings)
    values_24h = sorted(v for ts, v in readings if ts >= now - WINDOW_24H)
    last_1h = [v for ts, v in readings if ts >= now - WINDOW_1H]
    prior_24h = [
        v for ts, v in readings if now - WINDOW_1H - WINDOW_24H <= ts < now - WINDOW_1H
    ]
    n = len(values_7d)

    p50_7d = _quantile(values_7d, 0.5)
    p90_7d = _quantile(values_7d, 0.9)
    median_1h = _median(last_1h)
    median_prior_24h = _median(prior_24h)
    percentile_now = (sum(1 for v in values_7d if v < value) / n) if n else None

    if n < max(1, int(min_readings)):
        band = "insufficient_history"
        trend = "insufficient_history"
    else:
        quiet_cut, usual_cut, high_cut = band_cuts
        pct = percentile_now or 0.0
        if pct < quiet_cut:
            band = "quiet"
        elif pct < usual_cut:
            band = "usual"
        elif pct < high_cut:
            band = "high"
        else:
            band = "unusual"

        if median_1h is None or median_prior_24h is None or p50_7d is None or p90_7d is None:
            # Nothing recorded in one of the two windows (e.g. a stale node
            # whose observed_at has not moved in the last hour): no honest
            # trajectory to describe.
            trend = "insufficient_history"
        else:
            threshold = max(float(trend_min_delta), 0.5 * (p90_7d - p50_7d))
            delta = median_1h - median_prior_24h
            if delta > threshold:
                trend = "rising"
            elif delta < -threshold:
                trend = "settling"
            else:
                trend = "flat"

    return PredictionErrorMagnitudeV1(
        value=round(value, 6),
        age_sec=round(age_sec, 3),
        p50_7d=_round(p50_7d),
        p90_7d=_round(p90_7d),
        p50_24h=_round(_quantile(values_24h, 0.5)),
        p90_24h=_round(_quantile(values_24h, 0.9)),
        percentile_now=None if percentile_now is None else round(percentile_now, 6),
        n_readings_7d=n,
        median_1h=_round(median_1h),
        median_prior_24h=_round(median_prior_24h),
        trend=trend,
        band=band,
    )
