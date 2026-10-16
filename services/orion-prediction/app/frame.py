"""Raw readings -> a regular grain series with explicit gaps.

``to_grain`` buckets readings into left-closed grain buckets ``[t, t+grain)`` and aggregates them.
Empty buckets stay NaN unless they sit inside a gap no longer than ``max_fill_min``, which is filled
by time interpolation. Longer gaps stay NaN: they are never invented.

``train_ready`` produces the gap-free array a model needs (interpolating what is left) and reports
how much of it was filled, so a caller can refuse a fit that is mostly invented data.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from app.series import SeriesSpec


def floor_ts(ts: datetime, grain_min: int) -> datetime:
    epoch = int(ts.astimezone(timezone.utc).timestamp())
    step = grain_min * 60
    return datetime.fromtimestamp(epoch - epoch % step, tz=timezone.utc)


def to_grain(points: Iterable[tuple[datetime, float | None]], grain_min: int, agg: str,
             max_fill_min: int, start: datetime | None = None, end: datetime | None = None) -> pd.Series:
    """Regular UTC series indexed by bucket start, covering [start, end) (defaults: data span)."""
    rows = [(t, v) for t, v in points if v is not None and np.isfinite(v)]
    freq = f"{grain_min}min"
    if not rows and (start is None or end is None):
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    index = pd.DatetimeIndex([t for t, _ in rows]) if rows else pd.DatetimeIndex([], tz="UTC")
    raw = pd.Series([v for _, v in rows], index=index.tz_convert("UTC"), dtype=float).sort_index()
    lo = pd.Timestamp(floor_ts(start, grain_min)) if start else raw.index[0].floor(freq)
    hi = pd.Timestamp(floor_ts(end, grain_min)) if end else raw.index[-1].floor(freq) + pd.Timedelta(freq)
    raw = raw[(raw.index >= lo) & (raw.index < hi)]
    binned = getattr(raw.resample(freq, origin="epoch"), agg)() if len(raw) else pd.Series(dtype=float)
    idx = pd.date_range(lo, hi, freq=freq, inclusive="left", tz="UTC")
    out = binned.reindex(idx)
    limit = max_fill_min // grain_min
    if limit > 0 and out.notna().any():
        filled = out.interpolate(method="time", limit=limit, limit_area="inside")
        # interpolate(limit=) fills the first `limit` NaNs of a longer gap; undo that so long gaps stay whole.
        isna = out.isna()
        run_id = (isna != isna.shift()).cumsum()
        run_len = isna.groupby(run_id).transform("sum")
        out = filled.where(~(isna & (run_len > limit)), np.nan)
    return out


def mask_windows(series: pd.Series, windows: Sequence[tuple[datetime, datetime]]) -> pd.Series:
    out = series.copy()
    for start, end in windows:
        out[(out.index >= pd.Timestamp(start)) & (out.index < pd.Timestamp(end))] = np.nan
    return out


def train_ready(series: pd.Series) -> tuple[np.ndarray, float]:
    """Gap-free values for fitting, plus the share of points that had to be filled."""
    if series.empty:
        return np.array([], dtype=float), 1.0
    filled_share = float(series.isna().mean())
    vals = series.interpolate(method="linear", limit_direction="both").to_numpy(dtype=float)
    return vals, filled_share


def training_slice(series: pd.Series, spec: SeriesSpec, end: pd.Timestamp | None = None) -> pd.Series:
    """The model's training view: masked exclusions, after train_from, at most train_window long."""
    s = mask_windows(series, spec.exclude)
    if end is not None:
        s = s[s.index < end]
    if spec.train_from is not None:
        s = s[s.index >= pd.Timestamp(spec.train_from)]
    if len(s):
        s = s[s.index >= s.index[-1] - pd.Timedelta(minutes=spec.train_window_min) + pd.Timedelta(minutes=spec.grain_min)]
    # Leading NaNs carry no information and would be back-filled into invented history.
    first = s.first_valid_index()
    return s[s.index >= first] if first is not None else s.iloc[0:0]
