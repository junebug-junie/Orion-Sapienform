from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from app.frame import mask_windows, to_grain, train_ready, training_slice
from tests.helpers import T0, spec


def _pts(minutes):
    return [(T0 + timedelta(minutes=m), float(m)) for m in minutes]


def test_buckets_are_left_closed_means():
    s = to_grain(_pts([0, 1, 4, 5, 9]), 5, "mean", 0)
    assert s.iloc[0] == np.mean([0, 1, 4]) and s.iloc[1] == np.mean([5, 9])


def test_short_gap_filled_long_gap_kept_whole():
    # buckets 0..3 present, 4..5 missing (10 min <= 15), 6 present, 7..11 missing (25 min > 15), 12 present
    mins = [0, 5, 10, 15, 30, 60]
    s = to_grain(_pts(mins), 5, "mean", 15)
    assert s.iloc[4:6].notna().all()
    # The long gap must not be partially filled (interpolate(limit=) would fill its first 3 buckets).
    assert s.iloc[7:12].isna().all()


def test_explicit_window_reindexes_with_nan():
    s = to_grain(_pts([10]), 5, "mean", 0, start=T0, end=T0 + timedelta(minutes=30))
    assert len(s) == 6 and s.notna().sum() == 1


def test_train_ready_reports_filled_share():
    s = pd.Series([1.0, np.nan, 3.0, np.nan], index=pd.date_range(T0, periods=4, freq="5min"))
    y, share = train_ready(s)
    assert share == 0.5 and np.all(np.isfinite(y))


def test_training_slice_applies_exclude_train_from_and_window():
    sp = spec(train_window="1d", train_from="2026-09-02T00:00:00Z",
              exclude=[{"start": "2026-09-03T00:00:00Z", "end": "2026-09-03T01:00:00Z", "reason": "x"}])
    idx = pd.date_range(T0, periods=4 * 288, freq="5min", tz="UTC")
    s = pd.Series(1.0, index=idx)
    out = training_slice(s, sp)
    assert len(out) == 288 and out.index[0] >= pd.Timestamp("2026-09-02", tz="UTC")
    masked = mask_windows(s, sp.exclude)
    assert masked[(masked.index >= "2026-09-03 00:00Z") & (masked.index < "2026-09-03 01:00Z")].isna().all()


def test_no_readings_in_an_explicit_window_is_all_nan_not_a_crash():
    # Sensor outage during a scoring window: the reducer must mark buckets missing, not raise.
    s = to_grain([], 5, "mean", 15, start=T0, end=T0 + timedelta(minutes=20))
    assert len(s) == 4 and s.isna().all()
