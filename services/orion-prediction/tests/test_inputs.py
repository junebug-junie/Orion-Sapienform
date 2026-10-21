"""Derived covariates (lag, above, zero_before_first), the cluster/weather sources, and weather refresh."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from app.frame import covariate_series, load_points
from app.series import parse_registry
from tests.helpers import T0, spec_doc


class Store:
    def __init__(self, series_points):
        self.p = series_points
        self.weather = {}

    def points(self, spec, since, until=None):
        return [(t, v) for t, v in self.p.get(spec.series_id, []) if t >= since and (until is None or t < until)]

    def first_reading(self, spec):
        pts = [t for t, v in self.p.get(spec.series_id, []) if v is not None]
        return min(pts) if pts else None

    def weather_span(self):
        if not self.weather:
            return None, None
        obs = [t for t, (v, fc) in self.weather.items() if not fc]
        return min(self.weather), (max(obs) if obs else None)

    def upsert_weather(self, rows, fetched_at):
        for t, v, fc, src in rows:
            if t in self.weather and not self.weather[t][1] and fc:
                continue                      # a forecast never replaces an observation
            self.weather[t] = (v, fc)
        return len(rows)


def _reg():
    doc = spec_doc(covariates=["x.proxy", "x.hecate"])
    doc["series"] += [
        {"series_id": "x.proxy", "role": "covariate", "source": {"kind": "weather", "key": "temp_f"}, "unit": "F",
         "grain": "5min", "max_fill": "60min", "lag": "3h", "above": 75, "consumers": ["t"]},
        {"series_id": "x.hecate", "role": "covariate", "source": {"kind": "cluster", "node": "hecate", "key": "chassis_watts"},
         "unit": "W", "grain": "5min", "zero_before_first": True, "consumers": ["t"]},
    ]
    return parse_registry(doc)


def test_lag_and_above_shift_and_hinge_the_source():
    reg = _reg()
    hours = [T0 + timedelta(hours=k) for k in range(10)]
    st = Store({"x.proxy": [(h, 70.0 + 2 * k) for k, h in enumerate(hours)]})   # 70, 72, ..., 88 F
    pts = load_points(st, reg.get("x.proxy"), T0 + timedelta(hours=3), T0 + timedelta(hours=10))
    assert pts[0] == (T0 + timedelta(hours=3), 0.0)          # value at 03:00 = source at 00:00 (70 F) -> 0
    assert pts[-1] == (T0 + timedelta(hours=9), 7.0)         # value at 09:00 = source at 06:00 (82 F) -> 7


def test_hourly_weather_is_spread_over_5min_buckets():
    reg = _reg()
    hours = [T0 + timedelta(hours=k) for k in range(8)]
    st = Store({"x.proxy": [(h, 80.0) for h in hours]})
    idx = pd.date_range(T0 + timedelta(hours=4), periods=24, freq="5min", tz="UTC")
    s = covariate_series(st, reg.get("x.proxy"), idx[0].to_pydatetime(), (idx[-1] + pd.Timedelta(minutes=5)).to_pydatetime(), idx)
    assert s.notna().all() and np.allclose(s, 5.0)


def test_zero_before_first_reading_but_later_gaps_stay_gaps():
    reg = _reg()
    first = T0 + timedelta(hours=1)
    pts = [(first + timedelta(minutes=5 * k), 96.0) for k in range(12)]
    pts = [p for i, p in enumerate(pts) if not 6 <= i < 10]            # a 20-min gap after it started
    st = Store({"x.hecate": pts})
    idx = pd.date_range(T0, periods=24, freq="5min", tz="UTC")
    s = covariate_series(st, reg.get("x.hecate"), T0, T0 + timedelta(hours=2), idx)
    assert (s[s.index < first] == 0).all()
    assert s[(s.index >= first + timedelta(minutes=30)) & (s.index < first + timedelta(minutes=50))].isna().all()


def test_weather_refresh_backfills_once_and_marks_future_hours_as_forecast(monkeypatch):
    from app import weather

    now = datetime(2026, 10, 20, 12, 30, tzinfo=timezone.utc)
    calls = []

    def fake_archive(lat, lon, start, end, hourly):
        calls.append(("archive", start, end))
        hrs = pd.date_range(start, end + timedelta(days=1), freq="1h", inclusive="left")
        return {"timezone": "UTC", "hourly": {"time": [h.strftime("%Y-%m-%dT%H:%M") for h in hrs], "temperature_2m": [60.0] * len(hrs)}}

    def fake_forecast(lat, lon, **kw):
        calls.append(("forecast", kw["past_days"], kw["forecast_days"]))
        hrs = pd.date_range(now - timedelta(days=7), now + timedelta(days=2), freq="1h").floor("1h")
        return {"timezone": "UTC", "hourly": {"time": [h.strftime("%Y-%m-%dT%H:%M") for h in hrs], "temperature_2m": [65.0] * len(hrs)}}
    monkeypatch.setattr(weather.openmeteo, "archive", fake_archive)
    monkeypatch.setattr(weather.openmeteo, "forecast", fake_forecast)
    st = Store({})
    out = weather.refresh(st, 40.0, -111.0, now, datetime(2026, 9, 26, tzinfo=timezone.utc))
    assert out["backfilled_from"] and [c[0] for c in calls] == ["archive", "forecast"]
    assert st.weather[datetime(2026, 10, 21, 0, tzinfo=timezone.utc)][1] is True       # future hour = forecast
    assert st.weather[datetime(2026, 10, 19, 0, tzinfo=timezone.utc)][1] is False      # past hour = observed
    calls.clear()
    weather.refresh(st, 40.0, -111.0, now + timedelta(hours=1), datetime(2026, 9, 26, tzinfo=timezone.utc))
    assert [c[0] for c in calls] == ["forecast"]                                         # no second backfill


def test_the_newest_weather_hour_covers_the_rest_of_that_hour():
    # Serving at 12:37: the newest lagged value is 12:00's; buckets 12:05..12:35 must still have it.
    reg = _reg()
    st = Store({"x.proxy": [(T0 + timedelta(hours=k), 85.0) for k in range(10)]})     # lagged -> 03:00..12:00
    end = T0 + timedelta(hours=12, minutes=40)
    idx = pd.date_range(T0 + timedelta(hours=10), end, freq="5min", tz="UTC", inclusive="left")
    s = covariate_series(st, reg.get("x.proxy"), idx[0].to_pydatetime(), end, idx)
    assert s.notna().all() and np.allclose(s, 10.0)
