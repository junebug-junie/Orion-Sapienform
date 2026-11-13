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

    def missing_weather_hours(self, start, end):
        out, t = [], start.replace(minute=0, second=0, microsecond=0)
        while t < end:
            if t not in self.weather or self.weather[t][1]:
                out.append(t)
            t += timedelta(hours=1)
        return out

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


def _fake_weather(monkeypatch, now, hist_fails=False):
    from app import weather
    calls = []

    def hours(a, b):
        return pd.date_range(a, b, freq="1h", inclusive="left")

    def fake_hist(lat, lon, start, end, hourly):
        calls.append("history")
        if hist_fails:
            raise TimeoutError("historical-forecast-api down")
        h = hours(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1))
        return {"timezone": "UTC", "hourly": {"time": [x.strftime("%Y-%m-%dT%H:%M") for x in h], "temperature_2m": [60.0] * len(h)}}

    def fake_forecast(lat, lon, **kw):
        calls.append("forecast")
        h = hours(pd.Timestamp(now).floor("1h") - pd.Timedelta(days=7), pd.Timestamp(now).floor("1h") + pd.Timedelta(days=2))
        return {"timezone": "UTC", "hourly": {"time": [x.strftime("%Y-%m-%dT%H:%M") for x in h], "temperature_2m": [65.0] * len(h)}}
    monkeypatch.setattr(weather.openmeteo, "historical_forecast", fake_hist)
    monkeypatch.setattr(weather.openmeteo, "forecast", fake_forecast)
    return weather, calls


def test_weather_refresh_backfills_holes_once_and_marks_future_hours_as_forecast(monkeypatch):
    now = datetime(2026, 10, 20, 12, 30, tzinfo=timezone.utc)
    weather, calls = _fake_weather(monkeypatch, now)
    st = Store({})
    out = weather.refresh(st, 0.0, 0.0, now, datetime(2026, 9, 26, tzinfo=timezone.utc))
    assert calls == ["forecast", "history"] and out["backfilled"]["hours"] == 420   # 09-26 00:00 -> 10-13 12:00
    assert st.weather[datetime(2026, 10, 21, 0, tzinfo=timezone.utc)][1] is True       # future hour = forecast
    assert st.weather[datetime(2026, 10, 19, 0, tzinfo=timezone.utc)][1] is False      # past hour = observed
    calls.clear()
    weather.refresh(st, 0.0, 0.0, now + timedelta(hours=1), datetime(2026, 9, 26, tzinfo=timezone.utc))
    assert calls == ["forecast"]                                                         # nothing missing now
    del st.weather[datetime(2026, 10, 1, 6, tzinfo=timezone.utc)]                         # an interior hole
    calls.clear()
    weather.refresh(st, 0.0, 0.0, now + timedelta(hours=2), datetime(2026, 9, 26, tzinfo=timezone.utc))
    assert calls == ["forecast", "history"]


def test_a_failed_backfill_still_stores_the_forecast(monkeypatch):
    now = datetime(2026, 10, 20, 12, 30, tzinfo=timezone.utc)
    weather, calls = _fake_weather(monkeypatch, now, hist_fails=True)
    st = Store({})
    out = weather.refresh(st, 0.0, 0.0, now, datetime(2026, 9, 26, tzinfo=timezone.utc))
    assert out["ok"] and "down" in out["backfill_error"] and len(st.weather) > 200


def test_empty_weather_location_means_off_not_a_boot_crash(monkeypatch):
    from app.settings import Settings
    monkeypatch.setenv("ORION_SITUATION_WEATHER_LAT", "")
    monkeypatch.setenv("ORION_SITUATION_WEATHER_LON", "")
    s = Settings()
    assert s.weather_lat is None and s.weather_lon is None


def test_the_newest_weather_hour_covers_the_rest_of_that_hour():
    # Serving at 12:37: the newest lagged value is 12:00's; buckets 12:05..12:35 must still have it.
    reg = _reg()
    st = Store({"x.proxy": [(T0 + timedelta(hours=k), 85.0) for k in range(10)]})     # lagged -> 03:00..12:00
    end = T0 + timedelta(hours=12, minutes=40)
    idx = pd.date_range(T0 + timedelta(hours=10), end, freq="5min", tz="UTC", inclusive="left")
    s = covariate_series(st, reg.get("x.proxy"), idx[0].to_pydatetime(), end, idx)
    assert s.notna().all() and np.allclose(s, 10.0)



def test_first_weather_refresh_runs_before_the_selection_later_ones_after(monkeypatch):
    import asyncio
    import app.engine as engine_mod
    from app.engine import Engine
    from tests.test_engine import Clock, FakeStore, _setup, _sync

    reg, spec, s, store, end = _setup()
    order = []
    eng = Engine(reg, store, backtest_windows=14, backtest_step_min=180, select_interval_sec=21600, settle_sec=60,
                 publish_enabled=False, clock=Clock(end + timedelta(seconds=90)),
                 weather={"lat": 0.0, "lon": 0.0, "refresh_sec": 3600})

    def fake_refresh(now):
        order.append("weather")
        eng.weather_state = {"ok": True, "next_at": now + timedelta(hours=1)}
        return eng.weather_state
    eng.refresh_weather = fake_refresh
    real_select = eng.select
    eng.select = lambda sp, now: (order.append("select"), real_select(sp, now))[1]
    monkeypatch.setattr(engine_mod, "build_models", lambda sp, u: {k: v for k, v in
                        __import__("app.models", fromlist=["build_models"]).build_models(sp, u).items()
                        if k in ("naive", "seasonal_naive")})
    asyncio.run(eng.tick(_sync))
    assert order[:2] == ["weather", "select"]
    eng.clock.t += timedelta(hours=1, minutes=1)
    eng.state["test.temp"].next_select_at = eng.clock.t          # both due on the same tick
    order.clear()
    asyncio.run(eng.tick(_sync))
    assert order == ["select", "weather"]


class _Conn:
    def __init__(self, log, rows):
        self.log, self.rows = log, rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def transaction(self):
        return self

    def execute(self, sql, args=()):
        self.log.append((sql, list(args)))
        return self

    def fetchall(self):
        return self.rows


class _Pool:
    def __init__(self, rows):
        self.log, self.rows = [], rows

    def connection(self):
        return _Conn(self.log, self.rows)


def _office_store(rows):
    from app.store import PostgresStore
    st = PostgresStore.__new__(PostgresStore)
    st.pool = _Pool(rows)
    return st


def test_office_env_source_reads_only_live_fresh_rows_read_only_with_a_timeout():
    reg = parse_registry(spec_doc(source={"kind": "office_env", "node": "pan-1", "key": "office_temp_c"}))
    sp = reg.get("test.temp")
    st = _office_store([{"ts": T0, "v": 26.1}])
    assert st.points(sp, T0, T0 + timedelta(hours=1)) == [(T0, 26.1)]
    sqls = [s for s, _ in st.pool.log]
    assert sqls[0] == "SET TRANSACTION READ ONLY" and sqls[1].startswith("SET LOCAL statement_timeout = ")
    sql, args = st.pool.log[2]
    assert "FROM office_env_sample" in sql and "office_temp_c AS v" in sql
    assert "sensor_online AND NOT stale AND office_temp_c IS NOT NULL" in sql
    assert args == ["pan-1", T0, T0 + timedelta(hours=1)]


def test_office_env_store_refuses_a_key_outside_the_allowlist_even_if_the_spec_was_built_by_hand():
    import dataclasses

    import pytest

    from app.series import Source
    sp = parse_registry(spec_doc(source={"kind": "office_env", "node": "pan-1", "key": "office_temp_c"})).get("test.temp")
    bad = dataclasses.replace(sp, source=Source(kind="office_env", node="pan-1", key="payload_json"))
    st = _office_store([])
    with pytest.raises(ValueError, match="office_env key"):
        st.points(bad, T0)
    assert st.pool.log == []
