"""The situation layer's weather goes through the shared Open-Meteo client (orion/weather/openmeteo.py)."""
from __future__ import annotations

import json
from datetime import timezone

import pytest

from orion.weather import openmeteo


class _Resp:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_forecast_builds_the_request_and_hourly_series_parses_utc(monkeypatch):
    seen = {}

    def fake(url, timeout):
        seen["url"] = url
        return _Resp({"timezone": "UTC", "hourly": {"time": ["2026-10-20T00:00", "2026-10-20T01:00"],
                                                    "temperature_2m": [61.2, None]}})
    monkeypatch.setattr(openmeteo, "urlopen", fake)
    p = openmeteo.forecast(0.0, 0.0, hourly=("temperature_2m",), past_days=7)
    assert "past_days=7" in seen["url"] and "timezone=UTC" in seen["url"] and "hourly=temperature_2m" in seen["url"]
    s = openmeteo.hourly_series(p, "temperature_2m")
    assert s[0][0].tzinfo == timezone.utc and s[0][1] == 61.2 and s[1][1] is None


def test_hourly_series_refuses_a_local_time_response():
    with pytest.raises(ValueError):
        openmeteo.hourly_series({"timezone": "America/Denver", "hourly": {"time": [], "x": []}}, "x")


def test_situation_fetch_weather_uses_the_shared_client(monkeypatch):
    from orion.situational import context

    calls = {}

    def fake_forecast(lat, lon, **kw):
        calls.update(kw, lat=lat)
        return {"current": {"temperature_2m": 55.0}, "hourly": {"precipitation_probability": [0] * 24,
                                                                "temperature_2m": [55.0] * 24,
                                                                "wind_speed_10m": [3.0] * 24}}
    monkeypatch.setattr(context.openmeteo, "forecast", fake_forecast)
    cfg = context.settings_from_runtime(type("S", (), {"orion_situation_weather_provider": "openmeteo",
                                                       "orion_situation_weather_lat": 0.0,
                                                       "orion_situation_weather_lon": 0.0})())
    env = context._fetch_weather(cfg)
    assert env.available and env.current_weather.temperature_f == 55.0
    assert calls["tz"] == "auto" and calls["lat"] == 0.0 and "temperature_2m" in calls["hourly"]
