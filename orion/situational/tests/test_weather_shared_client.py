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


def _fetch_with(monkeypatch, payload):
    from orion.situational import context

    monkeypatch.setattr(context.openmeteo, "forecast", lambda lat, lon, **kw: payload)
    cfg = context.settings_from_runtime(type("S", (), {"orion_situation_weather_provider": "openmeteo",
                                                       "orion_situation_weather_lat": 0.0,
                                                       "orion_situation_weather_lon": 0.0})())
    return context._fetch_weather(cfg)


def test_forecast_windows_start_at_the_current_hour_not_midnight(monkeypatch):
    # Open-Meteo's hourly arrays begin at local midnight. At 21:15 the next-6h
    # window must cover 21:00-02:00 (cold, rainy here), not 00:00-05:00.
    times = [f"2026-10-26T{h:02d}:00" for h in range(24)] + [f"2026-10-27T{h:02d}:00" for h in range(24)]
    temps = [70.0] * 21 + [30.0] * 27
    probs = [0] * 21 + [90] * 27
    env = _fetch_with(monkeypatch, {
        "current": {"time": "2026-10-26T21:15", "temperature_2m": 33.0},
        "hourly": {"time": times, "temperature_2m": temps, "precipitation_probability": probs,
                   "wind_speed_10m": [3.0] * 48},
    })
    assert env.forecast_next_6h.temperature_high_f == 30.0
    assert env.forecast_next_6h.precipitation_probability_pct == 90
    assert env.practical_flags.icy_roads is True


def test_forecast_window_falls_back_to_start_without_times(monkeypatch):
    env = _fetch_with(monkeypatch, {"current": {"temperature_2m": 55.0},
                                    "hourly": {"temperature_2m": [55.0, 60.0], "precipitation_probability": [0, 0],
                                               "wind_speed_10m": [1.0, 1.0]}})
    assert env.forecast_next_2h.temperature_low_f == 55.0


def test_situation_block_shows_outside_temperature_and_no_lab_line():
    import asyncio
    from datetime import datetime
    from types import SimpleNamespace

    from orion.schemas.situation import EnvironmentContextV1, SituationBriefV1, WeatherCurrentV1
    from orion.situational import context

    now = datetime(2026, 10, 27, 3, 0, tzinfo=timezone.utc)
    cfg = context.settings_from_runtime(SimpleNamespace())
    time_ctx = context._build_time_context(cfg, SimpleNamespace(provider_status={}))
    brief = SituationBriefV1(
        generated_at=now,
        time=time_ctx,
        conversation_phase=asyncio.run(context._build_conversation_phase({}, time_ctx, now)),
        place=context._build_place_context(cfg),
        environment=EnvironmentContextV1(available=True, current_weather=WeatherCurrentV1(
            temperature_f=54.1, feels_like_f=47.7)),
    )
    text = "\n".join(context._build_prompt_fragment(brief, context._DEFAULT_PROMPT_MAX_CHARS).summary_lines)
    assert "Outside temperature now: 54F (feels like 48F)." in text
    assert "Lab" not in text
    assert "lab" not in brief.model_dump()
