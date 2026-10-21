"""Open-Meteo client: the one place Orion talks to the weather API.

Shared seam for the situation layer (live conditions for chat turns, ``orion/situational/context.py``)
and orion-prediction (hourly history + the next 48 h as forecast covariates). No API key; location is
the caller's (``ORION_SITUATION_WEATHER_LAT/LON``).

- ``forecast``: current conditions plus hourly values, optionally including the last ``past_days``.
- ``archive``: observed hourly history for a date range (the reanalysis archive; ~2 days behind).
- ``hourly_series``: one hourly field as ``[(utc datetime, value | None)]`` -- requires the request to
  have been made with ``timezone="UTC"`` (the default here).
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from typing import Any, Iterable
from urllib.parse import urlencode
from urllib.request import urlopen

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def _get(url: str, params: dict[str, Any], timeout: float) -> dict:
    with urlopen(f"{url}?{urlencode(params)}", timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def forecast(lat: float, lon: float, *, hourly: Iterable[str] = (), current: Iterable[str] = (),
             forecast_days: int = 2, past_days: int = 0, tz: str = "UTC", timeout: float = 4.0,
             temperature_unit: str = "fahrenheit", wind_speed_unit: str = "mph") -> dict:
    params: dict[str, Any] = {"latitude": lat, "longitude": lon, "forecast_days": forecast_days,
                              "temperature_unit": temperature_unit, "wind_speed_unit": wind_speed_unit,
                              "timezone": tz}
    if hourly:
        params["hourly"] = ",".join(hourly)
    if current:
        params["current"] = ",".join(current)
    if past_days:
        params["past_days"] = past_days
    return _get(FORECAST_URL, params, timeout)


def archive(lat: float, lon: float, start: date, end: date, *, hourly: Iterable[str],
            timeout: float = 15.0, temperature_unit: str = "fahrenheit") -> dict:
    return _get(ARCHIVE_URL, {"latitude": lat, "longitude": lon, "start_date": start.isoformat(),
                              "end_date": end.isoformat(), "hourly": ",".join(hourly),
                              "temperature_unit": temperature_unit, "timezone": "UTC"}, timeout)


def hourly_series(payload: dict, field: str) -> list[tuple[datetime, float | None]]:
    if (payload.get("timezone") or "UTC").upper() not in ("UTC", "GMT"):
        raise ValueError(f"hourly_series needs a UTC response, got timezone={payload.get('timezone')!r}")
    h = payload.get("hourly") or {}
    out = []
    for t, v in zip(h.get("time") or (), h.get(field) or ()):
        ts = datetime.fromisoformat(t).replace(tzinfo=timezone.utc)
        out.append((ts, None if v is None else float(v)))
    return out
