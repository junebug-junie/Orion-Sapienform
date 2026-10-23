"""Keep prediction_weather_hourly current from the shared Open-Meteo client.

Every refresh, first: the last 7 days and the next 48 h from the forecast API (stored even if the
backfill below fails). Then, when observed history before the 7-day window has holes or does not reach
back to ``backfill_from``, the missing span is filled from the historical-forecast API -- the same models
as the live feed, so history has no level jump. Observed values replace forecasts; forecasts never
replace observations. Only observed rows are ever read as covariate values (store.points).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from orion.weather import openmeteo

logger = logging.getLogger("orion-prediction.weather")
RECENT_DAYS = 7


def refresh(store, lat: float, lon: float, now: datetime, backfill_from: datetime | None) -> dict:
    hour_now = now.replace(minute=0, second=0, microsecond=0)
    fc = openmeteo.forecast(lat, lon, hourly=("temperature_2m",), past_days=RECENT_DAYS, forecast_days=2, timeout=10)
    rows = [(t, v, t > hour_now, "forecast" if t > hour_now else "recent")
            for t, v in openmeteo.hourly_series(fc, "temperature_2m") if v is not None]
    n = store.upsert_weather(rows, now)
    out = {"ok": True, "rows": n, "refreshed_at": now.isoformat(), "backfilled": None, "backfill_error": None}
    if backfill_from is not None:
        end = hour_now - timedelta(days=RECENT_DAYS)
        missing = store.missing_weather_hours(backfill_from, end)
        if missing:
            try:
                hist = openmeteo.historical_forecast(lat, lon, missing[0].date(), missing[-1].date(),
                                                     hourly=("temperature_2m",))
                want = set(missing)
                back = [(t, v, False, "history") for t, v in openmeteo.hourly_series(hist, "temperature_2m")
                        if v is not None and t in want]
                store.upsert_weather(back, now)
                out["backfilled"] = {"hours": len(back), "from": missing[0].isoformat(), "to": missing[-1].isoformat()}
            except Exception as exc:  # noqa: BLE001 -- the recent/forecast rows above are already stored
                out["backfill_error"] = f"{type(exc).__name__}: {exc}"[:300]
                logger.warning("prediction_weather_backfill_failed err=%s", out["backfill_error"])
    return out
