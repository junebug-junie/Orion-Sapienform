"""Keep prediction_weather_hourly current from the shared Open-Meteo client.

Every refresh: the last 7 days (observed) and the next 48 h (forecast) from the forecast API. When the
stored history does not reach back to ``backfill_from`` (first start), the gap is filled once from the
archive. Observed values replace forecast ones; forecasts never overwrite observations.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone

from orion.weather import openmeteo

logger = logging.getLogger("orion-prediction.weather")


def refresh(store, lat: float, lon: float, now: datetime, backfill_from: datetime | None) -> dict:
    rows = []
    lo, _ = store.weather_span()
    if backfill_from is not None and (lo is None or lo > backfill_from):
        end = min((lo or now) - timedelta(hours=1), now - timedelta(days=2))
        if end > backfill_from:
            arc = openmeteo.archive(lat, lon, backfill_from.date(), end.date(), hourly=("temperature_2m",))
            rows += [(t, v, False, "archive") for t, v in openmeteo.hourly_series(arc, "temperature_2m")
                     if v is not None]
    fc = openmeteo.forecast(lat, lon, hourly=("temperature_2m",), past_days=7, forecast_days=2, timeout=10)
    hour_now = now.replace(minute=0, second=0, microsecond=0)
    rows += [(t, v, t > hour_now, "forecast") for t, v in openmeteo.hourly_series(fc, "temperature_2m") if v is not None]
    n = store.upsert_weather(rows, now)
    return {"ok": True, "rows": n, "refreshed_at": now.isoformat(),
            "backfilled_from": backfill_from.isoformat() if backfill_from and (lo is None or lo > backfill_from) else None}
