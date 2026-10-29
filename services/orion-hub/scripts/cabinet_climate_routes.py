"""Hub read API for the Cabinet / Office climate overlay chart.

One window, four series on a shared time grid, each read from the table its own strip
already uses:

- ``office_temp_c``: ``office_env_sample`` (pan-1 SHT40).
- ``cabinet_temp_c``: ``orion_biometrics_summary.measurements.cabinet_temp_c`` (Nano BME680).
- ``outdoor_temp_c``: ``prediction_weather_hourly`` observed hours (Open-Meteo, stored in °F).
- ``orion_watts``: per-bucket mean wall watts of every node in ``orion_biometrics_cluster``
  plus the cabinet cooling plug (``home_cooling_sample``) -- the same sources the Orion
  energy tab counts. Each point carries ``sources`` so a node that went quiet reads as a
  smaller total with fewer sources, not as a silent drop.

Absent is not zero: an empty bucket is omitted, never filled. Each series is read on its
own; one failing table empties only that series and names it in ``errors``.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Mapping, Sequence

from fastapi import APIRouter, HTTPException, Query

from .cabinet_ambient_routes import _iso_utc, biometrics_summary_cutoff, parse_window
from .settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cabinet/climate", tags=["cabinet-climate"])

# Bucket width per window: ~288-336 points across the chart.
_GRAIN_SEC = {"24h": 300, "3d": 900, "7d": 1800}
WEATHER_GRAIN_SEC = 3600

SERIES = ("office_temp_c", "cabinet_temp_c", "outdoor_temp_c", "orion_watts")

_OFFICE_SQL = """
SELECT date_bin(make_interval(secs => $3), ts, 'epoch') AS t, avg(office_temp_c) AS v
FROM office_env_sample
WHERE node = $1 AND ts >= $2 AND office_temp_c IS NOT NULL
GROUP BY 1 ORDER BY 1
"""
# ``timestamp`` is varchar here; $2 is a biometrics_summary_cutoff() string.
_CABINET_SQL = """
SELECT date_bin(make_interval(secs => $3), "timestamp"::timestamptz, 'epoch') AS t,
       avg((measurements->>'cabinet_temp_c')::double precision) AS v
FROM orion_biometrics_summary
WHERE node = $1 AND "timestamp" >= $2 AND measurements ? 'cabinet_temp_c'
GROUP BY 1 ORDER BY 1
"""
_OUTDOOR_SQL = """
SELECT hour_ts AS t, (temp_f - 32) * 5.0 / 9.0 AS v
FROM prediction_weather_hourly
WHERE hour_ts >= $1 AND NOT is_forecast AND temp_f IS NOT NULL
ORDER BY 1
"""
# Mean per source per bucket, then summed: a source sampled more often must not weigh more.
_WATTS_SQL = """
SELECT t, sum(w) AS v, count(*) AS sources FROM (
  SELECT date_bin(make_interval(secs => $2), c.observed_at, 'epoch') AS t, n.key AS source,
         avg((n.value->>'chassis_watts')::double precision) AS w
  FROM orion_biometrics_cluster c, jsonb_each(c.measurements_by_node) n
  WHERE c.observed_at >= $1 AND jsonb_typeof(n.value->'chassis_watts') = 'number'
  GROUP BY 1, 2
  UNION ALL
  SELECT date_bin(make_interval(secs => $2), ts, 'epoch'), 'cooling', avg(cooling_watts)
  FROM home_cooling_sample
  WHERE ts >= $1 AND cooling_watts IS NOT NULL AND NOT COALESCE(stale, false) AND device_online
  GROUP BY 1
) per_source
GROUP BY 1 ORDER BY 1
"""


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def rows_to_points(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for row in rows:
        if row.get("t") is None or row.get("v") is None:
            continue
        point: dict[str, Any] = {"t": _iso_utc(row["t"]), "v": float(row["v"])}
        if row.get("sources") is not None:
            point["sources"] = int(row["sources"])
        points.append(point)
    return points


async def query_series_rows(*, hours: int, grain_sec: int) -> dict[str, Any]:
    """Return {series: rows | Exception}; one failed query does not sink the others."""
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured")

    import asyncpg

    since = _now_utc() - timedelta(hours=hours)
    queries: dict[str, tuple[str, tuple[Any, ...]]] = {
        "office_temp_c": (_OFFICE_SQL, (str(settings.OFFICE_ENV_NODE), since, float(grain_sec))),
        "cabinet_temp_c": (
            _CABINET_SQL,
            (str(settings.CABINET_AMBIENT_HISTORY_NODE), biometrics_summary_cutoff(since), float(grain_sec)),
        ),
        "outdoor_temp_c": (_OUTDOOR_SQL, (since,)),
        "orion_watts": (_WATTS_SQL, (since, float(grain_sec))),
    }
    out: dict[str, Any] = {}
    connection = await asyncpg.connect(dsn=database_url)
    try:
        for key, (sql, args) in queries.items():
            try:
                out[key] = list(await connection.fetch(sql, *args))
            except Exception as exc:  # noqa: BLE001
                out[key] = exc
    finally:
        await connection.close()
    return out


SeriesQuery = Callable[..., Awaitable[dict[str, Any]]]
_series_query: SeriesQuery = query_series_rows


@router.get("/overlay")
async def api_cabinet_climate_overlay(window: str = Query("24h")) -> dict[str, Any]:
    try:
        hours = parse_window(window)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    grain_sec = _GRAIN_SEC[window]
    base = {
        "window": window,
        "grain_sec": grain_sec,
        "series_grain_sec": {key: WEATHER_GRAIN_SEC if key == "outdoor_temp_c" else grain_sec for key in SERIES},
    }
    try:
        results = await _series_query(hours=hours, grain_sec=grain_sec)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cabinet climate overlay unavailable: %s", exc)
        return {"ok": False, **base, "series": {key: [] for key in SERIES},
                "errors": {key: "unavailable" for key in SERIES}}

    series: dict[str, list[dict[str, Any]]] = {}
    errors: dict[str, str] = {}
    for key in SERIES:
        rows = results.get(key)
        if isinstance(rows, Exception) or rows is None:
            if rows is not None:
                logger.warning("Cabinet climate series %s unavailable: %s", key, rows)
            series[key] = []
            errors[key] = "unavailable"
            continue
        series[key] = rows_to_points(rows)
    return {"ok": len(errors) < len(SERIES), **base, "series": series, "errors": errors}
