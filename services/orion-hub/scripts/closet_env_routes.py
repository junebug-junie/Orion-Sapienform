"""Hub read APIs for the server-closet environment (pan-1 closet Nano + SCD40).

Reads Postgres ``closet_env_sample`` (written by orion-sql-writer from
``orion:closet:env:sample``). Same shapes, staleness rules and downsampling as the
office strip; the pure helpers come from ``office_env_routes``.

Deploy-order safety: Hub can deploy before the sql-writer has created
``closet_env_sample``. A missing table means "no samples yet", so the queries return
no rows (latest ``sample: null``, history ``points: []``) instead of an error.

The closet Nano is the same pan-1 host and the same publisher cadence as the office
Nano, so the office ``OFFICE_ENV_*`` node/stale/max-points settings apply here too.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any, Awaitable, Callable, Mapping, Optional, Sequence

from fastapi import APIRouter, HTTPException, Query

from . import office_env_routes as env
from .cabinet_ambient_routes import parse_window
from .settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/closet/env", tags=["closet-env"])

_FIELDS = ("closet_co2_ppm", "closet_temp_c", "closet_humidity_pct")
_STAT_KEYS = (
    ("closet_co2_ppm", "co2"),
    ("closet_temp_c", "temp"),
    ("closet_humidity_pct", "humidity"),
)

LATEST_SQL = """
    SELECT ts, closet_co2_ppm, closet_temp_c, closet_humidity_pct,
           sensor_online, stale, sample_age_sec
    FROM closet_env_sample
    WHERE node = $1
    ORDER BY ts DESC
    LIMIT 1
"""

HISTORY_SQL = """
    SELECT ts AS t, closet_co2_ppm, closet_temp_c, closet_humidity_pct
    FROM closet_env_sample
    WHERE node = $1
      AND ts >= $2
      AND (closet_co2_ppm IS NOT NULL OR closet_temp_c IS NOT NULL
           OR closet_humidity_pct IS NOT NULL)
    ORDER BY ts ASC
"""


def _is_undefined_table(exc: BaseException) -> bool:
    try:
        import asyncpg
    except ImportError:  # pragma: no cover - asyncpg is a Hub dependency
        return False
    return isinstance(exc, asyncpg.exceptions.UndefinedTableError)


async def query_latest_row(*, node: str) -> Optional[Mapping[str, Any]]:
    try:
        return await env.run_query(LATEST_SQL, node, one=True)
    except Exception as exc:
        if _is_undefined_table(exc):
            logger.info("closet_env_sample does not exist yet; closet strip has no data")
            return None
        raise


async def query_history_rows(*, node: str, hours: int) -> Sequence[Mapping[str, Any]]:
    cutoff = env._now_utc() - timedelta(hours=hours)
    try:
        return await env.run_query(HISTORY_SQL, node, cutoff, one=False)
    except Exception as exc:
        if _is_undefined_table(exc):
            logger.info("closet_env_sample does not exist yet; closet history is empty")
            return []
        raise


_latest_query: Callable[..., Awaitable[Optional[Mapping[str, Any]]]] = query_latest_row
_history_query: Callable[..., Awaitable[Sequence[Mapping[str, Any]]]] = query_history_rows


def row_to_sample(row: Mapping[str, Any]) -> dict[str, Any]:
    return env.row_to_sample(row, fields=_FIELDS, flags=(), payload_fields=(), payload_flags=())


def _stats(raw_points: Sequence[Mapping[str, Any]], n: int) -> dict[str, Any]:
    return env._stats(raw_points, n, stat_keys=_STAT_KEYS)


@router.get("/latest")
async def api_closet_env_latest() -> dict[str, Any]:
    node = str(settings.OFFICE_ENV_NODE)
    try:
        row = await _latest_query(node=node)
    except Exception as exc:
        logger.warning("Closet env latest unavailable: %s", exc)
        return {"ok": False, "age_sec": None, "sample": None, "error": "closet_env_latest_unavailable"}
    return env._load_latest(
        row,
        stale_after_sec=float(settings.OFFICE_ENV_STALE_AFTER_SEC),
        now=env._now_utc(),
        to_sample=row_to_sample,
        fields=_FIELDS,
        flags=(),
    )


@router.get("/history")
async def api_closet_env_history(window: str = Query("24h")) -> dict[str, Any]:
    try:
        hours = parse_window(window)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    node = str(settings.OFFICE_ENV_NODE)
    max_points = int(settings.OFFICE_ENV_HISTORY_MAX_POINTS)
    base = {"node": node, "window": window}
    try:
        rows = await _history_query(node=node, hours=hours)
        raw_points = env.rows_to_points(rows, fields=_FIELDS)
        points = env.downsample_points(raw_points, max_points, fields=_FIELDS)
    except Exception as exc:
        logger.warning("Closet env history unavailable: %s", exc)
        return {
            "ok": False,
            **base,
            "points": [],
            "stats": _stats([], 0),
            "error": "closet_env_history_unavailable",
        }
    return {"ok": True, **base, "points": points, "stats": _stats(raw_points, len(points))}
