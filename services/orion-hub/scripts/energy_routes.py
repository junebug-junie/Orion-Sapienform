"""Hub read APIs for house electricity (orion-energy tables written by sql-writer).

NULL stays null in JSON: an unknown dollar amount is never rendered as $0.
A stakes snapshot older than ORION_ENERGY_STAKES_MAX_AGE_SEC is flagged `stale`
so the strip never shows old numbers as current.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timezone
from typing import Any, Mapping, Optional

from fastapi import APIRouter, Query

from .settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/energy", tags=["energy"])

_STAKES_SQL = """
SELECT as_of, cycle_start, cycle_end, covered_through, cycle_accumulated_kwh, cycle_to_date_total_usd,
       marginal_usd_per_kwh, orion_projected_total_usd, forecast_total_usd, forecast_as_of,
       projected_to_forecast_ratio, importer_state, pressure, pressure_reason, tariff_version
FROM energy_stakes_snapshot ORDER BY as_of DESC LIMIT 1
"""
_IMPORTER_SQL = """
SELECT as_of, state, reason, source, last_success_at, last_attempt_at, latest_interval_end, usage_lag_hours
FROM energy_importer_status ORDER BY as_of DESC LIMIT 1
"""
_RECONCILE_SQL = """
SELECT DISTINCT ON (reconcile_kind)
       reconcile_kind, billing_period_start, billing_period_end, utility_as_of, utility_kwh,
       utility_total_usd, utility_basis, orion_kwh, orion_total_usd, orion_method, reconcile_gap,
       delta_kwh, delta_usd, delta_pct, bucket_deltas, tariff_version, computed_at
FROM energy_reconcile
ORDER BY reconcile_kind, billing_period_start DESC, computed_at DESC, utility_as_of DESC
"""
# ESPI intervals may be 15 or 60 minutes, so "hours" is covered time, not a row count.
# The window starts at local midnight (days - 1) days ago so the oldest bar is a whole day.
_DAILY_SQL = """
SELECT (interval_start AT TIME ZONE $2)::date AS day,
       SUM(energy_kwh) AS kwh,
       SUM(EXTRACT(EPOCH FROM (interval_end - interval_start))) / 3600.0 AS hours
FROM energy_usage_interval
WHERE interval_start >= ((date_trunc('day', now() AT TIME ZONE $2) - ($1::int - 1) * interval '1 day') AT TIME ZONE $2)
GROUP BY 1 ORDER BY 1
"""


async def _connect():
    import asyncpg

    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured")
    return await asyncpg.connect(dsn=database_url)


async def _latest_query() -> tuple[Optional[Mapping[str, Any]], Optional[Mapping[str, Any]], list[Mapping[str, Any]]]:
    conn = await _connect()
    try:
        return (
            await conn.fetchrow(_STAKES_SQL),
            await conn.fetchrow(_IMPORTER_SQL),
            list(await conn.fetch(_RECONCILE_SQL)),
        )
    finally:
        await conn.close()


async def _daily_query(days: int, tz: str) -> list[Mapping[str, Any]]:
    conn = await _connect()
    try:
        return list(await conn.fetch(_DAILY_SQL, days, tz))
    finally:
        await conn.close()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _stale(stakes: Optional[Mapping[str, Any]]) -> Optional[bool]:
    if stakes is None:
        return None
    as_of = stakes.get("as_of")
    if not isinstance(as_of, datetime):
        return True
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)
    return (_now() - as_of).total_seconds() > float(settings.ORION_ENERGY_STAKES_MAX_AGE_SEC)


def _jsonable(row: Optional[Mapping[str, Any]]) -> Optional[dict[str, Any]]:
    if row is None:
        return None
    out: dict[str, Any] = {}
    for key, value in dict(row).items():
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            value = value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        elif isinstance(value, date):
            value = value.isoformat()
        elif key == "bucket_deltas" and isinstance(value, str):
            value = json.loads(value)
        out[key] = value
    return out


@router.get("/latest")
async def api_energy_latest() -> dict[str, Any]:
    try:
        stakes, importer, reconcile = await _latest_query()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Energy latest unavailable: %s", exc)
        return {
            "ok": False, "error": "energy_unavailable", "stale": None, "as_of": None, "covered_through": None,
            "stakes": None, "importer": None, "reconcile": {},
        }
    stakes_json = _jsonable(stakes) or {}
    return {
        "ok": stakes is not None,
        "stale": _stale(stakes),
        "as_of": stakes_json.get("as_of"),
        "covered_through": stakes_json.get("covered_through"),
        "stakes": _jsonable(stakes),
        "importer": _jsonable(importer),
        "reconcile": {row["reconcile_kind"]: _jsonable(row) for row in reconcile},
    }


@router.get("/usage/daily")
async def api_energy_usage_daily(days: int = Query(14, ge=1, le=90)) -> dict[str, Any]:
    try:
        rows = await _daily_query(days, str(settings.HUB_ENERGY_TIMEZONE))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Energy daily usage unavailable: %s", exc)
        return {"ok": False, "error": "energy_unavailable", "days": days, "points": []}
    points = [
        {
            "day": r["day"].isoformat() if isinstance(r["day"], date) else str(r["day"]),
            "kwh": float(r["kwh"]),
            "hours": int(round(float(r["hours"]))),
        }
        for r in rows
    ]
    return {"ok": True, "days": days, "points": points}
