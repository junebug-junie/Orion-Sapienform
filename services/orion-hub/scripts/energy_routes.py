"""Hub read APIs for house electricity (orion-energy tables written by sql-writer)
and Orion's own share of it (orion_energy_hourly, also written by sql-writer).

NULL stays null in JSON: an unknown dollar amount is never rendered as $0.
A stakes snapshot older than ORION_ENERGY_STAKES_MAX_AGE_SEC is flagged `stale`
so the strip never shows old numbers as current.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Literal, Mapping, Optional
from zoneinfo import ZoneInfo

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


# --- Orion's own energy against the house -------------------------------------------------
#
# Orion's kWh comes from orion_energy_hourly (per-node wall watts + the cabinet AC plug,
# integrated by sql-writer). The house is RMP's metered usage, about a day late. Orion's
# share is computed only over hours where BOTH exist, so a late or missing RMP hour never
# reads as Orion using 100% (or 0%) of the house.

Grain = Literal["hour", "day", "week", "month", "quarter", "year"]
_DEFAULT_PERIODS = {"hour": 48, "day": 31, "week": 13, "month": 12, "quarter": 8, "year": 3}
_MAX_PERIODS = {"hour": 744, "day": 366, "week": 104, "month": 60, "quarter": 20, "year": 10}
# Fixed display order; any other node that reports watts sorts after these, cooling last.
_SOURCE_ORDER = ("athena", "circe", "hecate")
_COOLING = "cooling"
# An hour joins the share only when every source reporting in it was measured for at least
# this much of it; a feed dropping mid-hour would otherwise pull the share down silently.
_SHARE_MIN_COVERED_SEC = 0.9 * 3600

_ORION_HOURLY_SQL = """
SELECT hour_start, source, kwh, covered_sec FROM orion_energy_hourly
WHERE hour_start >= $1 AND hour_start < $2
"""
_HOUSE_HOURLY_SQL = """
SELECT date_trunc('hour', interval_start) AS hour_start, SUM(energy_kwh) AS kwh,
       SUM(EXTRACT(EPOCH FROM (interval_end - interval_start))) AS covered_sec
FROM energy_usage_interval
WHERE interval_start >= $1 AND interval_start < $2
GROUP BY 1
"""
# One row per interval: the newest pricing wins when a tariff revision re-accrued it.
_PRICE_HOURLY_SQL = """
SELECT date_trunc('hour', interval_start) AS hour_start, SUM(interval_cost_usd) AS usd,
       SUM(marginal_usd_per_kwh * energy_kwh) / NULLIF(SUM(energy_kwh), 0) AS marginal_usd_per_kwh,
       AVG(marginal_usd_per_kwh) AS marginal_unweighted
FROM (
  SELECT DISTINCT ON (usage_point_id, interval_start) interval_start, energy_kwh, interval_cost_usd,
         marginal_usd_per_kwh
  FROM energy_cost_accrued
  WHERE interval_start >= $1 AND interval_start < $2
  ORDER BY usage_point_id, interval_start, computed_at DESC
) latest
GROUP BY 1
"""


def _bucket_floor(local: datetime, grain: str) -> datetime:
    """Start of the bucket holding `local` (an aware local time), as an aware local time."""
    tz = local.tzinfo
    if grain == "hour":
        return local.replace(minute=0, second=0, microsecond=0)
    d = local.date()
    if grain == "week":
        d = d - timedelta(days=d.weekday())
    elif grain == "month":
        d = d.replace(day=1)
    elif grain == "quarter":
        d = d.replace(month=(d.month - 1) // 3 * 3 + 1, day=1)
    elif grain == "year":
        d = d.replace(month=1, day=1)
    return datetime(d.year, d.month, d.day, tzinfo=tz)


def _bucket_step(start: datetime, grain: str, n: int) -> datetime:
    """Move a bucket start by n buckets, on the local calendar (DST-safe for day and up)."""
    if grain == "hour":
        return (start.astimezone(timezone.utc) + timedelta(hours=n)).astimezone(start.tzinfo)
    if grain in ("day", "week"):
        d = start.date() + timedelta(days=n * (7 if grain == "week" else 1))
        return datetime(d.year, d.month, d.day, tzinfo=start.tzinfo)
    months = {"month": 1, "quarter": 3, "year": 12}[grain] * n
    idx = start.year * 12 + (start.month - 1) + months
    return datetime(idx // 12, idx % 12 + 1, 1, tzinfo=start.tzinfo)


def _bucket_label(start: datetime, grain: str) -> str:
    if grain == "hour":
        return start.strftime("%b %-d %-I%p").replace("AM", "am").replace("PM", "pm")
    if grain == "day":
        return start.strftime("%a %b %-d")
    if grain == "week":
        return "Week of " + start.strftime("%b %-d")
    if grain == "month":
        return start.strftime("%b %Y")
    if grain == "quarter":
        return f"Q{(start.month - 1) // 3 + 1} {start.year}"
    return str(start.year)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _source_sort_key(source: str) -> tuple[int, str]:
    if source == _COOLING:
        return (2, source)
    if source in _SOURCE_ORDER:
        return (0, f"{_SOURCE_ORDER.index(source):02d}")
    return (1, source)


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _round(value: Optional[float], digits: int = 3) -> Optional[float]:
    return None if value is None else round(value, digits)


def aggregate_orion_energy(
    *,
    orion_rows: Iterable[Mapping[str, Any]],
    house_rows: Iterable[Mapping[str, Any]],
    price_rows: Iterable[Mapping[str, Any]],
    grain: str,
    periods: int,
    tz: ZoneInfo,
    now: datetime,
) -> dict[str, Any]:
    """Bucket hourly Orion and house rows. Pure: every input is passed in.

    Unknown stays null: a bucket with no Orion rows has orion_kwh null (not 0), a bucket with
    no RMP hours has house_kwh null, and orion_share needs at least one hour with both where
    every reporting source was measured for 90% of the hour (partial_hours counts the rest).
    orion_usd prices each Orion hour at that hour's next-kWh rate (what Orion adds to the
    bill under block pricing); orion_priced_kwh says how much of orion_kwh had a price.
    """
    current = _bucket_floor(now.astimezone(tz), grain)
    starts = [_bucket_step(current, grain, -i) for i in range(periods - 1, -1, -1)]
    ends = [_bucket_step(s, grain, 1) for s in starts]
    window_start, window_end = _utc(starts[0]), _utc(ends[-1])
    position = {_utc(s): i for i, s in enumerate(starts)}

    def index_of(hour: datetime) -> Optional[int]:
        return position.get(_utc(_bucket_floor(_utc(hour).astimezone(tz), grain)))

    orion_by_hour: dict[datetime, dict[str, tuple[float, float]]] = {}
    for row in orion_rows:
        orion_by_hour.setdefault(_utc(row["hour_start"]), {})[str(row["source"])] = (
            float(row["kwh"]), float(row["covered_sec"]))
    house_by_hour = {_utc(r["hour_start"]): float(r["kwh"]) for r in house_rows}
    price_by_hour: dict[datetime, tuple[Optional[float], Optional[float]]] = {}
    for r in price_rows:
        marginal = r.get("marginal_usd_per_kwh")
        if marginal is None:
            marginal = r.get("marginal_unweighted")
        price_by_hour[_utc(r["hour_start"])] = (
            None if r.get("usd") is None else float(r["usd"]),
            None if marginal is None else float(marginal),
        )

    sources = sorted({s for per in orion_by_hour.values() for s in per}, key=_source_sort_key)
    buckets = [
        {"kwh": {}, "covered": {}, "house_kwh": 0.0, "house_hours": 0, "overlap_hours": 0,
         "orion_overlap": 0.0, "house_overlap": 0.0, "partial_hours": 0, "orion_usd": 0.0,
         "orion_priced_kwh": 0.0, "orion_priced_hours": 0, "house_usd": 0.0, "house_priced_hours": 0}
        for _ in starts
    ]

    for hour, per in orion_by_hour.items():
        i = index_of(hour)
        if i is None:
            continue
        b = buckets[i]
        orion_kwh = 0.0
        for source, (kwh, covered) in per.items():
            b["kwh"][source] = b["kwh"].get(source, 0.0) + kwh
            b["covered"][source] = b["covered"].get(source, 0.0) + covered
            orion_kwh += kwh
        if min(covered for _, covered in per.values()) < _SHARE_MIN_COVERED_SEC:
            b["partial_hours"] += 1
        elif hour in house_by_hour:
            b["overlap_hours"] += 1
            b["orion_overlap"] += orion_kwh
            b["house_overlap"] += house_by_hour[hour]
        marginal = price_by_hour.get(hour, (None, None))[1]
        if marginal is not None:
            b["orion_usd"] += orion_kwh * marginal
            b["orion_priced_kwh"] += orion_kwh
            b["orion_priced_hours"] += 1

    for hour, kwh in house_by_hour.items():
        i = index_of(hour)
        if i is None:
            continue
        buckets[i]["house_kwh"] += kwh
        buckets[i]["house_hours"] += 1
        usd = price_by_hour.get(hour, (None, None))[0]
        if usd is not None:
            buckets[i]["house_usd"] += usd
            buckets[i]["house_priced_hours"] += 1

    now_utc = _utc(now)
    out_buckets = []
    for start, end, b in zip(starts, ends, buckets):
        elapsed = (min(_utc(end), now_utc) - _utc(start)).total_seconds() / 3600.0
        orion_kwh = sum(b["kwh"].values()) if b["kwh"] else None
        out_buckets.append({
            "start": _iso(start),
            "end": _iso(end),
            "label": _bucket_label(start, grain),
            "hours_elapsed": _round(max(elapsed, 0.0), 2),
            "complete": _utc(end) <= now_utc,
            "kwh_by_source": {s: _round(v) for s, v in b["kwh"].items()},
            "covered_hours_by_source": {s: _round(v / 3600.0, 2) for s, v in b["covered"].items()},
            "orion_kwh": _round(orion_kwh),
            "house_kwh": _round(b["house_kwh"]) if b["house_hours"] else None,
            "house_hours": b["house_hours"],
            "overlap_hours": b["overlap_hours"],
            "partial_hours": b["partial_hours"],
            "orion_kwh_overlap": _round(b["orion_overlap"]) if b["overlap_hours"] else None,
            "house_kwh_overlap": _round(b["house_overlap"]) if b["overlap_hours"] else None,
            "orion_share": (_round(b["orion_overlap"] / b["house_overlap"], 4)
                            if b["overlap_hours"] and b["house_overlap"] > 0 else None),
            "orion_usd": _round(b["orion_usd"], 2) if b["orion_priced_hours"] else None,
            "orion_priced_kwh": _round(b["orion_priced_kwh"]),
            "house_usd": _round(b["house_usd"], 2) if b["house_priced_hours"] else None,
            "house_priced_hours": b["house_priced_hours"],
        })
    return {
        "ok": True,
        "grain": grain,
        "periods": periods,
        "timezone": str(tz),
        "sources": sources,
        "window_start": _iso(window_start),
        "window_end": _iso(window_end),
        "buckets": out_buckets,
    }


async def _orion_query(start: datetime, end: datetime) -> tuple[list, list, list]:
    conn = await _connect()
    try:
        return (
            list(await conn.fetch(_ORION_HOURLY_SQL, start, end)),
            list(await conn.fetch(_HOUSE_HOURLY_SQL, start, end)),
            list(await conn.fetch(_PRICE_HOURLY_SQL, start, end)),
        )
    finally:
        await conn.close()


@router.get("/orion")
async def api_energy_orion(
    grain: Grain = Query("day"),
    periods: Optional[int] = Query(None, ge=1),
) -> dict[str, Any]:
    periods = min(periods or _DEFAULT_PERIODS[grain], _MAX_PERIODS[grain])
    tz = ZoneInfo(str(settings.HUB_ENERGY_TIMEZONE))
    now = _now()
    current = _bucket_floor(now.astimezone(tz), grain)
    start = _utc(_bucket_step(current, grain, -(periods - 1)))
    end = _utc(_bucket_step(current, grain, 1))
    try:
        orion_rows, house_rows, price_rows = await _orion_query(start, end)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Orion energy unavailable: %s", exc)
        return {"ok": False, "error": "energy_unavailable", "grain": grain, "periods": periods,
                "sources": [], "buckets": []}
    return aggregate_orion_energy(
        orion_rows=orion_rows, house_rows=house_rows, price_rows=price_rows,
        grain=grain, periods=periods, tz=tz, now=now,
    )
