"""Hub read APIs for cabinet cooling (Shelly Wave) latest state and history."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Iterable, Mapping, Optional, Sequence

from fastapi import APIRouter, HTTPException, Query

from .cabinet_ambient_routes import _iso_utc, _parse_db_timestamp, parse_window
from .settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cabinet/cooling", tags=["cabinet-cooling"])


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _product_from_row(row: Mapping[str, Any]) -> Optional[str]:
    payload_json = row.get("payload_json")
    if isinstance(payload_json, dict):
        device = payload_json.get("device")
        if isinstance(device, dict) and device.get("product"):
            return str(device["product"])
    return None


def row_to_sample(row: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a DB row into an absent-safe API sample."""
    sample: dict[str, Any] = {
        "ts": _iso_utc(row["ts"]),
        "controller_ready": bool(row["controller_ready"]),
        "device_online": bool(row["device_online"]),
    }
    if row.get("cooling_watts") is not None:
        sample["cooling_watts"] = float(row["cooling_watts"])
    if row.get("cooling_volts") is not None:
        sample["cooling_volts"] = float(row["cooling_volts"])
    if row.get("switch_on") is not None:
        sample["switch_on"] = bool(row["switch_on"])
    product = _product_from_row(row)
    if product:
        sample["product"] = product
    return sample


def rows_to_points(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Convert selected DB rows without inventing absent optional values."""
    points: list[dict[str, Any]] = []
    for row in rows:
        timestamp = row.get("t")
        if timestamp is None:
            continue
        point: dict[str, Any] = {"t": _iso_utc(timestamp)}
        if row.get("cooling_watts") is not None:
            point["cooling_watts"] = float(row["cooling_watts"])
        if row.get("cooling_volts") is not None:
            point["cooling_volts"] = float(row["cooling_volts"])
        if len(point) > 1:
            points.append(point)
    return points


def downsample_cooling_points(
    points: Sequence[Mapping[str, Any]], max_points: int
) -> list[dict[str, Any]]:
    """Bucket-average ordered cooling points to at most ``max_points``."""
    if max_points < 1:
        raise ValueError("max_points must be positive")
    if len(points) <= max_points:
        return [dict(point) for point in points]

    sampled: list[dict[str, Any]] = []
    fields = ("cooling_watts", "cooling_volts")
    for bucket_index in range(max_points):
        start = bucket_index * len(points) // max_points
        end = (bucket_index + 1) * len(points) // max_points
        bucket = points[start:end]
        if not bucket:
            continue
        result: dict[str, Any] = {"t": bucket[0]["t"]}
        for field in fields:
            values = [
                float(point[field])
                for point in bucket
                if point.get(field) is not None
            ]
            if values:
                result[field] = sum(values) / len(values)
        sampled.append(result)
    return sampled


async def query_latest_row(*, node: str) -> Optional[Mapping[str, Any]]:
    """Read the newest cooling sample from Hub's configured Postgres."""
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured")

    import asyncpg

    connection = await asyncpg.connect(dsn=database_url)
    try:
        return await connection.fetchrow(
            """
            SELECT
              ts,
              cooling_watts,
              cooling_volts,
              switch_on,
              controller_ready,
              device_online,
              payload_json
            FROM home_cooling_sample
            WHERE node = $1
            ORDER BY ts DESC
            LIMIT 1
            """,
            node,
        )
    finally:
        await connection.close()


async def query_history_rows(*, node: str, hours: int) -> Sequence[Mapping[str, Any]]:
    """Read cooling watts history from Hub's configured Postgres."""
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured")

    import asyncpg

    cutoff = _now_utc() - timedelta(hours=hours)
    connection = await asyncpg.connect(dsn=database_url)
    try:
        return await connection.fetch(
            """
            SELECT
              ts AS t,
              cooling_watts,
              cooling_volts
            FROM home_cooling_sample
            WHERE node = $1
              AND ts >= $2
            ORDER BY ts ASC
            """,
            node,
            cutoff,
        )
    finally:
        await connection.close()


LatestQuery = Callable[..., Awaitable[Optional[Mapping[str, Any]]]]
HistoryQuery = Callable[..., Awaitable[Sequence[Mapping[str, Any]]]]
_latest_query: LatestQuery = query_latest_row
_history_query: HistoryQuery = query_history_rows


def _stats(raw_points: Sequence[Mapping[str, Any]], n: int) -> dict[str, Any]:
    watts_values = [
        float(point["cooling_watts"])
        for point in raw_points
        if point.get("cooling_watts") is not None
    ]
    return {
        "n_raw": len(raw_points),
        "n": n,
        "watts_min": min(watts_values) if watts_values else None,
        "watts_max": max(watts_values) if watts_values else None,
    }


def _load_latest(row: Optional[Mapping[str, Any]], *, stale_after_sec: float, now: datetime) -> dict[str, Any]:
    if row is None:
        return {"ok": False, "age_sec": None, "sample": None}

    received_at = _parse_db_timestamp(row["ts"])
    age_sec = (now.astimezone(timezone.utc) - received_at).total_seconds()
    stale = age_sec > stale_after_sec
    return {
        "ok": not stale,
        "age_sec": age_sec,
        "sample": row_to_sample(row),
    }


@router.get("/latest")
async def api_cabinet_cooling_latest() -> dict[str, Any]:
    node = str(settings.CABINET_AMBIENT_HISTORY_NODE)
    try:
        row = await _latest_query(node=node)
    except Exception as exc:
        logger.warning("Cabinet cooling latest unavailable: %s", exc)
        return {"ok": False, "age_sec": None, "sample": None}

    return _load_latest(
        row,
        stale_after_sec=float(settings.CABINET_SENSORS_STALE_AFTER_SEC),
        now=_now_utc(),
    )


@router.get("/history")
async def api_cabinet_cooling_history(
    window: str = Query("24h"),
) -> dict[str, Any]:
    try:
        hours = parse_window(window)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    node = str(settings.CABINET_AMBIENT_HISTORY_NODE)
    max_points = int(settings.CABINET_AMBIENT_HISTORY_MAX_POINTS)
    base = {
        "node": node,
        "window": window,
    }
    try:
        rows = await _history_query(node=node, hours=hours)
        raw_points = rows_to_points(rows)
        points = downsample_cooling_points(raw_points, max_points)
    except Exception as exc:
        logger.warning("Cabinet cooling history unavailable: %s", exc)
        return {
            "ok": False,
            **base,
            "points": [],
            "stats": _stats([], 0),
            "error": "cooling_history_unavailable",
        }

    return {
        "ok": True,
        **base,
        "points": points,
        "stats": _stats(raw_points, len(points)),
    }
