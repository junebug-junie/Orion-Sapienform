"""Hub read APIs for the office environment (pan-1 SHT40): latest state and history.

Reads Postgres ``office_env_sample`` (written by orion-sql-writer from
``orion:office:env:sample``). Absent is not zero: stale samples carry NULL readings,
so they appear as gaps in history, never as a flat line at 0.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Iterable, Mapping, Optional, Sequence

from fastapi import APIRouter, HTTPException, Query

from .cabinet_ambient_routes import _iso_utc, _parse_db_timestamp, parse_window
from .settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/office/env", tags=["office-env"])

_FIELDS = ("office_temp_c", "office_humidity_pct")


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _payload_dict(row: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    """``payload_json`` is a Postgres ``json`` column: asyncpg hands it back as text."""
    payload_json = row.get("payload_json")
    if isinstance(payload_json, str):
        try:
            payload_json = json.loads(payload_json)
        except ValueError:
            return None
    return payload_json if isinstance(payload_json, dict) else None


def row_to_sample(row: Mapping[str, Any]) -> dict[str, Any]:
    """DB row -> absent-safe API sample (missing readings are omitted, not zeroed)."""
    sample: dict[str, Any] = {
        "ts": _iso_utc(row["ts"]),
        "sensor_online": bool(row["sensor_online"]),
    }
    for field in _FIELDS:
        if row.get(field) is not None:
            sample[field] = float(row[field])
    return sample


def rows_to_points(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for row in rows:
        timestamp = row.get("t")
        if timestamp is None:
            continue
        point: dict[str, Any] = {"t": _iso_utc(timestamp)}
        for field in _FIELDS:
            if row.get(field) is not None:
                point[field] = float(row[field])
        if len(point) > 1:
            points.append(point)
    return points


def downsample_points(points: Sequence[Mapping[str, Any]], max_points: int) -> list[dict[str, Any]]:
    """Bucket-average ordered points to at most ``max_points``."""
    if max_points < 1:
        raise ValueError("max_points must be positive")
    if len(points) <= max_points:
        return [dict(p) for p in points]
    sampled: list[dict[str, Any]] = []
    for i in range(max_points):
        bucket = points[i * len(points) // max_points : (i + 1) * len(points) // max_points]
        if not bucket:
            continue
        out: dict[str, Any] = {"t": bucket[0]["t"]}
        for field in _FIELDS:
            values = [float(p[field]) for p in bucket if p.get(field) is not None]
            if values:
                out[field] = sum(values) / len(values)
        sampled.append(out)
    return sampled


async def query_latest_row(*, node: str) -> Optional[Mapping[str, Any]]:
    """Newest sample row, stale or not (the stale flag is itself the signal)."""
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured")

    import asyncpg

    connection = await asyncpg.connect(dsn=database_url)
    try:
        return await connection.fetchrow(
            """
            SELECT ts, office_temp_c, office_humidity_pct, sensor_online,
                   stale, sample_age_sec, payload_json
            FROM office_env_sample
            WHERE node = $1
            ORDER BY ts DESC
            LIMIT 1
            """,
            node,
        )
    finally:
        await connection.close()


async def query_history_rows(*, node: str, hours: int) -> Sequence[Mapping[str, Any]]:
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured")

    import asyncpg

    cutoff = _now_utc() - timedelta(hours=hours)
    connection = await asyncpg.connect(dsn=database_url)
    try:
        return await connection.fetch(
            """
            SELECT ts AS t, office_temp_c, office_humidity_pct
            FROM office_env_sample
            WHERE node = $1
              AND ts >= $2
              AND (office_temp_c IS NOT NULL OR office_humidity_pct IS NOT NULL)
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
    out: dict[str, Any] = {"n_raw": len(raw_points), "n": n}
    for field, key in (("office_temp_c", "temp"), ("office_humidity_pct", "humidity")):
        values = [float(p[field]) for p in raw_points if p.get(field) is not None]
        out[f"{key}_min"] = min(values) if values else None
        out[f"{key}_max"] = max(values) if values else None
    return out


def _load_latest(
    row: Optional[Mapping[str, Any]], *, stale_after_sec: float, now: datetime
) -> dict[str, Any]:
    """``ok`` needs BOTH a recent row (the publisher is alive) and a non-stale
    sample (the sensor itself is alive)."""
    if row is None:
        return {"ok": False, "age_sec": None, "sample": None}
    received_at = _parse_db_timestamp(row["ts"])
    age_sec = (now.astimezone(timezone.utc) - received_at).total_seconds()
    sensor_stale = bool(row.get("stale"))
    out: dict[str, Any] = {
        "ok": age_sec <= stale_after_sec and not sensor_stale,
        "age_sec": age_sec,
        "sample": row_to_sample(row),
        "sensor_stale": sensor_stale,
    }
    sample_age = row.get("sample_age_sec")
    if isinstance(sample_age, (int, float)) and not isinstance(sample_age, bool):
        out["sample_age_sec"] = float(sample_age)
    return out


@router.get("/latest")
async def api_office_env_latest() -> dict[str, Any]:
    node = str(settings.OFFICE_ENV_NODE)
    try:
        row = await _latest_query(node=node)
    except Exception as exc:
        logger.warning("Office env latest unavailable: %s", exc)
        return {"ok": False, "age_sec": None, "sample": None}
    return _load_latest(
        row, stale_after_sec=float(settings.OFFICE_ENV_STALE_AFTER_SEC), now=_now_utc()
    )


@router.get("/history")
async def api_office_env_history(window: str = Query("24h")) -> dict[str, Any]:
    try:
        hours = parse_window(window)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    node = str(settings.OFFICE_ENV_NODE)
    max_points = int(settings.CABINET_AMBIENT_HISTORY_MAX_POINTS)
    base = {"node": node, "window": window}
    try:
        rows = await _history_query(node=node, hours=hours)
        raw_points = rows_to_points(rows)
        points = downsample_points(raw_points, max_points)
    except Exception as exc:
        logger.warning("Office env history unavailable: %s", exc)
        return {
            "ok": False,
            **base,
            "points": [],
            "stats": _stats([], 0),
            "error": "office_env_history_unavailable",
        }
    return {"ok": True, **base, "points": points, "stats": _stats(raw_points, len(points))}
