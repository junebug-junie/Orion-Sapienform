"""Hub read APIs for the office environment (pan-1 SHT40 + CCS811 + PMSA003I + TSL2591):
latest state and history.

Reads Postgres ``office_env_sample`` (written by orion-sql-writer from
``orion:office:env:sample``). Absent is not zero: stale samples carry NULL readings,
so they appear as gaps in history, never as a flat line at 0.

Deploy-order safety: the particulate and light readings (``_PAYLOAD_FIELDS``) and the
``light_saturated`` flag are read from ``payload_json`` (the full validated sample the
sql-writer stores on every row), never from their own columns. Those columns are added
by the sql-writer at its boot, and Hub can deploy first; a query naming them would fail
the whole strip, including temperature. Rows written before the new sensors existed
simply have no such keys, so the readings are absent, not zero.

The pure helpers here take a ``fields`` tuple, so ``closet_env_routes.py`` reuses them
for the server-closet SCD40.
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

# Readings that have their own long-standing columns.
_COLUMN_FIELDS = ("office_temp_c", "office_humidity_pct", "office_eco2_ppm", "office_tvoc_ppb")
# Readings read from payload_json->'measurements' (see the module docstring).
_PAYLOAD_FIELDS = ("office_pm1_ug_m3", "office_pm25_ug_m3", "office_pm10_ug_m3", "office_lux", "office_sound_dba")
_FIELDS = _COLUMN_FIELDS + _PAYLOAD_FIELDS
# Boolean state flags carried into the latest sample. NULL / missing = no claim, omitted.
_COLUMN_FLAGS = ("air_warming",)
_PAYLOAD_FLAGS = ("light_saturated",)
_FLAGS = _COLUMN_FLAGS + _PAYLOAD_FLAGS
_STAT_KEYS = (
    ("office_temp_c", "temp"),
    ("office_humidity_pct", "humidity"),
    ("office_eco2_ppm", "eco2"),
    ("office_tvoc_ppb", "tvoc"),
    ("office_pm1_ug_m3", "pm1"),
    ("office_pm25_ug_m3", "pm25"),
    ("office_pm10_ug_m3", "pm10"),
    ("office_lux", "lux"),
    ("office_sound_dba", "sound"),
)


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


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _payload_section(row: Mapping[str, Any], section: str) -> Mapping[str, Any]:
    payload = _payload_dict(row) or {}
    value = payload.get(section)
    return value if isinstance(value, dict) else {}


def row_to_sample(
    row: Mapping[str, Any],
    *,
    fields: Sequence[str] = _FIELDS,
    flags: Sequence[str] = _FLAGS,
    payload_fields: Sequence[str] = _PAYLOAD_FIELDS,
    payload_flags: Sequence[str] = _PAYLOAD_FLAGS,
) -> dict[str, Any]:
    """DB row -> absent-safe API sample (missing readings are omitted, not zeroed).

    ``payload_fields`` / ``payload_flags`` come from ``payload_json`` when the row has no
    column for them (deploy-order safety, see the module docstring)."""
    measurements = _payload_section(row, "measurements") if payload_fields else {}
    state = _payload_section(row, "state") if payload_flags else {}
    sample: dict[str, Any] = {
        "ts": _iso_utc(row["ts"]),
        "sensor_online": bool(row["sensor_online"]),
    }
    for field in fields:
        value = row.get(field)
        if value is not None:
            sample[field] = float(value)
        elif field in payload_fields and _is_number(measurements.get(field)):
            sample[field] = float(measurements[field])
    # NULL on rows written before a sensor existed: no claim either way, so omitted.
    for flag in flags:
        value = row.get(flag)
        if value is None and flag in payload_flags:
            value = state.get(flag)
        if value is not None:
            sample[flag] = bool(value)
    return sample


def rows_to_points(
    rows: Iterable[Mapping[str, Any]], *, fields: Sequence[str] = _FIELDS
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for row in rows:
        timestamp = row.get("t")
        if timestamp is None:
            continue
        point: dict[str, Any] = {"t": _iso_utc(timestamp)}
        for field in fields:
            if row.get(field) is not None:
                point[field] = float(row[field])
        if len(point) > 1:
            points.append(point)
    return points


GAP_FACTOR = 3.0


def _ts(point: Mapping[str, Any]) -> float:
    return datetime.fromisoformat(str(point["t"]).replace("Z", "+00:00")).timestamp()


def _split_at_gaps(points: Sequence[Mapping[str, Any]]) -> list[Sequence[Mapping[str, Any]]]:
    """Split ordered points into segments wherever the raw spacing exceeds
    ``GAP_FACTOR`` x the median raw spacing (a sensor outage)."""
    if len(points) < 3:
        return [points]
    times = [_ts(p) for p in points]
    gaps = [b - a for a, b in zip(times, times[1:])]
    median = sorted(gaps)[len(gaps) // 2]
    if median <= 0:
        return [points]
    limit = median * GAP_FACTOR
    segments: list[Sequence[Mapping[str, Any]]] = []
    start = 0
    for i, gap in enumerate(gaps, start=1):
        if gap > limit:
            segments.append(points[start:i])
            start = i
    segments.append(points[start:])
    return segments


def _bucket(
    points: Sequence[Mapping[str, Any]], n: int, fields: Sequence[str]
) -> list[dict[str, Any]]:
    if len(points) <= n:
        return [dict(p) for p in points]
    out_points: list[dict[str, Any]] = []
    for i in range(n):
        bucket = points[i * len(points) // n : (i + 1) * len(points) // n]
        if not bucket:
            continue
        out: dict[str, Any] = {"t": bucket[0]["t"]}
        for field in fields:
            values = [float(p[field]) for p in bucket if p.get(field) is not None]
            if values:
                out[field] = sum(values) / len(values)
        out_points.append(out)
    return out_points


def downsample_points(
    points: Sequence[Mapping[str, Any]], max_points: int, *, fields: Sequence[str] = _FIELDS
) -> list[dict[str, Any]]:
    """Bucket-average ordered points to about ``max_points``, never across an outage.

    Averaging is done per contiguous segment, and the first point after an outage carries
    ``gap: true`` so the chart breaks its line there even when buckets are wide. The
    result can exceed ``max_points`` by at most the number of segments.
    """
    if max_points < 1:
        raise ValueError("max_points must be positive")
    segments = _split_at_gaps(points)
    total = sum(len(s) for s in segments) or 1
    sampled: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        share = max(1, round(max_points * len(segment) / total))
        chunk = _bucket(segment, share, fields)
        if index > 0 and chunk:
            chunk[0]["gap"] = True
        sampled.extend(chunk)
    _mark_field_gaps(sampled, fields)
    return sampled


def _mark_field_gaps(points: list[dict[str, Any]], fields: Sequence[str]) -> None:
    """Per-field outages that rows alone do not show: the CCS811 warm-up leaves eCO2 empty
    while temperature keeps rows coming. A point that has a field after a point lacking it
    gets that field in ``gap_fields``, so the chart breaks that one line there."""
    for field in fields:
        seen = missing = False
        for point in points:
            if point.get(field) is None:
                missing = seen
                continue
            if missing:
                point.setdefault("gap_fields", []).append(field)
            seen, missing = True, False


def _payload_measurement_sql(field: str) -> str:
    return f"(payload_json->'measurements'->>'{field}')::double precision"


LATEST_SQL = """
    SELECT ts, office_temp_c, office_humidity_pct, office_eco2_ppm, office_tvoc_ppb,
           air_warming, sensor_online, stale, sample_age_sec, payload_json
    FROM office_env_sample
    WHERE node = $1
    ORDER BY ts DESC
    LIMIT 1
"""

# New readings come from payload_json, not their columns (module docstring). The
# json ``->>`` path is NULL on old rows, so they read as absent.
HISTORY_SQL = """
    SELECT t, {select_fields}
    FROM (
        SELECT ts AS t, {column_fields}, {payload_fields}
        FROM office_env_sample
        WHERE node = $1
          AND ts >= $2
    ) AS s
    WHERE {any_reading}
    ORDER BY t ASC
""".format(
    select_fields=", ".join(_FIELDS),
    column_fields=", ".join(_COLUMN_FIELDS),
    payload_fields=", ".join(f"{_payload_measurement_sql(f)} AS {f}" for f in _PAYLOAD_FIELDS),
    any_reading=" OR ".join(f"{f} IS NOT NULL" for f in _FIELDS),
)


async def run_query(sql: str, *args: Any, one: bool) -> Any:
    """One short-lived asyncpg connection per call (same as the cabinet routes)."""
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured")

    import asyncpg

    connection = await asyncpg.connect(dsn=database_url)
    try:
        if one:
            return await connection.fetchrow(sql, *args)
        return await connection.fetch(sql, *args)
    finally:
        await connection.close()


async def query_latest_row(*, node: str) -> Optional[Mapping[str, Any]]:
    """Newest sample row, stale or not (the stale flag is itself the signal)."""
    return await run_query(LATEST_SQL, node, one=True)


async def query_history_rows(*, node: str, hours: int) -> Sequence[Mapping[str, Any]]:
    cutoff = _now_utc() - timedelta(hours=hours)
    return await run_query(HISTORY_SQL, node, cutoff, one=False)


LatestQuery = Callable[..., Awaitable[Optional[Mapping[str, Any]]]]
HistoryQuery = Callable[..., Awaitable[Sequence[Mapping[str, Any]]]]
_latest_query: LatestQuery = query_latest_row
_history_query: HistoryQuery = query_history_rows


def _stats(
    raw_points: Sequence[Mapping[str, Any]],
    n: int,
    *,
    stat_keys: Sequence[tuple[str, str]] = _STAT_KEYS,
) -> dict[str, Any]:
    out: dict[str, Any] = {"n_raw": len(raw_points), "n": n}
    for field, key in stat_keys:
        values = [float(p[field]) for p in raw_points if p.get(field) is not None]
        out[f"{key}_min"] = min(values) if values else None
        out[f"{key}_max"] = max(values) if values else None
    return out


def _load_latest(
    row: Optional[Mapping[str, Any]],
    *,
    stale_after_sec: float,
    now: datetime,
    to_sample: Callable[[Mapping[str, Any]], dict[str, Any]] = row_to_sample,
    fields: Sequence[str] = _FIELDS,
    flags: Sequence[str] = _FLAGS,
) -> dict[str, Any]:
    """``ok`` needs BOTH a recent row (the publisher is alive) and a non-stale
    sample (the sensor itself is alive)."""
    if row is None:
        return {"ok": False, "age_sec": None, "sample": None}
    received_at = _parse_db_timestamp(row["ts"])
    age_sec = (now.astimezone(timezone.utc) - received_at).total_seconds()
    sensor_stale = bool(row.get("stale"))
    publisher_stale = age_sec > stale_after_sec
    sample = to_sample(row)
    if publisher_stale:
        # The publisher stopped: the newest row is history, not a current reading. Do not
        # hand the UI a number it could show as live.
        for field in (*fields, *flags):
            sample.pop(field, None)
        sample["sensor_online"] = False
    out: dict[str, Any] = {
        "ok": not publisher_stale and not sensor_stale,
        "age_sec": age_sec,
        "sample": sample,
        "sensor_stale": sensor_stale,
        "publisher_stale": publisher_stale,
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
        return {"ok": False, "age_sec": None, "sample": None, "error": "office_env_latest_unavailable"}
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
    max_points = int(settings.OFFICE_ENV_HISTORY_MAX_POINTS)
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
