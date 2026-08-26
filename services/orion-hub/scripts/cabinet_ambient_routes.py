"""Hub read APIs for cabinet ambient-audio latest state and history."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Mapping, Optional, Sequence

from fastapi import APIRouter, HTTPException, Query
from pydantic import ValidationError

from orion.schemas.telemetry.ambient_audio import (
    AMBIENT_AUDIO_SCHEMA_V1,
    AmbientAudioSnapshotV1,
)

from .settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cabinet/ambient", tags=["cabinet-ambient"])

_STALE_STATUSES = frozenset({"stale", "error", "missing"})
_VALID_STATUSES = _STALE_STATUSES | {"ok"}
_WINDOW_HOURS = {"24h": 24, "3d": 72, "7d": 168}
_GRAIN_SEC = 30
_FRACTIONAL_TS_RE = re.compile(
    r"^(?P<head>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})\."
    r"(?P<frac>\d+)(?P<tz>Z|[+-]\d{2}(?::\d{2})?)$"
)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_db_timestamp(value: Any) -> datetime:
    """Parse Hub/Postgres timestamp values (ISO Z, or sql-writer varchar form)."""
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip().replace("Z", "+00:00")
        if "T" not in text and " " in text:
            text = text.replace(" ", "T", 1)
        # asyncpg/sql-writer often stores '+00' not '+00:00'
        if text.endswith("+00") and not text.endswith("+00:00"):
            text = text[:-3] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            # sql-writer stores variable-width fractional seconds; Python 3.10
            # requires exactly 3 or 6 digits after the decimal.
            match = _FRACTIONAL_TS_RE.match(text)
            if not match:
                raise
            frac = match.group("frac").ljust(6, "0")[:6]
            tz = match.group("tz")
            if tz == "Z":
                tz = "+00:00"
            elif len(tz) == 3:
                tz = f"{tz}:00"
            parsed = datetime.fromisoformat(f"{match.group('head')}.{frac}{tz}")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_received_at(value: str) -> Optional[datetime]:
    try:
        return _parse_db_timestamp(value)
    except (TypeError, ValueError):
        return None


def parse_window(window: str) -> int:
    """Return supported history window length in hours."""
    try:
        return _WINDOW_HOURS[window]
    except KeyError as exc:
        raise ValueError(f"unsupported ambient history window: {window}") from exc


def _iso_utc(value: Any) -> str:
    return _parse_db_timestamp(value).isoformat().replace("+00:00", "Z")


def rows_to_points(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Convert selected DB rows without inventing absent optional values."""
    points: list[dict[str, Any]] = []
    for row in rows:
        rms = row.get("rms")
        timestamp = row.get("t")
        if rms is None or timestamp is None:
            continue
        point: dict[str, Any] = {"t": _iso_utc(timestamp), "rms": float(rms)}
        if row.get("peak") is not None:
            point["peak"] = int(round(float(row["peak"])))
        if row.get("activity") is not None:
            point["activity"] = float(row["activity"])
        points.append(point)
    return points


def downsample_points(
    points: Sequence[Mapping[str, Any]], max_points: int
) -> list[dict[str, Any]]:
    """Bucket-average ordered points to at most ``max_points``."""
    if max_points < 1:
        raise ValueError("max_points must be positive")
    if len(points) <= max_points:
        return [dict(point) for point in points]

    sampled: list[dict[str, Any]] = []
    fields = ("rms", "peak", "activity")
    for bucket_index in range(max_points):
        start = bucket_index * len(points) // max_points
        end = (bucket_index + 1) * len(points) // max_points
        bucket = points[start:end]
        if not bucket:
            continue
        result: dict[str, Any] = {"t": bucket[0]["t"]}
        for field in fields:
            values = [float(point[field]) for point in bucket if point.get(field) is not None]
            if values:
                result[field] = sum(values) / len(values)
        sampled.append(result)
    return sampled


def _load_latest(
    path: str | Path, *, stale_after_sec: float, now: datetime
) -> dict[str, Any]:
    snapshot_path = Path(path)
    try:
        raw = json.loads(snapshot_path.read_text(encoding="utf-8"))
        snapshot = AmbientAudioSnapshotV1.model_validate(raw)
    except (
        OSError,
        json.JSONDecodeError,
        UnicodeDecodeError,
        ValidationError,
    ) as exc:
        logger.warning("Ambient audio snapshot unreadable at %s: %s", snapshot_path, exc)
        return {"ok": False, "age_sec": None, "snapshot": None}

    status = snapshot.status.lower()
    if snapshot.schema_ != AMBIENT_AUDIO_SCHEMA_V1 or status not in _VALID_STATUSES:
        logger.warning("Ambient audio snapshot at %s has unsupported schema/status", snapshot_path)
        return {"ok": False, "age_sec": None, "snapshot": None}

    received_at = _parse_received_at(snapshot.received_at)
    age_sec = None
    if received_at is not None:
        age_sec = (now.astimezone(timezone.utc) - received_at).total_seconds()
    stale = (
        status in _STALE_STATUSES
        or received_at is None
        or (age_sec is not None and age_sec > stale_after_sec)
    )
    return {
        "ok": not stale,
        "age_sec": age_sec,
        "snapshot": snapshot.model_dump(by_alias=True, exclude_none=True),
    }


async def query_history_rows(*, node: str, hours: int) -> Sequence[Mapping[str, Any]]:
    """Read ambient-bearing biometrics summaries from Hub's configured Postgres."""
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured")

    import asyncpg

    cutoff = _iso_utc(_now_utc() - timedelta(hours=hours))
    connection = await asyncpg.connect(dsn=database_url)
    try:
        return await connection.fetch(
            """
            SELECT
              timestamp AS t,
              (measurements->>'cabinet_ambient_rms')::double precision AS rms,
              (measurements->>'cabinet_ambient_peak')::double precision AS peak,
              (pressures->>'cabinet_ambient_audio_activity')::double precision AS activity
            FROM orion_biometrics_summary
            WHERE node = $1
              AND timestamp >= $2
              AND measurements ? 'cabinet_ambient_rms'
            ORDER BY timestamp ASC
            """,
            node,
            cutoff,
        )
    finally:
        await connection.close()


HistoryQuery = Callable[..., Awaitable[Sequence[Mapping[str, Any]]]]
_history_query: HistoryQuery = query_history_rows


def _stats(raw_points: Sequence[Mapping[str, Any]], n: int) -> dict[str, Any]:
    rms_values = [float(point["rms"]) for point in raw_points]
    activity_values = [
        float(point["activity"])
        for point in raw_points
        if point.get("activity") is not None
    ]
    return {
        "n_raw": len(raw_points),
        "n": n,
        "rms_min": min(rms_values) if rms_values else None,
        "rms_max": max(rms_values) if rms_values else None,
        "activity_max": max(activity_values) if activity_values else None,
    }


@router.get("/latest")
def api_cabinet_ambient_latest() -> dict[str, Any]:
    return _load_latest(
        str(settings.AMBIENT_AUDIO_PATH),
        stale_after_sec=float(settings.AMBIENT_AUDIO_STALE_AFTER_SEC),
        now=_now_utc(),
    )


@router.get("/history")
async def api_cabinet_ambient_history(
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
        "grain_sec": _GRAIN_SEC,
    }
    try:
        rows = await _history_query(node=node, hours=hours)
        raw_points = rows_to_points(rows)
        points = downsample_points(raw_points, max_points)
    except Exception as exc:
        logger.warning("Cabinet ambient history unavailable: %s", exc)
        return {
            "ok": False,
            **base,
            "points": [],
            "stats": _stats([], 0),
            "error": "ambient_history_unavailable",
        }

    return {
        "ok": True,
        **base,
        "points": points,
        "stats": _stats(raw_points, len(points)),
    }
