"""Hub operator API for Athena Nano cabinet sensor snapshots.

Reads host files written by ``scripts/orion_cabinet_sensor_reader.py``
(typically under ``/run/orion-sensors/``). Reuses
``orion.telemetry.cabinet_sensors`` helpers; Hub pressures use a
process-local ``CabinetSensorTracker`` (operator-debug approximations,
not shared with biometrics baselines).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, Mapping, Optional, Sequence

from fastapi import APIRouter, HTTPException, Query

from orion.telemetry.cabinet_sensors import (
    CabinetPressureConfig,
    CabinetSensorTracker,
    compute_cabinet_pressures,
    extract_cabinet_measurements,
)
from orion.telemetry.cabinet_snapshot_merge import load_merged_cabinet_sensors

from .cabinet_ambient_routes import downsample_points, parse_window
from .settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cabinet/sensors", tags=["cabinet-sensors"])

_TRACKER = CabinetSensorTracker(CabinetPressureConfig())
_GRAIN_SEC = 30

_SENSOR_HISTORY_SERIES: tuple[tuple[str, str], ...] = (
    ("temp_c", "cabinet_temp_c"),
    ("humidity_pct", "cabinet_humidity_pct"),
    ("lidar_mm", "cabinet_lidar_mm"),
    ("als_raw", "cabinet_als_raw"),
    ("climate_activity", "cabinet_climate_activity"),
    ("proximity_activity", "cabinet_proximity_activity"),
    ("uv_activity", "cabinet_uv_activity"),
)

HistoryQuery = Callable[..., Awaitable[Sequence[Mapping[str, Any]]]]
_history_query: HistoryQuery | None = None


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_received_at(value: str) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _load_json_object(path: str | Path) -> Optional[Dict[str, Any]]:
    """Best-effort JSON object load. Missing/unreadable → None (never raises)."""
    snapshot_path = Path(path)
    if not snapshot_path.is_file():
        return None
    try:
        raw_text = snapshot_path.read_text(encoding="utf-8")
        data = json.loads(raw_text)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.warning("Cabinet sensor JSON unreadable at %s: %s", snapshot_path, exc)
        return None
    if not isinstance(data, dict):
        logger.warning("Cabinet sensor JSON at %s is not an object", snapshot_path)
        return None
    return data


def build_cabinet_sensors_latest(
    *,
    sensors_path: str,
    boot_path: str,
    stale_after_sec: float,
    now: Optional[datetime] = None,
    tracker: Optional[CabinetSensorTracker] = None,
    sensors_b_path: str = "",
    boot_b_path: str = "",
) -> Dict[str, Any]:
    """Assemble the /latest response payload (pure enough for unit tests)."""
    now_dt = now or _now_utc()
    active_tracker = tracker if tracker is not None else _TRACKER

    secondary_sensors = (sensors_b_path or "").strip()
    secondary_boot = (boot_b_path or "").strip()

    merged_payload = load_merged_cabinet_sensors(
        sensors_path,
        secondary_path=secondary_sensors or None,
        stale_after_sec=stale_after_sec,
        now=now_dt,
    )

    boot = _load_json_object(boot_path)
    boot_b = _load_json_object(secondary_boot) if secondary_boot else None

    primary_raw = _load_json_object(sensors_path)
    secondary_raw = _load_json_object(secondary_sensors) if secondary_sensors else None

    sources: Dict[str, Any] = {}
    if primary_raw is not None:
        sources["a"] = {"snapshot": primary_raw, "boot": boot}
    if secondary_raw is not None:
        sources["b"] = {"snapshot": secondary_raw, "boot": boot_b}

    if merged_payload is None:
        return {
            "ok": False,
            "age_sec": None,
            "snapshot": None,
            "boot": boot,
            "sources": sources or None,
            "measurements": {},
            "pressures": {},
        }

    merged_frame = merged_payload["frame"]
    received_at = merged_payload.get("received_at")
    stale = bool(merged_payload.get("stale"))
    age_sec: Optional[float] = None
    if isinstance(received_at, str) and received_at.strip():
        received_dt = _parse_received_at(received_at)
        if received_dt is not None:
            age_sec = (now_dt.astimezone(timezone.utc) - received_dt).total_seconds()

    snapshot = {
        "status": "ok" if not stale else "stale",
        "received_at": received_at,
        "device": None,
        "frame": merged_frame,
    }
    if merged_payload.get("sources"):
        devices = [
            meta.get("device")
            for meta in merged_payload["sources"].values()
            if isinstance(meta, dict) and meta.get("device")
        ]
        if len(devices) == 1:
            snapshot["device"] = devices[0]
        elif devices:
            snapshot["device"] = " + ".join(devices)

    measurements: Dict[str, float] = {}
    pressures: Dict[str, float] = {}

    sensors_payload = {
        "frame": merged_frame,
        "received_at": received_at,
        "stale": stale,
    }
    if not stale:
        measurements = extract_cabinet_measurements(sensors_payload)
        if measurements:
            pressures.update(compute_cabinet_pressures(measurements, active_tracker))
    pressures["cabinet_sensor_staleness"] = 1.0 if stale else 0.0

    return {
        "ok": (not stale) and isinstance(merged_frame, dict),
        "age_sec": age_sec,
        "snapshot": snapshot,
        "boot": boot,
        "sources": sources or None,
        "measurements": measurements,
        "pressures": pressures,
    }


@router.get("/latest")
def api_cabinet_sensors_latest() -> Dict[str, Any]:
    return build_cabinet_sensors_latest(
        sensors_path=str(settings.CABINET_SENSORS_PATH),
        boot_path=str(settings.CABINET_BOOT_PATH),
        stale_after_sec=float(settings.CABINET_SENSORS_STALE_AFTER_SEC),
        sensors_b_path=str(settings.CABINET_SENSORS_B_PATH),
        boot_b_path=str(settings.CABINET_BOOT_B_PATH),
        now=_now_utc(),
        tracker=_TRACKER,
    )


def _iso_utc(value: Any) -> str:
    from .cabinet_ambient_routes import _parse_db_timestamp

    return _parse_db_timestamp(value).isoformat().replace("+00:00", "Z")


def rows_to_sensor_series(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Convert SQL rows to per-series [{t, v}, ...] without zero-filling gaps."""
    series: dict[str, list[dict[str, Any]]] = {key: [] for key, _ in _SENSOR_HISTORY_SERIES}
    for row in rows:
        timestamp = row.get("t")
        if timestamp is None:
            continue
        iso_t = _iso_utc(timestamp)
        for out_key, _column in _SENSOR_HISTORY_SERIES:
            raw = row.get(out_key)
            if raw is None:
                continue
            series[out_key].append({"t": iso_t, "v": float(raw)})
    return series


def _series_stats(raw_series: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for key, points in raw_series.items():
        values = [float(point["v"]) for point in points if point.get("v") is not None]
        if not values:
            continue
        stats[key] = {
            "n_raw": len(values),
            "min": min(values),
            "max": max(values),
        }
    return stats


def _downsample_series(
    points: Sequence[Mapping[str, Any]], max_points: int
) -> list[dict[str, Any]]:
    if not points:
        return []
    bucket_input = [{"t": p["t"], "rms": p["v"]} for p in points]
    down = downsample_points(bucket_input, max_points)
    return [{"t": p["t"], "v": p["rms"]} for p in down if p.get("rms") is not None]


async def query_sensor_history_rows(*, node: str, hours: int) -> Sequence[Mapping[str, Any]]:
    """Read cabinet Nano measurements from biometrics summaries."""
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
              (measurements->>'cabinet_temp_c')::double precision AS temp_c,
              (measurements->>'cabinet_humidity_pct')::double precision AS humidity_pct,
              (measurements->>'cabinet_lidar_mm')::double precision AS lidar_mm,
              (measurements->>'cabinet_als_raw')::double precision AS als_raw,
              (pressures->>'cabinet_climate_activity')::double precision AS climate_activity,
              (pressures->>'cabinet_proximity_activity')::double precision AS proximity_activity,
              (pressures->>'cabinet_uv_activity')::double precision AS uv_activity
            FROM orion_biometrics_summary
            WHERE node = $1
              AND timestamp >= $2
              AND measurements ? 'cabinet_temp_c'
            ORDER BY timestamp ASC
            """,
            node,
            cutoff,
        )
    finally:
        await connection.close()


@router.get("/history")
async def api_cabinet_sensors_history(
    window: str = Query("24h"),
) -> dict[str, Any]:
    node = str(settings.CABINET_AMBIENT_HISTORY_NODE)
    max_points = int(settings.CABINET_AMBIENT_HISTORY_MAX_POINTS)
    query = _history_query or query_sensor_history_rows
    try:
        hours = parse_window(window)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    base = {"node": node, "window": window, "grain_sec": _GRAIN_SEC}
    try:
        rows = await query(node=node, hours=hours)
        raw_series = rows_to_sensor_series(rows)
        sampled = {
            key: _downsample_series(points, max_points)
            for key, points in raw_series.items()
        }
    except Exception as exc:
        logger.warning("Cabinet sensor history unavailable: %s", exc)
        empty = {key: [] for key, _ in _SENSOR_HISTORY_SERIES}
        return {
            "ok": False,
            **base,
            "series": empty,
            "stats": {},
            "error": "sensor_history_unavailable",
        }

    return {
        "ok": True,
        **base,
        "series": sampled,
        "stats": _series_stats(raw_series),
    }
