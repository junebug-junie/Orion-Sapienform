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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter

from orion.telemetry.cabinet_sensors import (
    CabinetPressureConfig,
    CabinetSensorTracker,
    compute_cabinet_pressures,
    extract_cabinet_measurements,
)

from .settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cabinet/sensors", tags=["cabinet-sensors"])

_STALE_STATUSES = frozenset({"stale", "error", "missing"})

_TRACKER = CabinetSensorTracker(CabinetPressureConfig())


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


def _compute_stale_and_age(
    snapshot: Dict[str, Any],
    *,
    stale_after_sec: float,
    now: datetime,
) -> tuple[bool, Optional[float]]:
    received_at = snapshot.get("received_at")
    age_sec: Optional[float] = None
    received_dt: Optional[datetime] = None
    if isinstance(received_at, str) and received_at.strip():
        received_dt = _parse_received_at(received_at)
        if received_dt is not None:
            age_sec = (now.astimezone(timezone.utc) - received_dt).total_seconds()

    status = str(snapshot.get("status") or "").lower()
    stale = status in _STALE_STATUSES
    if not stale:
        if received_dt is None:
            stale = True
        elif age_sec is not None and age_sec > stale_after_sec:
            stale = True
    return stale, age_sec


def build_cabinet_sensors_latest(
    *,
    sensors_path: str,
    boot_path: str,
    stale_after_sec: float,
    now: Optional[datetime] = None,
    tracker: Optional[CabinetSensorTracker] = None,
) -> Dict[str, Any]:
    """Assemble the /latest response payload (pure enough for unit tests)."""
    now_dt = now or _now_utc()
    active_tracker = tracker if tracker is not None else _TRACKER

    snapshot = _load_json_object(sensors_path)
    boot = _load_json_object(boot_path)

    if snapshot is None:
        return {
            "ok": False,
            "age_sec": None,
            "snapshot": None,
            "boot": boot,
            "measurements": {},
            "pressures": {},
        }

    frame = snapshot.get("frame")
    has_frame = isinstance(frame, dict)
    stale, age_sec = _compute_stale_and_age(
        snapshot, stale_after_sec=stale_after_sec, now=now_dt
    )

    measurements: Dict[str, float] = {}
    pressures: Dict[str, float] = {}

    if has_frame:
        sensors_payload = {
            "frame": frame,
            "received_at": snapshot.get("received_at"),
            "stale": stale,
        }
        if not stale:
            measurements = extract_cabinet_measurements(sensors_payload)
            if measurements:
                pressures.update(compute_cabinet_pressures(measurements, active_tracker))
        # Sensors payload exists (frame present) → always emit staleness 0/1.
        pressures["cabinet_sensor_staleness"] = 1.0 if stale else 0.0

    return {
        "ok": (not stale) and has_frame,
        "age_sec": age_sec,
        "snapshot": snapshot,
        "boot": boot,
        "measurements": measurements,
        "pressures": pressures,
    }


@router.get("/latest")
def api_cabinet_sensors_latest() -> Dict[str, Any]:
    return build_cabinet_sensors_latest(
        sensors_path=str(settings.CABINET_SENSORS_PATH),
        boot_path=str(settings.CABINET_BOOT_PATH),
        stale_after_sec=float(settings.CABINET_SENSORS_STALE_AFTER_SEC),
        now=_now_utc(),
        tracker=_TRACKER,
    )
