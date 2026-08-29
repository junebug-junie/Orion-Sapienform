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
from orion.telemetry.cabinet_snapshot_merge import load_merged_cabinet_sensors

from .settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cabinet/sensors", tags=["cabinet-sensors"])

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
