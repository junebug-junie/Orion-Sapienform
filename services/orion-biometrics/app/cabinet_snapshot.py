from __future__ import annotations

"""Load host-local cabinet sensor snapshot into BiometricsSampleV1.sensors.

Reads the atomic JSON file written by ``scripts/orion_cabinet_sensor_reader.py``
(typically ``/run/orion-sensors/latest.json``). Missing or unreadable files
leave ``sensors`` unset on the sample — never ``{}``, never zero-filled.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_STALE_STATUSES = frozenset({"stale", "error", "missing"})


def _parse_received_at(value: str) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_cabinet_sensors_snapshot(
    path: str | Path,
    *,
    stale_after_sec: float,
    now: Optional[datetime] = None,
) -> Optional[Dict[str, Any]]:
    """Return ``{frame, received_at, stale}`` or ``None`` to omit ``sensors``.

    Never raises on bad or missing files — logs and returns ``None`` instead.
    """
    snapshot_path = Path(path)
    if not snapshot_path.is_file():
        return None

    try:
        raw_text = snapshot_path.read_text(encoding="utf-8")
        data = json.loads(raw_text)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.warning("Cabinet sensor snapshot unreadable at %s: %s", snapshot_path, exc)
        return None

    if not isinstance(data, dict):
        logger.warning("Cabinet sensor snapshot at %s is not a JSON object", snapshot_path)
        return None

    frame = data.get("frame")
    if not isinstance(frame, dict):
        return None

    received_at = data.get("received_at")
    if not isinstance(received_at, str) or not received_at.strip():
        logger.warning("Cabinet sensor snapshot at %s missing received_at", snapshot_path)
        return None

    status = str(data.get("status") or "").lower()
    stale = status in _STALE_STATUSES

    if not stale:
        received_dt = _parse_received_at(received_at)
        if received_dt is None:
            stale = True
        else:
            now_dt = now or datetime.now(timezone.utc)
            age_sec = (now_dt.astimezone(timezone.utc) - received_dt).total_seconds()
            if age_sec > stale_after_sec:
                stale = True

    return {
        "frame": frame,
        "received_at": received_at,
        "stale": stale,
    }
