from __future__ import annotations

"""Load host-local cabinet sensor snapshot into BiometricsSampleV1.sensors.

Reads atomic JSON from ``scripts/orion_cabinet_sensor_reader.py`` (primary
``/run/orion-sensors/latest.json``, optional secondary
``/run/orion-sensors/b/latest.json``). Missing or unreadable files leave
``sensors`` unset on the sample — never ``{}``, never zero-filled.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from orion.telemetry.cabinet_snapshot_merge import load_merged_cabinet_sensors


def load_cabinet_sensors_snapshot(
    path: str | Path,
    *,
    stale_after_sec: float,
    now: Optional[datetime] = None,
    secondary_path: str | Path | None = None,
) -> Optional[Dict[str, Any]]:
    """Return ``{frame, received_at, stale, sources?}`` or ``None``.

    Never raises on bad or missing files.
    """
    secondary = str(secondary_path).strip() if secondary_path else None
    return load_merged_cabinet_sensors(
        path,
        secondary_path=secondary or None,
        stale_after_sec=stale_after_sec,
        now=now,
    )
