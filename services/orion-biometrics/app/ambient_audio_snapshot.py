from __future__ import annotations

"""Load host-local ambient audio snapshot into BiometricsSampleV1."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import ValidationError

from orion.schemas.telemetry.ambient_audio import (
    AMBIENT_AUDIO_SCHEMA_V1,
    AmbientAudioSnapshotV1,
)

logger = logging.getLogger(__name__)

_STALE_STATUSES = frozenset({"stale", "error", "missing"})
_VALID_STATUSES = _STALE_STATUSES | {"ok"}


def _parse_received_at(value: str) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_ambient_audio_snapshot(
    path: str | Path,
    *,
    stale_after_sec: float,
    now: Optional[datetime] = None,
) -> Optional[Dict[str, Any]]:
    """Return validated audio levels or ``None`` to omit ``ambient_audio``."""
    snapshot_path = Path(path)
    if not snapshot_path.is_file():
        return None

    try:
        data = json.loads(snapshot_path.read_text(encoding="utf-8"))
        snapshot = AmbientAudioSnapshotV1.model_validate(data)
    except (
        OSError,
        json.JSONDecodeError,
        UnicodeDecodeError,
        ValidationError,
    ) as exc:
        logger.warning("Ambient audio snapshot unreadable at %s: %s", snapshot_path, exc)
        return None

    if snapshot.schema_ != AMBIENT_AUDIO_SCHEMA_V1 or snapshot.status.lower() not in _VALID_STATUSES:
        logger.warning(
            "Ambient audio snapshot at %s has unsupported schema/status",
            snapshot_path,
        )
        return None

    stale = snapshot.status.lower() in _STALE_STATUSES
    if not stale:
        received_dt = _parse_received_at(snapshot.received_at)
        if received_dt is None:
            stale = True
        else:
            now_dt = now or datetime.now(timezone.utc)
            age_sec = (now_dt.astimezone(timezone.utc) - received_dt).total_seconds()
            stale = age_sec > stale_after_sec

    return {
        "rms": snapshot.rms,
        "peak": snapshot.peak,
        "received_at": snapshot.received_at,
        "stale": stale,
        "device": snapshot.device,
        "window_sec": snapshot.window_sec,
    }
