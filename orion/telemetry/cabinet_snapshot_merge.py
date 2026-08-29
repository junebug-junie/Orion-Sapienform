from __future__ import annotations

"""Merge host-local cabinet Nano snapshots from one or two USB readers.

Primary + optional secondary atomic JSON files (see
``scripts/orion_cabinet_sensor_reader.py``) are loaded and merged into a
single ``orion.sensor_frame.v1`` frame for biometrics and Hub. Each channel
comes from the newest non-stale source that reports it.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from orion.schemas.telemetry.cabinet_sensor_frame import FRAME_SCHEMA_V1

logger = logging.getLogger(__name__)

_STALE_STATUSES = frozenset({"stale", "error", "missing"})

FRAME_CHANNEL_KEYS = (
    "environment",
    "uv",
    "magnetic",
    "particulate",
    "lidar",
    "imu",
)


@dataclass(frozen=True)
class CabinetSnapshotSource:
    """One reader's loaded snapshot."""

    source_id: str
    frame: Optional[Dict[str, Any]]
    received_at: Optional[str]
    stale: bool
    status: Optional[str]
    device: Optional[str]
    raw: Optional[Dict[str, Any]]


def _parse_received_at(value: str) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _is_snapshot_stale(
    data: Dict[str, Any],
    *,
    stale_after_sec: float,
    now: datetime,
) -> bool:
    status = str(data.get("status") or "").lower()
    if status in _STALE_STATUSES:
        return True
    received_at = data.get("received_at")
    if not isinstance(received_at, str) or not received_at.strip():
        return True
    received_dt = _parse_received_at(received_at)
    if received_dt is None:
        return True
    age_sec = (now.astimezone(timezone.utc) - received_dt).total_seconds()
    return age_sec > stale_after_sec


def load_cabinet_snapshot_file(
    path: str | Path,
    *,
    source_id: str,
    stale_after_sec: float,
    now: Optional[datetime] = None,
) -> Optional[CabinetSnapshotSource]:
    """Load one reader snapshot. Returns None when file missing or unusable."""
    snapshot_path = Path(path)
    if not snapshot_path.is_file():
        return None

    try:
        data = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.warning("Cabinet sensor snapshot unreadable at %s: %s", snapshot_path, exc)
        return None

    if not isinstance(data, dict):
        logger.warning("Cabinet sensor snapshot at %s is not a JSON object", snapshot_path)
        return None

    frame = data.get("frame")
    if frame is not None and not isinstance(frame, dict):
        frame = None

    received_at = data.get("received_at")
    if received_at is not None and not isinstance(received_at, str):
        received_at = None

    now_dt = now or datetime.now(timezone.utc)
    stale = _is_snapshot_stale(data, stale_after_sec=stale_after_sec, now=now_dt)

    return CabinetSnapshotSource(
        source_id=source_id,
        frame=frame,
        received_at=received_at,
        stale=stale,
        status=str(data.get("status") or "") or None,
        device=str(data.get("device") or "") or None,
        raw=data,
    )


def _channel_block(source: CabinetSnapshotSource, key: str, *, trust_stale: bool = False) -> Optional[Dict[str, Any]]:
    if source.stale and not trust_stale:
        return None
    if not isinstance(source.frame, dict):
        return None
    block = source.frame.get(key)
    return block if isinstance(block, dict) else None


def merge_cabinet_frame_channels(
    sources: list[CabinetSnapshotSource],
) -> Dict[str, Any]:
    """Merge sensor sub-objects from multiple sources into one frame dict."""
    merged: Dict[str, Any] = {"schema": FRAME_SCHEMA_V1}
    best_seq = -1
    best_uptime = -1
    newest_received: Optional[datetime] = None

    for source in sources:
        if not isinstance(source.frame, dict):
            continue
        seq = source.frame.get("seq")
        if isinstance(seq, int) and seq > best_seq:
            best_seq = seq
        uptime = source.frame.get("uptime_ms")
        if isinstance(uptime, int) and uptime > best_uptime:
            best_uptime = uptime
        if source.received_at:
            received_dt = _parse_received_at(source.received_at)
            if received_dt is not None and (
                newest_received is None or received_dt > newest_received
            ):
                newest_received = received_dt

    if best_seq >= 0:
        merged["seq"] = best_seq
    if best_uptime >= 0:
        merged["uptime_ms"] = best_uptime

    for key in FRAME_CHANNEL_KEYS:
        winner: Optional[CabinetSnapshotSource] = None
        winner_at: Optional[datetime] = None
        for source in sources:
            block = _channel_block(source, key)
            if block is None:
                continue
            received_dt = (
                _parse_received_at(source.received_at)
                if source.received_at
                else None
            )
            if winner is None:
                winner = source
                winner_at = received_dt
                merged[key] = block
                continue
            if received_dt is not None and (winner_at is None or received_dt > winner_at):
                winner = source
                winner_at = received_dt
                merged[key] = block
        _ = winner

    return merged


def merge_cabinet_sensors_payload(
    sources: list[CabinetSnapshotSource],
) -> Optional[Dict[str, Any]]:
    """Return biometrics-shaped ``{frame, received_at, stale, sources}``."""
    usable = [s for s in sources if s.frame is not None]
    if not usable:
        return None

    merged_frame = merge_cabinet_frame_channels(sources)
    has_channel = any(key in merged_frame for key in FRAME_CHANNEL_KEYS)
    if not has_channel:
        return None

    fresh_contributors = [
        s
        for s in sources
        if not s.stale and any(_channel_block(s, key) is not None for key in FRAME_CHANNEL_KEYS)
    ]
    stale = not fresh_contributors

    newest_received: Optional[str] = None
    newest_dt: Optional[datetime] = None
    for source in fresh_contributors:
        if not source.received_at:
            continue
        received_dt = _parse_received_at(source.received_at)
        if received_dt is None:
            continue
        if newest_dt is None or received_dt > newest_dt:
            newest_dt = received_dt
            newest_received = source.received_at

    if newest_received is None:
        for source in sources:
            if source.received_at:
                newest_received = source.received_at
                break

    source_debug: Dict[str, Any] = {}
    for source in sources:
        source_debug[source.source_id] = {
            "received_at": source.received_at,
            "stale": source.stale,
            "status": source.status,
            "device": source.device,
        }

    return {
        "frame": merged_frame,
        "received_at": newest_received,
        "stale": stale,
        "sources": source_debug,
    }


def load_merged_cabinet_sensors(
    primary_path: str | Path,
    *,
    secondary_path: str | Path | None = None,
    stale_after_sec: float,
    now: Optional[datetime] = None,
    primary_id: str = "a",
    secondary_id: str = "b",
) -> Optional[Dict[str, Any]]:
    """Load primary and optional secondary snapshots and merge for downstream."""
    primary = load_cabinet_snapshot_file(
        primary_path,
        source_id=primary_id,
        stale_after_sec=stale_after_sec,
        now=now,
    )
    sources: list[CabinetSnapshotSource] = []
    if primary is not None:
        sources.append(primary)

    if secondary_path:
        secondary = load_cabinet_snapshot_file(
            secondary_path,
            source_id=secondary_id,
            stale_after_sec=stale_after_sec,
            now=now,
        )
        if secondary is not None:
            sources.append(secondary)

    if not sources:
        return None

    if len(sources) == 1:
        only = sources[0]
        if only.frame is None or not isinstance(only.received_at, str):
            return None
        return {
            "frame": only.frame,
            "received_at": only.received_at,
            "stale": only.stale,
            "sources": {
                only.source_id: {
                    "received_at": only.received_at,
                    "stale": only.stale,
                    "status": only.status,
                    "device": only.device,
                }
            },
        }

    return merge_cabinet_sensors_payload(sources)
