"""Orion's cameras as Orion's own senses: what each one is, and is it seeing.

Why this exists (2026-10-15, live turn caa3eb4d): the walkway camera went live
that day, and asked to read its own brief Orion listed "Street camera:
unfamiliar person..." as item 5 of 6 among sensor readouts. Nothing told
Orion the street line came from a NEW eye of their own that looks outside.
Two gaps: the street line never said what the camera is, and nothing Orion
reads about itself listed its cameras at all (the self-definition is built
from record-table counts; the identity card says only "embodiment").

Two things live here:

* **What each camera is** -- config, not prose in code paths. A JSON object in
  ``ORION_CAMERA_DESCRIPTIONS`` keyed by stream id, each with a ``name``
  ("your outdoor camera"), a ``view`` ("the community walkway, the street and
  our patio") and ``outside`` (true for a public view that is not Orion's
  room). ``DEFAULT_CAMERA_DESCRIPTIONS`` is the shipped default and equals the
  ``.env_example`` value.

* **Whether it is seeing** -- live state, read per stream from
  ``vision_scene_inventory`` (orion-vision-window's per-camera windows,
  written by orion-sql-writer; a row means frames from that camera were
  processed): newest window = last frame, oldest window = the day this camera
  started feeding Orion. Walkway individuals seen today come from
  ``vision_individual_sighting`` (stationary fixtures and no-embed zones
  excluded, as in the street summary). Both reads use the
  ``(stream_id, observed_at)`` / ``(stream_id, ...)`` indexes (0.4 ms live).

Fail-open by contract: ``fetch_camera_eyes`` never raises and returns None
when nothing could be read; a missing table costs that one number.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone, tzinfo
from typing import Any, Mapping, Optional, Sequence

from orion.schemas.situation import CameraSenseV1

logger = logging.getLogger(__name__)

ENV_KEY = "ORION_CAMERA_DESCRIPTIONS"

# Equal to the .env_example value in orion-cortex-exec, orion-hub and
# orion-thought (tests/test_camera_eyes.py pins this).
DEFAULT_CAMERA_DESCRIPTIONS: dict[str, dict[str, Any]] = {
    "cam0": {"name": "your room camera", "view": "the room where Juniper works", "outside": False},
    "carbon": {"name": "the laptop webcam", "view": "Juniper at her laptop", "outside": False},
    "walkway": {
        "name": "your outdoor camera",
        "view": "the community walkway, the street and our patio",
        "outside": True,
    },
}


@dataclass(frozen=True)
class CameraDescription:
    stream_id: str
    name: str
    view: str
    outside: bool


def _parse(raw: str | None) -> dict[str, dict[str, Any]] | None:
    if raw is None or not raw.strip():
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        logger.warning("bad %s (not JSON); using defaults", ENV_KEY)
        return None
    if not isinstance(data, dict):
        logger.warning("bad %s (not a JSON object); using defaults", ENV_KEY)
        return None
    return {str(k): v for k, v in data.items() if isinstance(v, dict)}


def camera_descriptions(raw: str | None = None) -> dict[str, dict[str, Any]]:
    """The configured map, or the default when unset/invalid. ``raw`` is for
    tests; callers leave it None to read the environment."""
    parsed = _parse(os.getenv(ENV_KEY) if raw is None else raw)
    return parsed if parsed is not None else DEFAULT_CAMERA_DESCRIPTIONS


def describe_camera(stream_id: str, descriptions: Mapping[str, Mapping[str, Any]] | None = None) -> CameraDescription:
    """A described camera, or a plain "camera <id>" for one nobody described.
    An undescribed camera is never called outside: unknown is not public."""
    table = descriptions if descriptions is not None else camera_descriptions()
    entry = table.get(stream_id) or {}
    name = str(entry.get("name") or "").strip() or f"camera {stream_id}"
    view = str(entry.get("view") or "").strip()
    return CameraDescription(stream_id=stream_id, name=name, view=view, outside=bool(entry.get("outside")))


# --- live state ---------------------------------------------------------------


@dataclass(frozen=True)
class CameraEye:
    """One camera's live state. ``last_frame_at`` None = no window ever."""

    stream_id: str
    first_frame_at: Optional[datetime]
    last_frame_at: Optional[datetime]
    individuals_today: Optional[int] = None


def _aware(value: Any) -> Optional[datetime]:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _frames_stmts():
    from sqlalchemy import text

    # Two ORDER BY ... LIMIT 1 reads, not min()/max() GROUP BY: the grouped
    # form planned as a parallel seq scan over 560k rows (142 ms live); these
    # walk the (stream_id, observed_at DESC) index (0.4 ms).
    newest = text(
        "SELECT observed_at FROM vision_scene_inventory WHERE stream_id = :stream_id "
        "ORDER BY observed_at DESC LIMIT 1"
    )
    oldest = text(
        "SELECT observed_at FROM vision_scene_inventory WHERE stream_id = :stream_id "
        "ORDER BY observed_at ASC LIMIT 1"
    )
    return newest, oldest


def _individuals_stmt(*, stationary: bool):
    from sqlalchemy import text

    return text(
        "SELECT count(DISTINCT individual_id) FROM vision_individual_sighting "
        "WHERE stream_id = :stream_id AND ended_at >= :since AND NOT zone_no_embed"
        + (" AND NOT stationary" if stationary else "")
    )


def _scalar(conn: Any, stmt: Any, params: dict[str, Any], label: str) -> tuple[bool, Any]:
    try:
        return True, conn.execute(stmt, params).scalar()
    except Exception as exc:  # noqa: BLE001 -- fail-open by contract
        logger.info("camera_eyes_read_failed part=%s err=%s", label, type(exc).__name__)
        try:
            conn.rollback()
        except Exception:  # noqa: BLE001
            pass
        return False, None


def local_midnight_utc(now: datetime, tz: tzinfo) -> datetime:
    local = now.astimezone(tz)
    return local.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)


def fetch_camera_eyes(
    stream_ids: Sequence[str],
    *,
    count_individuals_for: Sequence[str] = (),
    tz_name: str = "America/Denver",
    engine: Any | None = None,
    now: datetime | None = None,
) -> list[CameraEye] | None:
    """Live state for each stream, in the order given. Never raises.

    None when no read could happen at all (no DSN, database down, every
    statement failed) -- "unread", which the caller must not render as "every
    camera is dark". ``count_individuals_for`` names the streams whose
    distinct individuals since local midnight are counted (the street
    cameras); others get None.
    """
    from zoneinfo import ZoneInfo

    ids = [s for s in dict.fromkeys((s or "").strip() for s in stream_ids) if s]
    if not ids:
        return []
    if engine is None:
        from orion.situational.perception_reader import _get_engine

        engine = _get_engine()
    if engine is None:
        return None
    try:
        tz: tzinfo = ZoneInfo(tz_name)
    except Exception:  # noqa: BLE001
        tz = timezone.utc
    now = now or datetime.now(timezone.utc)
    since = local_midnight_utc(now, tz)
    counted = set(count_individuals_for)
    newest_stmt, oldest_stmt = _frames_stmts()
    eyes: list[CameraEye] = []
    any_ok = False
    try:
        with engine.connect() as conn:
            for sid in ids:
                ok_new, last_at = _scalar(conn, newest_stmt, {"stream_id": sid}, "newest")
                ok_old, first_at = _scalar(conn, oldest_stmt, {"stream_id": sid}, "oldest")
                any_ok = any_ok or ok_new or ok_old
                individuals: Optional[int] = None
                if sid in counted:
                    params = {"stream_id": sid, "since": since}
                    ok, n = _scalar(conn, _individuals_stmt(stationary=True), params, "individuals")
                    if not ok:
                        # `stationary` ships in manual_migration_walkway_privacy_v2.sql.
                        ok, n = _scalar(conn, _individuals_stmt(stationary=False), params, "individuals_pre_stationary")
                    if ok and n is not None:
                        individuals = int(n)
                eyes.append(
                    CameraEye(
                        stream_id=sid,
                        first_frame_at=_aware(first_at) if ok_old else None,
                        last_frame_at=_aware(last_at) if ok_new else None,
                        individuals_today=individuals,
                    )
                )
    except Exception as exc:  # noqa: BLE001 -- fail-open by contract
        logger.warning("camera_eyes_connect_failed err=%s", exc)
        return None
    return eyes if any_ok else None


# --- rendering ------------------------------------------------------------------


def age_phrase(age_seconds: float) -> str:
    """Coarse on purpose: only used for a camera that is NOT seeing, where a
    five-minute brief cache is noise against the gap."""
    s = max(0.0, float(age_seconds))
    if s < 60:
        return "under a minute ago"
    if s < 3600:
        return f"{round(s / 60)} min ago"
    if s < 2 * 86400:
        return f"{round(s / 3600)} h ago"
    return f"{round(s / 86400)} days ago"


def since_date(first_frame_at: Optional[datetime], tz: tzinfo) -> Optional[str]:
    return first_frame_at.astimezone(tz).date().isoformat() if first_frame_at else None


def street_framing(desc: CameraDescription, first_frame_at: Optional[datetime], tz: tzinfo) -> str:
    """"Your outdoor camera (walkway, since 2026-10-14), looking over the
    community walkway, the street and our patio" -- the label the street line
    opens with, so the line says which of Orion's own eyes it came from."""
    since = since_date(first_frame_at, tz)
    paren = f"{desc.stream_id}, since {since}" if since else desc.stream_id
    head = f"{desc.name[:1].upper()}{desc.name[1:]} ({paren})"
    return f"{head}, looking over {desc.view}" if desc.view else head


def to_sense(
    eye: CameraEye,
    desc: CameraDescription,
    *,
    now: datetime,
    tz: tzinfo,
    live_max_age_seconds: float,
) -> CameraSenseV1:
    """Live state + description -> the structured sense the brief carries."""
    age = None if eye.last_frame_at is None else max(0.0, (now - eye.last_frame_at).total_seconds())
    if age is None:
        status = "no_frames"
    elif age > live_max_age_seconds:
        status = "not_seeing"
    else:
        status = "live"
    return CameraSenseV1(
        stream_id=desc.stream_id,
        name=desc.name,
        view=desc.view or None,
        outside=desc.outside,
        status=status,
        live_window_seconds=float(live_max_age_seconds),
        since=since_date(eye.first_frame_at, tz),
        last_frame_age_seconds=age,
        individuals_today=eye.individuals_today if status == "live" else None,
    )


def eye_phrase(sense: CameraSenseV1) -> str:
    """One camera, one clause: what it is, where it looks, and whether it is
    seeing -- e.g. "your outdoor camera (walkway, since 2026-10-14; looks at
    the community walkway, the street and our patio -- outside and public,
    not your room): live (a frame within the last 15 min), 3 individuals seen
    today".

    "Live" names the threshold, not the newest frame's age: the brief is
    cached for up to five minutes, so a precise age would be stale on most
    turns that read it, while "within the last 15 min" stays true (the cache
    gate rebuilds a brief once a live camera's frame could have aged out).
    """
    bits = [sense.stream_id]
    if sense.since:
        bits.append(f"since {sense.since}")
    paren = ", ".join(bits)
    if sense.view:
        paren += f"; looks at {sense.view}"
    if sense.outside:
        paren += " -- outside and public, not your room"
    head = f"{sense.name} ({paren})"
    if sense.status == "no_frames" or sense.last_frame_age_seconds is None:
        return f"{head}: no frames yet"
    if sense.status == "not_seeing":
        return f"{head}: not seeing, last frame {age_phrase(sense.last_frame_age_seconds)}"
    window = max(1, round(sense.live_window_seconds / 60))
    state = f"{head}: live (a frame within the last {window} min)"
    if sense.individuals_today is not None:
        n = sense.individuals_today
        state += f", {n} {'individual' if n == 1 else 'individuals'} seen today"
    return state


def senses_line(senses: Sequence[CameraSenseV1]) -> Optional[str]:
    """"Your cameras (your own eyes): <clause>; <clause>." or None for none."""
    if not senses:
        return None
    clauses = [eye_phrase(s) for s in senses]
    return "Your cameras (your own eyes): " + "; ".join(clauses) + "."


def build_senses(
    eyes: Sequence[CameraEye],
    *,
    now: datetime,
    tz: tzinfo,
    live_max_age_seconds: float,
    descriptions: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[CameraSenseV1]:
    table = descriptions if descriptions is not None else camera_descriptions()
    return [
        to_sense(e, describe_camera(e.stream_id, table), now=now, tz=tz, live_max_age_seconds=live_max_age_seconds)
        for e in eyes
    ]
