"""Read-only Self-brain API: realtime tail + playback range + window bounds.

Reads the substrate_brain_frame_log table directly from Postgres (same DB the
other substrate panels use, env POSTGRES_URI). Degrades to empty-with-200 when
the log is empty or POSTGRES_URI is unset. No writes.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Query

router = APIRouter(prefix="/api/self-brain", tags=["self-brain"])

_MAX_TAIL = 120
_DEFAULT_RANGE_MAX = 240


def _engine():
    uri = os.getenv("POSTGRES_URI", "").strip()
    if not uri:
        return None
    try:
        from sqlalchemy import create_engine

        return create_engine(uri, pool_pre_ping=True)
    except Exception:
        return None


def _coerce(value: Any) -> dict | None:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


@router.get("/frames/tail")
async def frames_tail(limit: int = Query(default=1, ge=1, le=_MAX_TAIL)) -> dict[str, Any]:
    engine = _engine()
    if engine is None:
        return {"frames": [], "phase": None}
    from sqlalchemy import text

    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT frame_json FROM substrate_brain_frame_log "
                    "ORDER BY generated_at DESC LIMIT :limit"
                ),
                {"limit": int(limit)},
            ).mappings().all()
    except Exception:
        return {"frames": [], "phase": None}
    frames = [f for f in (_coerce(r["frame_json"]) for r in rows) if f]
    frames.reverse()
    phase = frames[-1].get("phase") if frames else None
    return {"frames": frames, "phase": phase}


@router.get("/frames/range")
async def frames_range(
    from_: str = Query(alias="from"),
    to: str = Query(...),
    max: int = Query(default=_DEFAULT_RANGE_MAX, ge=1, le=2000),
) -> dict[str, Any]:
    engine = _engine()
    if engine is None:
        return {"frames": []}
    from sqlalchemy import text

    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT frame_json FROM substrate_brain_frame_log "
                    "WHERE generated_at >= :start AND generated_at <= :end "
                    "ORDER BY generated_at ASC"
                ),
                {"start": from_, "end": to},
            ).mappings().all()
    except Exception:
        return {"frames": []}
    frames = [f for f in (_coerce(r["frame_json"]) for r in rows) if f]
    if len(frames) > max:
        step = len(frames) / max
        frames = [frames[int(i * step)] for i in range(max)]
    return {"frames": frames}


@router.get("/window")
async def window() -> dict[str, Any]:
    engine = _engine()
    empty = {
        "earliest": None,
        "latest": None,
        "frame_count": 0,
        "phase": None,
        "server_now": datetime.now(timezone.utc).isoformat(),
    }
    if engine is None:
        return empty
    from sqlalchemy import text

    try:
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT min(generated_at) AS earliest, max(generated_at) AS latest, "
                    "count(*) AS n FROM substrate_brain_frame_log"
                )
            ).mappings().first()
            phase_row = conn.execute(
                text(
                    "SELECT phase FROM substrate_brain_frame_log "
                    "ORDER BY generated_at DESC LIMIT 1"
                )
            ).mappings().first()
    except Exception:
        return empty

    def _iso(v):
        return v.isoformat() if hasattr(v, "isoformat") else v

    return {
        "earliest": _iso(row["earliest"]) if row else None,
        "latest": _iso(row["latest"]) if row else None,
        "frame_count": int(row["n"]) if row else 0,
        "phase": (phase_row["phase"] if phase_row else None),
        "server_now": datetime.now(timezone.utc).isoformat(),
    }
