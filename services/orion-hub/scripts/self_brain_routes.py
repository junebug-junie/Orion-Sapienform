"""Read-only Self-brain API: realtime tail + playback range + window bounds.

Reads the substrate_brain_frame_log table directly from Postgres (same DB the
other substrate panels use, env POSTGRES_URI). Degrades to empty-with-200 when
the log is empty or POSTGRES_URI is unset. No writes.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from fastapi import APIRouter, Query

router = APIRouter(prefix="/api/self-brain", tags=["self-brain"])

_MAX_TAIL = 120
_DEFAULT_RANGE_MAX = 240


#: Static, small (6 entries, BrainRegionV1.dimension's whole Literal set) --
#: no reason to hit orion.metrics.lineage.build_graph() (which resolves ~630
#: nodes across the whole repo) on every page load for this. lru_cache(1)
#: builds it once, lazily, on first request -- not at import time, since a
#: module-level orion.metrics.lineage import here would run that resolution
#: on every Hub cold start whether this endpoint is ever hit or not. Tests
#: reset it via `_region_provenance.cache_clear()` (a real public method,
#: unlike poking a private module global directly -- review finding,
#: 2026-09-04).
@lru_cache(maxsize=1)
def _region_provenance() -> dict[str, dict[str, Any]]:
    from orion.metrics.lineage import resolve_brain_regions

    return {
        node.name: {
            "producer_service": node.producer_service,
            "urn": node.urn,
            "upstream": list(node.upstream),
        }
        for node in resolve_brain_regions()
    }


#: Process-wide engine, NOT one per request. `/frames/tail` is the hub's
#: single most-polled endpoint -- self-brain.js refreshes it every
#: TAIL_POLL_MS (3s), and it was measured at 163 requests in 5 minutes on
#: 2026-09-03. Building a fresh engine (and therefore a fresh QueuePool and a
#: fresh TCP connection) per request cost 24.6ms p50 against 1.2ms with a
#: cached one, and the discarded engines were never disposed, so their
#: connections lingered -- in a repo with live Postgres connection-exhaustion
#: history (PR #2010).
_ENGINE: Any = None
_ENGINE_URI: str = ""
_ENGINE_LOCK = threading.Lock()


def _engine():
    global _ENGINE, _ENGINE_URI
    uri = os.getenv("POSTGRES_URI", "").strip()
    if not uri:
        return None
    with _ENGINE_LOCK:
        if _ENGINE is not None and _ENGINE_URI == uri:
            return _ENGINE
        try:
            from sqlalchemy import create_engine

            # connect_timeout: SQLAlchemy has none by default, so an
            # unreachable-but-not-refusing Postgres blocks the worker thread
            # on the OS TCP timeout (minutes). Same value and rationale as
            # orion/substrate/mutation_control_surface.py::_engine.
            stale, _ENGINE = _ENGINE, create_engine(
                uri, pool_pre_ping=True, connect_args={"connect_timeout": 2}
            )
            _ENGINE_URI = uri
            if stale is not None:
                try:
                    stale.dispose()
                except Exception:
                    pass
            return _ENGINE
        except Exception:
            return None


def _coerce(value: Any) -> dict | None:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def _frames_tail_sync(limit: int) -> dict[str, Any]:
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


@router.get("/frames/tail")
async def frames_tail(limit: int = Query(default=1, ge=1, le=_MAX_TAIL)) -> dict[str, Any]:
    """Blocking DB work goes to a worker thread, never inline.

    SQLAlchemy here is synchronous, so calling it from this `async def` held
    the hub's event loop for the whole round trip -- and this is the most
    polled endpoint on the hub (every 3s from self-brain.js, 163 hits in 5
    minutes measured 2026-09-03). Same defect as
    /api/biometrics/preview/induction; see biometrics_preview_routes.py.

    Also decodes up to _MAX_TAIL frames of JSONB (`_coerce` -> json.loads),
    which is real CPU, not just I/O -- another reason it does not belong on
    the loop.
    """
    return await asyncio.to_thread(_frames_tail_sync, limit)


@router.get("/frames/range")
async def frames_range(
    from_: str = Query(alias="from"),
    to: str = Query(...),
    max: int = Query(default=_DEFAULT_RANGE_MAX, ge=1, le=2000),
) -> dict[str, Any]:
    return await asyncio.to_thread(_frames_range_sync, from_, to, max)


def _frames_range_sync(from_: str, to: str, max: int) -> dict[str, Any]:
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
    return await asyncio.to_thread(_window_sync)


def _window_sync() -> dict[str, Any]:
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


@router.get("/region-provenance")
async def region_provenance() -> dict[str, Any]:
    """Which service backs each of the 6 BrainRegionV1.dimension values --
    for the region detail panel's "what produced this" affordance. Static
    (see _region_provenance()'s own comment), so no worker-thread/DB dance
    needed here unlike every other route in this file."""
    return _region_provenance()
