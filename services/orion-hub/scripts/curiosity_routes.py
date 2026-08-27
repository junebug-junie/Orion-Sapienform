"""Curiosity Atlas — a read-only operator surface for Orion's world view.

WHY. On 2026-08-27 the accumulation loop went to zero and the only symptom was
one log line — `priors=0/0` — that nobody read for four hours. Everything needed
to see it was already in FalkorDB, Postgres, Redis and docker logs; nothing put
it on one screen. This module is that screen.

STRICTLY READ-ONLY, and that is a design constraint rather than a phase-one
scope cut. Hub never writes to `orion_worldview` — Orion authors its own graph —
and a surface that could edit it would need auth thinking, an audit trail, and a
story about what it means for an operator to overwrite a belief Orion formed.
There are no POST routes here and there should not be.

Follows `concept_atlas_routes.py`: two JSON GETs plus a standalone page route,
degrading to an honest "unavailable" payload rather than a 500, because this is
an interpretability surface and a broken panel must never take Hub down.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse

from orion.curiosity.atlas import read_atlas, to_payload
from orion.curiosity.worldview import WorldviewReader

logger = logging.getLogger("orion-hub.curiosity_routes")

router = APIRouter(prefix="/curiosity", tags=["curiosity"])

# Same keys the loop itself uses to reach Orion's graph, read through the same
# settings object -- a second copy of the host/port here is how a dashboard ends
# up describing a graph nobody is writing to.
_NO_CACHE = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}


def _build_reader() -> Optional[WorldviewReader]:
    """The same reader the loop builds, or None when the graph half is off.

    None and "unreachable" are deliberately different: a graph that was never
    configured is not an outage, and rendering the two identically is the exact
    confusion `worldview.read_snapshot` exists to prevent.
    """
    try:
        from app.settings import get_settings

        cfg = get_settings()
    except Exception as exc:  # noqa: BLE001 -- a dashboard never 500s on config
        logger.warning("curiosity_atlas_settings_unavailable err=%s", exc)
        return None
    host = getattr(cfg, "HUB_CURIOSITY_GRAPH_HOST", "") or ""
    user = getattr(cfg, "HUB_CURIOSITY_GRAPH_ORION_USER", "") or ""
    password = getattr(cfg, "HUB_CURIOSITY_GRAPH_ORION_PASSWORD", "") or ""
    if not (host and user and password):
        return None
    return WorldviewReader(
        host=host,
        port=int(getattr(cfg, "HUB_CURIOSITY_GRAPH_PORT", 6380) or 6380),
        graph_name=getattr(cfg, "HUB_CURIOSITY_GRAPH_OWN", "orion_worldview"),
    )


async def _read_schedule() -> dict[str, Any]:
    """Cooldown, daily cap and next-eligible — the half of the picture that
    lives in Redis rather than the graph.

    The key names are IMPORTED from the loop that writes them, never retyped.
    A dashboard reading `orion:curiosity:count:...` from its own string literal
    would keep rendering a confident 0 forever the day that prefix changes.

    The daily counter is keyed on the OPERATOR'S LOCAL date, which is the same
    convention the loop counts against -- reading it as UTC here would show the
    wrong bucket for most of Juniper's evening.
    """
    out: dict[str, Any] = {
        "available": False,
        "last_investigation_at": None,
        "next_eligible_at": None,
        "runs_today": None,
        "daily_cap": None,
        "cooldown_sec": None,
    }
    try:
        from app.settings import get_settings

        from . import main as hub_main
        from .curiosity_investigation import (
            _COOLDOWN_KEY,
            _DAILY_COUNT_KEY_PREFIX,
        )

        cfg = get_settings()
        out["daily_cap"] = int(
            getattr(cfg, "HUB_CURIOSITY_INVESTIGATION_DAILY_CAP", 0) or 0
        )
        cooldown = float(
            getattr(cfg, "HUB_CURIOSITY_INVESTIGATION_MIN_COOLDOWN_SEC", 0) or 0
        )
        out["cooldown_sec"] = cooldown

        redis = getattr(getattr(hub_main, "bus", None), "redis", None)
        if redis is None:
            return out

        local_date = datetime.now().astimezone().date().isoformat()
        last = await redis.get(_COOLDOWN_KEY)
        count = await redis.get(f"{_DAILY_COUNT_KEY_PREFIX}{local_date}")
        if isinstance(last, (bytes, bytearray)):
            last = last.decode("utf-8", errors="replace")
        if isinstance(count, (bytes, bytearray)):
            count = count.decode("utf-8", errors="replace")

        out["available"] = True
        out["last_investigation_at"] = str(last) if last else None
        out["runs_today"] = int(count) if count else 0
        if last and cooldown:
            try:
                stamped = datetime.fromisoformat(str(last))
                if stamped.tzinfo is None:
                    stamped = stamped.replace(tzinfo=timezone.utc)
                out["next_eligible_at"] = (
                    stamped + timedelta(seconds=cooldown)
                ).isoformat()
            except ValueError:
                # The loop writes this key, not Orion, so a malformed value is a
                # real defect worth seeing rather than swallowing.
                logger.warning("curiosity_atlas_bad_cooldown_stamp value=%r", last)
    except Exception as exc:  # noqa: BLE001 -- a dashboard never 500s
        logger.warning("curiosity_atlas_schedule_unavailable err=%s", exc)
    return out


@router.get("/api/atlas")
async def curiosity_atlas_api() -> JSONResponse:
    """Everything the page draws, in one read.

    One endpoint rather than four because every panel is a projection of the
    same graph read: splitting it would let the priors panel and the runs panel
    disagree about the same run, which is precisely the class of confusion this
    surface exists to remove.
    """
    reader = _build_reader()
    if reader is None:
        payload: dict[str, Any] = {
            "available": False,
            "reason": "graph_not_configured",
        }
    else:
        view = await asyncio.to_thread(read_atlas, reader)
        payload = to_payload(view)
    payload["schedule"] = await _read_schedule()
    return JSONResponse(content=payload, headers=_NO_CACHE)


@router.get("")
@router.get("/")
async def curiosity_atlas_page() -> HTMLResponse:
    from .main import TEMPLATES_DIR, build_hub_ui_asset_version

    template = (TEMPLATES_DIR / "curiosity_atlas.html").read_text(encoding="utf-8")
    rendered = template.replace(
        "{{HUB_UI_ASSET_VERSION}}", build_hub_ui_asset_version()
    )
    return HTMLResponse(content=rendered, status_code=200, headers=_NO_CACHE)
