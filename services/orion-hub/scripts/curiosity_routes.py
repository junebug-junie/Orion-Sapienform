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
from zoneinfo import ZoneInfo

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

    The daily counter is keyed on the OPERATOR'S LOCAL date, and the zone comes
    from the same setting the loop uses (`HUB_ENDOGENOUS_OUTREACH_TZ`), NOT from
    this process's own locale. The container sets no `TZ`, so `datetime.now()
    .astimezone()` here is UTC: between 18:00 and 23:59 in Juniper's zone that
    reads tomorrow's key, finds nothing, and reports `runs_today: 0` while the
    loop is at cap -- the "Runs today" tile would sit at 0 of 3 every evening
    and the at-cap highlight would never fire. Caught in review; the earlier
    version of this docstring claimed to avoid the exact bug it had.
    """
    out: dict[str, Any] = {
        "available": False,
        "local_date": None,
        "tz": None,
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

        try:
            tz = ZoneInfo(getattr(cfg, "HUB_ENDOGENOUS_OUTREACH_TZ", "UTC") or "UTC")
        except Exception:  # noqa: BLE001 -- same fallback the loop takes
            tz = timezone.utc
        local_date = datetime.now(timezone.utc).astimezone(tz).date().isoformat()
        last = await redis.get(_COOLDOWN_KEY)
        count = await redis.get(f"{_DAILY_COUNT_KEY_PREFIX}{local_date}")
        if isinstance(last, (bytes, bytearray)):
            last = last.decode("utf-8", errors="replace")
        if isinstance(count, (bytes, bytearray)):
            count = count.decode("utf-8", errors="replace")

        out["available"] = True
        out["local_date"] = local_date
        out["tz"] = str(tz)
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


async def _read_journals(run_ids: list[str]) -> dict[str, str]:
    """What each run actually SAID, which is its real output.

    The graph holds the run's structure -- priors, findings, hops -- but the
    prose Orion wrote for Juniper lives in Postgres, and a page that shows the
    node counts without it describes the shape of a turn while hiding what the
    turn was for. Juniper, looking at the first version of this page: "Don't
    know what tools are being used or what the actual output is of the run."

    Keyed `curiosity:<run_id>` by the loop (`curiosity_investigation.py:318`).
    """
    if not run_ids:
        return {}
    try:
        from . import main as hub_main

        pool = getattr(getattr(hub_main, "app", None), "state", None)
        pool = getattr(pool, "memory_pg_pool", None)
        if pool is None:
            return {}
        refs = [f"curiosity:{r}" for r in run_ids]
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT source_ref, body FROM journal_entries "
                "WHERE source_ref = ANY($1::text[])",
                refs,
            )
        return {
            str(r["source_ref"]).split(":", 1)[1]: str(r["body"] or "")
            for r in rows
            if ":" in str(r["source_ref"])
        }
    except Exception as exc:  # noqa: BLE001 -- a dashboard never 500s
        logger.warning("curiosity_atlas_journal_unavailable err=%s", exc)
        return {}


def _wrote_on(
    runs: list[dict[str, Any]], local_date: Optional[str], tz_name: Optional[str]
) -> Optional[int]:
    """How many runs wrote a node on the SAME calendar day the counter keys on.

    Computed here rather than in the browser. The daily counter is keyed in
    `HUB_ENDOGENOUS_OUTREACH_TZ`, while a browser's `isToday` is the viewer's
    own zone, and comparing a count from one zone against a count from another
    makes the "wrote nothing at all" banner fire or stay silent for reasons that
    have nothing to do with Orion. It happens to be right today because Juniper
    and the configured zone are both MDT; it would be wrong for any viewer who
    is not, and wrong for everyone for the hours the two dates disagree.

    An undated run counts as today: its only timestamp comes from a
    `:TurnOutcome`, so a turn killed mid-write has no date, and calling that
    "not today" would report it as traceless when it plainly left a trace.
    """
    if not local_date:
        return None
    try:
        tz = ZoneInfo(tz_name) if tz_name else timezone.utc
    except Exception:  # noqa: BLE001
        tz = timezone.utc
    n = 0
    for run in runs:
        if not run.get("total_added"):
            continue
        stamp = run.get("written_at")
        if not stamp:
            n += 1
            continue
        when = datetime.fromtimestamp(int(stamp) / 1000, timezone.utc).astimezone(tz)
        if when.date().isoformat() == local_date:
            n += 1
    return n


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
        # `to_payload` and not just `read_atlas`: building the payload walks
        # every prior's trajectory and deep-copies every dataclass, and Hub runs
        # one uvicorn worker -- doing that on the event loop stalls every
        # connected websocket, which is the rule `WorldviewReader` already
        # states for its own blocking call.
        payload = await asyncio.to_thread(lambda: to_payload(read_atlas(reader)))
        journals = await _read_journals(
            [r["run_id"] for r in payload.get("runs", []) if r.get("run_id")]
        )
        for run in payload.get("runs", []):
            run["journal"] = journals.get(run.get("run_id", ""), "")
    payload["schedule"] = await _read_schedule()
    payload["schedule"]["runs_wrote_today"] = _wrote_on(
        payload.get("runs", []), payload["schedule"].get("local_date"),
        payload["schedule"].get("tz"),
    )
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
