"""Temporal Self day view (Temporal Self rev 4): Orion's day as arcs, threads and regulation.

Read-only. Every data route is a thin proxy to orion-durable-runs, which owns the chronicle
(arcs, day frames, cursors) and the regulation thread:

- ``GET /temporal-self``: the standalone page (embedded as the Hub "Temporal Self" tab).
- ``GET /api/temporal-self/frame``      -> ``/temporal-self/frame`` (current day frame)
- ``GET /api/temporal-self/day/{day}``  -> ``/temporal-self/day/{day}`` (a closed day, or 404)
- ``GET /api/temporal-self/arcs``       -> ``/temporal-self/arcs?day_id=&kind=&limit=``
  (``arcs`` and ``day`` return slim arcs -- ARC_FIELDS -- unless ``full=true``)
- ``GET /api/temporal-self/threads``    -> ``/temporal-self/threads``
- ``GET /api/temporal-self/cursors``    -> ``/temporal-self/cursors``
- ``GET /api/temporal-self/regulation`` -> ``/regulation/state``
- ``GET /api/temporal-self/immune``     -> ``/health`` ``temporal_self.immune`` (+ chronicle)

Upstream base URL: ``HUB_CURIOSITY_DURABLE_RUNS_URL`` (the Hub's existing durable-runs base
URL; same service, so no second key to drift). Upstream down, timed out or answering non-JSON
is a 502 with a named reason; an upstream 404/503 passes through with its own detail (a day
that is not closed, the chronicle tables missing). Never a 500 for an upstream fault.
"""
from __future__ import annotations

import re
from typing import Any, Optional

import httpx
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse

from .settings import settings

router = APIRouter(prefix="/api/temporal-self", tags=["temporal-self"])
page_router = APIRouter(tags=["temporal-self"])

_NO_CACHE = {"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0", "Pragma": "no-cache"}
UPSTREAM_TIMEOUT_SEC = 5.0
_DAY_ID = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_KIND = re.compile(r"[a-z_]{1,40}")
# The fields the tab renders. Full arcs carry segments/evidence/body (~418 KB for one live day of
# 227 arcs); slim is the default, `full=true` returns upstream verbatim.
ARC_FIELDS = ("kind", "arc_id", "status", "began_at", "ended_at", "subject_ref", "subject_label",
              "cumulative_dwell_sec", "attention_returns", "carried_from_previous_day", "closed_reason")
_PASSTHROUGH = frozenset({404, 503})


def _base_url() -> str:
    base = str(getattr(settings, "HUB_CURIOSITY_DURABLE_RUNS_URL", "") or "").strip()
    if not base:
        raise HTTPException(503, "durable_runs_url_not_configured (HUB_CURIOSITY_DURABLE_RUNS_URL)")
    return base.rstrip("/")


async def _upstream(path: str, params: Optional[dict[str, Any]] = None) -> Any:
    base = _base_url()
    try:
        async with httpx.AsyncClient(base_url=base, timeout=UPSTREAM_TIMEOUT_SEC) as client:
            resp = await client.get(path, params={k: v for k, v in (params or {}).items() if v is not None})
    except (httpx.HTTPError, httpx.InvalidURL) as exc:
        raise HTTPException(502, f"durable_runs_unavailable: {type(exc).__name__} at {base}{path}") from exc
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail")
        except (ValueError, AttributeError):
            detail = None
        detail = str(detail or resp.text or "")[:300]
        if resp.status_code in _PASSTHROUGH:
            raise HTTPException(resp.status_code, detail or f"durable_runs_{resp.status_code}")
        raise HTTPException(502, f"durable_runs_error {resp.status_code} on {path}: {detail}")
    try:
        return resp.json()
    except ValueError as exc:
        raise HTTPException(502, f"durable_runs_bad_json on {path}") from exc


def _slim_arcs(arcs: Any) -> Any:
    if not isinstance(arcs, list):
        return arcs
    return [{k: a.get(k) for k in ARC_FIELDS} if isinstance(a, dict) else a for a in arcs]


def _ok(body: Any) -> JSONResponse:
    return JSONResponse(content=body, headers=_NO_CACHE)


@page_router.get("/temporal-self")
async def temporal_self_page() -> HTMLResponse:
    from .main import TEMPLATES_DIR, build_hub_ui_asset_version

    template = (TEMPLATES_DIR / "temporal_self.html").read_text(encoding="utf-8")
    return HTMLResponse(template.replace("{{HUB_UI_ASSET_VERSION}}", build_hub_ui_asset_version()),
                        headers=_NO_CACHE)


@router.get("/frame")
async def frame() -> JSONResponse:
    return _ok(await _upstream("/temporal-self/frame"))


@router.get("/day/{day_id}")
async def day(day_id: str, full: bool = Query(default=False)) -> JSONResponse:
    if not _DAY_ID.fullmatch(day_id):
        raise HTTPException(400, "day_id must be YYYY-MM-DD")
    body = await _upstream(f"/temporal-self/day/{day_id}")
    if not full and isinstance(body, dict):
        body = {**body, "arcs": _slim_arcs(body.get("arcs"))}
    return _ok(body)


@router.get("/arcs")
async def arcs(
    day_id: Optional[str] = Query(default=None),
    kind: Optional[str] = Query(default=None),
    limit: int = Query(default=2000, ge=1, le=2000),
    full: bool = Query(default=False),
) -> JSONResponse:
    if day_id is not None and not _DAY_ID.fullmatch(day_id):
        raise HTTPException(400, "day_id must be YYYY-MM-DD")
    if kind is not None and not _KIND.fullmatch(kind):
        raise HTTPException(400, "kind must be a lowercase arc kind")
    # limit defaults to 2000 (upstream's max), not upstream's 500: the tab shows the whole day.
    body = await _upstream("/temporal-self/arcs", {"day_id": day_id, "kind": kind, "limit": limit})
    if not full and isinstance(body, dict):
        body = {**body, "arcs": _slim_arcs(body.get("arcs"))}
    return _ok(body)


@router.get("/threads")
async def threads() -> JSONResponse:
    return _ok(await _upstream("/temporal-self/threads"))


@router.get("/cursors")
async def cursors() -> JSONResponse:
    return _ok(await _upstream("/temporal-self/cursors"))


@router.get("/regulation")
async def regulation() -> JSONResponse:
    return _ok(await _upstream("/regulation/state"))


@router.get("/immune")
async def immune() -> JSONResponse:
    health = await _upstream("/health")
    ts = (health or {}).get("temporal_self") if isinstance(health, dict) else None
    if not isinstance(ts, dict):
        raise HTTPException(503, "temporal_self thread is not reported by durable-runs /health")
    return _ok({"immune": ts.get("immune"), "chronicle": ts.get("chronicle"),
                "arousal_level": ts.get("arousal_level"), "since": ts.get("since"),
                "last_error": ts.get("last_error")})
