"""Personality panel (personality graph spec rev 2, section 7 "Observability"). Read-only.

What Orion keeps returning to, as the durable-runs personality reducer records it, and the exact
cue text every lane (chat stance, reverie, curiosity kickoff, outreach) is handed:

- ``GET /personality``: the standalone page (embedded as the Hub "Personality" tab).
- ``GET /api/personality/frame``        -> ``/personality/frame`` (PersonalityFrameV1)
- ``GET /api/personality/dispositions`` -> ``/personality/dispositions?status=&limit=``
- ``GET /api/personality/evidence``     -> ``/personality/evidence?disposition_id=&limit=``
- ``GET /api/personality/cue``          the frame rendered here with
  ``orion.personality.cue.project_personality_cue`` -- the same pure function the lanes call --
  at its DEFAULT ``max_age_sec``/``top_k``. That is exactly what the lanes get while their own
  knobs (orion-thought ``PERSONALITY_CUE_MAX_AGE_SEC``/``_TOP_K``/``_ENABLED``) sit at those
  defaults; the parameters used ride in ``render_params`` so a divergence is visible, not hidden.
  The Hub's fetch budget (5 s) is also longer than a lane's, so a lane can still get
  ``fetch_failed`` when this route rendered a cue. Fail-open: always 200; an upstream fault is
  reported in ``error``/``upstream_status`` next to an empty cue (``skip_reason=fetch_failed``),
  and a renderer fault is ``skip_reason=render_failed``.

Upstream base URL: ``HUB_CURIOSITY_DURABLE_RUNS_URL`` (reused; no new key). Unset is a 503.
Upstream down, timed out, 5xx or non-JSON is a 502 with a named reason; an upstream 404/503
passes through with its own detail. Never a 500 for an upstream fault.
"""
from __future__ import annotations

import re
from typing import Any, Optional

import httpx
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse

from orion.personality.cue import DEFAULT_MAX_AGE_SEC, DEFAULT_TOP_K, project_personality_cue
from orion.schemas.personality import PersonalityCueV1

from .settings import settings

router = APIRouter(prefix="/api/personality", tags=["personality"])
page_router = APIRouter(tags=["personality"])

_NO_CACHE = {"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0", "Pragma": "no-cache"}
UPSTREAM_TIMEOUT_SEC = 5.0
STATUSES = ("active", "candidate", "retired", "all")
MAX_LIMIT = 200
_DISPOSITION_ID = re.compile(r"[A-Za-z0-9_.:\-]{1,128}")
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


def _ok(body: Any) -> JSONResponse:
    return JSONResponse(content=body, headers=_NO_CACHE)


@page_router.get("/personality")
async def personality_page() -> HTMLResponse:
    from .main import TEMPLATES_DIR, build_hub_ui_asset_version

    template = (TEMPLATES_DIR / "personality.html").read_text(encoding="utf-8")
    return HTMLResponse(template.replace("{{HUB_UI_ASSET_VERSION}}", build_hub_ui_asset_version()),
                        headers=_NO_CACHE)


@router.get("/frame")
async def frame() -> JSONResponse:
    return _ok(await _upstream("/personality/frame"))


@router.get("/dispositions")
async def dispositions(
    status: str = Query(default="active"),
    limit: int = Query(default=50, ge=1, le=MAX_LIMIT),
) -> JSONResponse:
    if status not in STATUSES:
        raise HTTPException(400, f"status must be one of {', '.join(STATUSES)}")
    return _ok(await _upstream("/personality/dispositions", {"status": status, "limit": limit}))


@router.get("/evidence")
async def evidence(
    disposition_id: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=MAX_LIMIT),
) -> JSONResponse:
    if disposition_id is not None and not _DISPOSITION_ID.fullmatch(disposition_id):
        raise HTTPException(400, "disposition_id must be 1-128 of [A-Za-z0-9_.:-]")
    return _ok(await _upstream("/personality/evidence", {"disposition_id": disposition_id, "limit": limit}))


@router.get("/cue")
async def cue() -> JSONResponse:
    """The cue text exactly as the lanes render it. Never fails: an upstream fault rides along."""
    body: Any = None
    error: Optional[str] = None
    upstream_status: Optional[int] = None
    try:
        body = await _upstream("/personality/frame")
    except HTTPException as exc:
        error, upstream_status = str(exc.detail), exc.status_code
    try:
        # None -> fetch_failed; a body that does not validate -> invalid_frame.
        cue_body = project_personality_cue(body).model_dump(mode="json")
    except Exception as exc:  # noqa: BLE001 -- the renderer promises never to raise; backstop anyway
        from datetime import datetime, timezone

        cue_body = PersonalityCueV1(rendered_at=datetime.now(timezone.utc),
                                    skip_reason="render_failed").model_dump(mode="json")
        error = error or f"render_failed: {type(exc).__name__}"
    return _ok({
        "cue": cue_body,
        "render_params": {"max_age_sec": DEFAULT_MAX_AGE_SEC, "top_k": DEFAULT_TOP_K},
        "frame": body if isinstance(body, dict) else None,
        "error": error,
        "upstream_status": upstream_status,
    })
