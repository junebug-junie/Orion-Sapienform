"""Personality tab (personality graph spec rev 4, move 4 "Juniper feels it").

Built around Orion's choices, not Orion's prose. Every read is a proxy to orion-durable-runs, the
single writer of the personality tables:

- ``GET /personality``: the standalone page (embedded as the Hub "Personality" tab).
- ``GET /api/personality/frame``      -> ``/personality/frame`` (PersonalityFrameV1, what Prime reads)
- ``GET /api/personality/choices``    -> ``/personality/choices?limit=``: the newest choices, every
  lane, with what Orion picked, what Orion passed over (market rank), and "said nothing" declines.
- ``GET /api/personality/tendencies`` -> ``/personality/tendencies?limit=``: choice-built tendencies
  (rev 2 prose nodes never), each with example choices and Juniper's latest answer.
- ``POST /api/personality/tendencies/{disposition_id}/verdict`` body ``{"verdict": "yes"|"no"}`` ->
  the same path on durable-runs. A "no" retires the tendency there.
- ``GET /api/personality/arms``       -> ``/personality/arms?days=``: market decisions with the
  personality care arm vs the holdback arm, by outcome.

Upstream base URL: ``HUB_CURIOSITY_DURABLE_RUNS_URL`` (reused; no new key). Unset is a 503.
Upstream down, timed out, 5xx or non-JSON is a 502 with a named reason; an upstream 404/409/503
passes through with its own detail. Never a 500 for an upstream fault.
"""
from __future__ import annotations

import re
from typing import Any, Literal, Optional

import httpx
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict

from .settings import settings

router = APIRouter(prefix="/api/personality", tags=["personality"])
page_router = APIRouter(tags=["personality"])

_NO_CACHE = {"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0", "Pragma": "no-cache"}
UPSTREAM_TIMEOUT_SEC = 5.0
MAX_LIMIT = 200
_DISPOSITION_ID = re.compile(r"[A-Za-z0-9_.:\-]{1,128}")
_PASSTHROUGH = frozenset({404, 409, 503})


def _base_url() -> str:
    base = str(getattr(settings, "HUB_CURIOSITY_DURABLE_RUNS_URL", "") or "").strip()
    if not base:
        raise HTTPException(503, "durable_runs_url_not_configured (HUB_CURIOSITY_DURABLE_RUNS_URL)")
    return base.rstrip("/")


async def _upstream(path: str, params: Optional[dict[str, Any]] = None, *, json_body: Any = None) -> Any:
    """GET ``path`` (or POST ``json_body`` to it) on durable-runs."""
    base = _base_url()
    try:
        async with httpx.AsyncClient(base_url=base, timeout=UPSTREAM_TIMEOUT_SEC) as client:
            if json_body is not None:
                resp = await client.post(path, json=json_body)
            else:
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


class VerdictIn(BaseModel):
    """Juniper's answer to "Does this look like Orion?"."""

    model_config = ConfigDict(extra="forbid")
    verdict: Literal["yes", "no"]


@router.get("/choices")
async def choices(limit: int = Query(default=30, ge=1, le=MAX_LIMIT)) -> JSONResponse:
    return _ok(await _upstream("/personality/choices", {"limit": limit}))


@router.get("/tendencies")
async def tendencies(limit: int = Query(default=20, ge=1, le=100)) -> JSONResponse:
    return _ok(await _upstream("/personality/tendencies", {"limit": limit}))


@router.post("/tendencies/{disposition_id}/verdict")
async def tendency_verdict(disposition_id: str, body: VerdictIn) -> JSONResponse:
    """Juniper's yes / no, written by durable-runs. A no retires the tendency."""
    if not _DISPOSITION_ID.fullmatch(disposition_id):
        raise HTTPException(400, "disposition_id must be 1-128 of [A-Za-z0-9_.:-]")
    return _ok(await _upstream(f"/personality/tendencies/{disposition_id}/verdict",
                               json_body={"verdict": body.verdict}))


@router.get("/arms")
async def arms(days: int = Query(default=7, ge=1, le=30)) -> JSONResponse:
    return _ok(await _upstream("/personality/arms", {"days": days}))
