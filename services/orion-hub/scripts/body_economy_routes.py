"""Body economy tab (Phase 3 of docs/superpowers/specs/2026-10-23-organism-body-economy-design.md).

Read-only. The page is ``static/body-economy.html`` (embedded as the Hub "Body economy" tab). Its one
data route is a thin proxy to orion-durable-runs, which owns the physiology report:

- ``GET /api/body-economy/report`` -> ``/physiology/report`` (schema ``physiology.report.v1``)
  query: ``window=24h|7d|30d``, ``sections=``, ``family=``, ``outcome=funded|deferred``, ``limit=``

Upstream base URL: ``HUB_CURIOSITY_DURABLE_RUNS_URL`` (the Hub's existing durable-runs base URL;
same service, so no second key to drift). The Hub never queries physiology tables itself.

Upstream down, timed out or answering non-JSON is a 502. Its JSON body names the reason and
``unreachable_since`` (the first failure since the last good answer), so the page can say
"durable-runs unreachable since ..." instead of drawing blank charts. An upstream 4xx/503 passes
through with its own detail. Never a 500 for an upstream fault.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from .settings import settings

router = APIRouter(prefix="/api/body-economy", tags=["body-economy"])

_NO_CACHE = {"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0", "Pragma": "no-cache"}
UPSTREAM_PATH = "/physiology/report"
# H5 budget is 2 s for the page; the upstream reads hourly rollups. 8 s leaves room for a 30 d window
# on a busy host without hanging the tab.
UPSTREAM_TIMEOUT_SEC = 8.0
WINDOWS = ("24h", "7d", "30d")
_SECTIONS = re.compile(r"[a-z_]{1,40}(,[a-z_]{1,40}){0,20}")
_FAMILY = re.compile(r"[A-Za-z0-9_.:\-]{1,80}")
_PASSTHROUGH = frozenset({400, 404, 422, 503})

# Process-local reachability memory: when the upstream last answered, and when it first failed after
# that. Only for the "unreachable since" message; nothing decides on it.
_reach: dict[str, Optional[str]] = {"last_ok_at": None, "unreachable_since": None}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _mark_ok() -> None:
    _reach["last_ok_at"] = _now_iso()
    _reach["unreachable_since"] = None


def _mark_down() -> str:
    if not _reach["unreachable_since"]:
        _reach["unreachable_since"] = _now_iso()
    return str(_reach["unreachable_since"])


def _err(status: int, reason: str, *, down: bool) -> JSONResponse:
    body: dict[str, Any] = {"detail": reason, "last_ok_at": _reach["last_ok_at"]}
    if down:
        body["unreachable_since"] = _mark_down()
    return JSONResponse(status_code=status, content=body, headers=_NO_CACHE)


@router.get("/report")
async def report(
    window: str = Query(default="24h"),
    sections: Optional[str] = Query(default=None),
    family: Optional[str] = Query(default=None),
    outcome: Optional[str] = Query(default=None),
    limit: int = Query(default=500, ge=1, le=2000),
) -> JSONResponse:
    if window not in WINDOWS:
        return _err(400, f"window must be one of {', '.join(WINDOWS)}", down=False)
    if sections is not None and not _SECTIONS.fullmatch(sections):
        return _err(400, "sections must be comma-separated section names", down=False)
    if family is not None and not _FAMILY.fullmatch(family):
        return _err(400, "family must be a want family name", down=False)
    if outcome is not None and outcome not in ("funded", "deferred"):
        return _err(400, "outcome must be funded or deferred", down=False)

    base = str(getattr(settings, "HUB_CURIOSITY_DURABLE_RUNS_URL", "") or "").strip().rstrip("/")
    if not base:
        return _err(503, "durable_runs_url_not_configured (HUB_CURIOSITY_DURABLE_RUNS_URL)", down=True)
    params = {"window": window, "sections": sections, "family": family, "outcome": outcome, "limit": limit}
    try:
        async with httpx.AsyncClient(base_url=base, timeout=UPSTREAM_TIMEOUT_SEC) as client:
            resp = await client.get(UPSTREAM_PATH, params={k: v for k, v in params.items() if v is not None})
    except (httpx.HTTPError, httpx.InvalidURL) as exc:
        return _err(502, f"durable_runs_unavailable: {type(exc).__name__} at {base}{UPSTREAM_PATH}", down=True)
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail")
        except (ValueError, AttributeError):
            detail = None
        detail = str(detail or resp.text or "")[:300]
        if resp.status_code in _PASSTHROUGH:
            # The service answered: it is reachable, even if the report is refused or missing.
            _mark_ok()
            return _err(resp.status_code, detail or f"durable_runs_{resp.status_code}", down=False)
        return _err(502, f"durable_runs_error {resp.status_code} on {UPSTREAM_PATH}: {detail}", down=True)
    try:
        body = resp.json()
    except ValueError:
        return _err(502, f"durable_runs_bad_json on {UPSTREAM_PATH}", down=True)
    if not isinstance(body, dict):
        return _err(502, f"durable_runs_bad_shape on {UPSTREAM_PATH}: expected an object", down=True)
    _mark_ok()
    return JSONResponse(content=body, headers=_NO_CACHE)
