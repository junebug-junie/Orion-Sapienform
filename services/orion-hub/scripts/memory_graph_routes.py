from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import asyncpg
import requests
from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import ValidationError

from orion.memory_graph.approve import approve_memory_graph_draft, preview_validate_only
from orion.memory_graph.dto import SuggestDraftV1

from .session import ensure_session

logger = logging.getLogger("orion-hub.memory_graph")

router = APIRouter(tags=["memory-graph"])


def _pool(request: Request):
    pool = getattr(request.app.state, "memory_pg_pool", None)
    if pool is None:
        raise HTTPException(status_code=503, detail="memory_store_unavailable")
    return pool


@router.post("/api/memory/graph/validate")
async def memory_graph_validate(
    request: Request,
    body: Dict[str, Any],
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    from scripts.main import bus

    await ensure_session(x_orion_session_id, bus)
    try:
        draft = SuggestDraftV1.model_validate(body)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.errors()) from e
    ok, violations, preview = preview_validate_only(draft)
    return {"ok": ok, "violations": violations, "preview": preview}


@router.post("/api/memory/graph/approve")
async def memory_graph_approve(
    request: Request,
    body: Dict[str, Any],
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    from scripts.main import bus
    from scripts.settings import settings

    await ensure_session(x_orion_session_id, bus)
    if not getattr(settings, "GRAPHDB_URL", "").strip():
        raise HTTPException(status_code=503, detail="graphdb_unconfigured")
    try:
        draft = SuggestDraftV1.model_validate(body.get("draft") or body)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.errors()) from e
    named = (
        str(body.get("named_graph_iri") or "").strip()
        or getattr(settings, "MEMORY_GRAPH_DEFAULT_NAMED_GRAPH", "").strip()
    )
    if not named:
        raise HTTPException(status_code=400, detail="named_graph_iri_required")

    pool = _pool(request)
    try:
        result = await approve_memory_graph_draft(
            draft,
            pool,
            graphdb_url=str(settings.GRAPHDB_URL),
            graphdb_repo=str(settings.GRAPHDB_REPO or "collapse"),
            graphdb_user=str(settings.GRAPHDB_USER or ""),
            graphdb_pass=str(settings.GRAPHDB_PASS or ""),
            named_graph_iri=named,
        )
    except ValueError as exc:
        if str(exc) == "hierarchy_cycle":
            raise HTTPException(status_code=400, detail="hierarchy_cycle") from exc
        logger.warning("memory_graph_approve_failed error=%s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except asyncpg.PostgresError as exc:
        logger.warning("memory_graph_approve_failed postgres error=%s", exc)
        raise HTTPException(status_code=503, detail="memory_store_error") from exc
    except requests.RequestException as exc:
        logger.warning("memory_graph_approve_failed graphdb error=%s", exc)
        raise HTTPException(status_code=503, detail="graphdb_unavailable") from exc

    if not result.ok:
        return {"ok": False, "violations": result.violations, "card_ids": []}
    return {
        "ok": True,
        "violations": [],
        "card_ids": [str(x) for x in result.card_ids],
    }
