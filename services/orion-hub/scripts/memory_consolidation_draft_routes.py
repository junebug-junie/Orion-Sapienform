from __future__ import annotations

import logging
from typing import Any, Dict, Literal, Optional

from fastapi import APIRouter, Header, HTTPException, Request

from orion.memory_graph.draft_repository import (
    get_consolidation_draft,
    list_consolidation_drafts,
    update_consolidation_draft_status,
)

from .session import ensure_session
from .memory_routes import _http_if_missing_memory_schema

logger = logging.getLogger("orion-hub.memory_consolidation_drafts")

router = APIRouter(tags=["memory-consolidation-drafts"])

DraftStatus = Literal["pending_review", "approved", "rejected"]


def _pool(request: Request):
    pool = getattr(request.app.state, "memory_pg_pool", None)
    if pool is None:
        raise HTTPException(status_code=503, detail="memory_store_unavailable")
    return pool


async def _need_session(x_orion_session_id: Optional[str]) -> str:
    from .main import bus

    return await ensure_session(x_orion_session_id, bus)


def _summarize_draft(draft_payload: dict[str, Any]) -> dict[str, int]:
    if not isinstance(draft_payload, dict):
        return {"entities": 0, "situations": 0, "edges": 0}
    return {
        "entities": len(draft_payload.get("entities") or []) if isinstance(draft_payload.get("entities"), list) else 0,
        "situations": len(draft_payload.get("situations") or []) if isinstance(draft_payload.get("situations"), list) else 0,
        "edges": len(draft_payload.get("edges") or []) if isinstance(draft_payload.get("edges"), list) else 0,
    }


def _public_item(row: dict[str, Any], *, include_draft: bool) -> dict[str, Any]:
    draft_payload = row.get("draft") if isinstance(row.get("draft"), dict) else {}
    out: dict[str, Any] = {
        "draft_id": row.get("draft_id"),
        "memory_window_id": row.get("memory_window_id"),
        "status": row.get("status"),
        "turn_correlation_ids": row.get("turn_correlation_ids") or [],
        "turn_count": len(row.get("turn_correlation_ids") or []),
        "created_at": row.get("created_at"),
        "summary": _summarize_draft(draft_payload),
    }
    if include_draft:
        out["draft"] = draft_payload
    return out


@router.get("/api/memory/consolidation/drafts")
async def list_memory_consolidation_drafts(
    request: Request,
    status: DraftStatus = "pending_review",
    limit: int = 50,
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    await _need_session(x_orion_session_id)
    pool = _pool(request)
    try:
        rows = await list_consolidation_drafts(pool, status=status, limit=limit)
    except Exception as exc:
        _http_if_missing_memory_schema(exc)
        logger.warning("consolidation_drafts_list_failed error=%s", exc)
        raise HTTPException(status_code=503, detail="memory_store_error") from exc
    return {
        "items": [_public_item(row, include_draft=False) for row in rows],
        "count": len(rows),
        "status": status,
    }


@router.get("/api/memory/consolidation/drafts/{draft_id}")
async def get_memory_consolidation_draft(
    request: Request,
    draft_id: str,
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    await _need_session(x_orion_session_id)
    pool = _pool(request)
    try:
        row = await get_consolidation_draft(pool, draft_id)
    except Exception as exc:
        _http_if_missing_memory_schema(exc)
        logger.warning("consolidation_draft_get_failed draft_id=%s error=%s", draft_id, exc)
        raise HTTPException(status_code=503, detail="memory_store_error") from exc
    if row is None:
        raise HTTPException(status_code=404, detail="consolidation_draft_not_found")
    return _public_item(row, include_draft=True)


@router.post("/api/memory/consolidation/drafts/{draft_id}/status")
async def set_memory_consolidation_draft_status(
    request: Request,
    draft_id: str,
    body: Dict[str, Any],
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    await _need_session(x_orion_session_id)
    status_raw = str((body or {}).get("status") or "").strip()
    if status_raw not in ("rejected", "pending_review"):
        raise HTTPException(status_code=400, detail="invalid_consolidation_draft_status")
    pool = _pool(request)
    try:
        row = await update_consolidation_draft_status(pool, draft_id, status=status_raw)  # type: ignore[arg-type]
    except Exception as exc:
        _http_if_missing_memory_schema(exc)
        logger.warning(
            "consolidation_draft_status_failed draft_id=%s status=%s error=%s",
            draft_id,
            status_raw,
            exc,
        )
        raise HTTPException(status_code=503, detail="memory_store_error") from exc
    if row is None:
        raise HTTPException(status_code=404, detail="consolidation_draft_not_found")
    return _public_item(row, include_draft=False)
