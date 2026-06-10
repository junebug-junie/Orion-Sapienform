from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Query, Request
from pydantic import ValidationError

from orion.core.storage import memory_cards as mc_dal
from orion.memory.crystallization.active_packet import build_active_packet
from orion.memory.crystallization.governor import GovernorError, approve, quarantine, reject, supersede
from orion.memory.crystallization.projection_cards import build_memory_card_projection
from orion.memory.crystallization.projection_chroma import build_chroma_upsert
from orion.memory.crystallization.projection_graphiti import GraphitiAdapter
from orion.memory.crystallization.proposer import propose
from orion.memory.crystallization.repository import (
    get_crystallization,
    insert_crystallization,
    insert_history,
    insert_retrieval_event,
    list_crystallizations,
    update_crystallization,
)
from orion.memory.crystallization.schemas import MemoryCrystallizationProposeRequestV1
from orion.memory.crystallization.validator import validate_proposal

from .session import ensure_session

try:
    from asyncpg.exceptions import UndefinedTableError as _AsyncpgUndefinedTableError
except ImportError:
    _AsyncpgUndefinedTableError = None  # type: ignore[misc, assignment]

logger = logging.getLogger("orion-hub.crystallization")

router = APIRouter(tags=["memory-crystallizations"])


def _pool(request: Request):
    pool = getattr(request.app.state, "memory_pg_pool", None)
    if pool is None:
        raise HTTPException(status_code=503, detail="memory_store_unavailable")
    return pool


def _http_if_missing_schema(exc: BaseException) -> None:
    if _AsyncpgUndefinedTableError is not None and isinstance(exc, _AsyncpgUndefinedTableError):
        raise HTTPException(status_code=503, detail="memory_crystallization_schema_missing") from exc


async def _need_session(x_orion_session_id: Optional[str]) -> str:
    from .main import bus

    return await ensure_session(x_orion_session_id, bus)


def _graphiti(request: Request) -> GraphitiAdapter:
    from scripts.settings import settings

    return GraphitiAdapter(
        enabled=bool(getattr(settings, "GRAPHITI_ENABLED", False)),
        url=getattr(settings, "GRAPHITI_URL", None),
        falkordb_uri=getattr(settings, "FALKORDB_URI", None),
    )


@router.post("/api/memory/crystallizations/propose")
async def crystallization_propose(
    request: Request,
    body: Dict[str, Any],
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    await _need_session(x_orion_session_id)
    pool = _pool(request)
    try:
        req = MemoryCrystallizationProposeRequestV1.model_validate(body)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.errors()) from e

    crystallization = propose(req)
    try:
        stored_id = await insert_crystallization(pool, crystallization)
        await insert_history(
            pool,
            crystallization_id=stored_id,
            op="propose",
            actor=req.proposed_by,
            before=None,
            after={"status": crystallization.status},
        )
    except Exception as exc:
        _http_if_missing_schema(exc)
        logger.warning("crystallization_propose_failed error=%s", exc)
        raise HTTPException(status_code=400, detail="propose_failed") from exc

    row = await get_crystallization(pool, stored_id)
    if not row:
        raise HTTPException(status_code=500, detail="propose_missing_row")
    return row.model_dump(mode="json")


@router.get("/api/memory/crystallizations/proposals")
async def crystallization_list_proposals(
    request: Request,
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
    limit: int = Query(default=200, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> Dict[str, Any]:
    await _need_session(x_orion_session_id)
    pool = _pool(request)
    try:
        items = await list_crystallizations(pool, status="proposed", limit=limit, offset=offset)
        quarantined = await list_crystallizations(pool, status="quarantined", limit=limit, offset=offset)
    except Exception as exc:
        _http_if_missing_schema(exc)
        raise HTTPException(status_code=503, detail="list_failed") from exc
    return {
        "items": [i.model_dump(mode="json") for i in items + quarantined],
        "count": len(items) + len(quarantined),
    }


@router.get("/api/memory/crystallizations/proposals/{crystallization_id}")
async def crystallization_get_proposal(
    request: Request,
    crystallization_id: str,
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    await _need_session(x_orion_session_id)
    pool = _pool(request)
    try:
        row = await get_crystallization(pool, crystallization_id)
    except Exception as exc:
        _http_if_missing_schema(exc)
        raise HTTPException(status_code=503, detail="get_failed") from exc
    if not row or row.status not in ("proposed", "quarantined"):
        raise HTTPException(status_code=404, detail="proposal_not_found")
    return row.model_dump(mode="json")


@router.post("/api/memory/crystallizations/proposals/{crystallization_id}/validate")
async def crystallization_validate_proposal(
    request: Request,
    crystallization_id: str,
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    await _need_session(x_orion_session_id)
    pool = _pool(request)
    row = await get_crystallization(pool, crystallization_id)
    if not row:
        raise HTTPException(status_code=404, detail="proposal_not_found")

    result = validate_proposal(row)
    updated = row.model_copy(deep=True)
    if result.valid:
        updated.governance.validation_status = "valid"
        updated.governance.validation_errors = []
    else:
        updated.governance.validation_status = "invalid"
        updated.governance.validation_errors = list(result.errors)

    await update_crystallization(pool, updated)
    return {"valid": result.valid, "errors": result.errors, "crystallization": updated.model_dump(mode="json")}


@router.post("/api/memory/crystallizations/proposals/{crystallization_id}/approve")
async def crystallization_approve_proposal(
    request: Request,
    crystallization_id: str,
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
    body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    session = await _need_session(x_orion_session_id)
    pool = _pool(request)
    row = await get_crystallization(pool, crystallization_id)
    if not row:
        raise HTTPException(status_code=404, detail="proposal_not_found")

    # Governor path requires explicit approval actor
    row.governance.approved_by = session
    reason = (body or {}).get("reason") if body else None
    try:
        updated, history = approve(row, actor=session, reason=reason)
    except GovernorError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    await update_crystallization(pool, updated)
    await insert_history(
        pool,
        crystallization_id=crystallization_id,
        op=history["op"],
        actor=session,
        before=history.get("before"),
        after=history.get("after"),
        reason=reason,
    )
    return updated.model_dump(mode="json")


@router.post("/api/memory/crystallizations/proposals/{crystallization_id}/reject")
async def crystallization_reject_proposal(
    request: Request,
    crystallization_id: str,
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
    body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    session = await _need_session(x_orion_session_id)
    pool = _pool(request)
    row = await get_crystallization(pool, crystallization_id)
    if not row:
        raise HTTPException(status_code=404, detail="proposal_not_found")

    reason = (body or {}).get("reason") if body else None
    try:
        updated, history = reject(row, actor=session, reason=reason)
    except GovernorError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    await update_crystallization(pool, updated)
    await insert_history(pool, crystallization_id=crystallization_id, op=history["op"], actor=session, before=history.get("before"), after=history.get("after"), reason=reason)
    return updated.model_dump(mode="json")


@router.post("/api/memory/crystallizations/proposals/{crystallization_id}/quarantine")
async def crystallization_quarantine_proposal(
    request: Request,
    crystallization_id: str,
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
    body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    session = await _need_session(x_orion_session_id)
    pool = _pool(request)
    row = await get_crystallization(pool, crystallization_id)
    if not row:
        raise HTTPException(status_code=404, detail="proposal_not_found")

    errors = (body or {}).get("errors") or ["operator_quarantine"]
    reason = (body or {}).get("reason") if body else None
    updated, history = quarantine(row, actor=session, errors=errors, reason=reason)
    await update_crystallization(pool, updated)
    await insert_history(pool, crystallization_id=crystallization_id, op=history["op"], actor=session, before=history.get("before"), after=history.get("after"), reason=reason)
    return updated.model_dump(mode="json")


@router.get("/api/memory/crystallizations")
async def crystallization_list(
    request: Request,
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
    status: Optional[str] = None,
    kind: Optional[str] = None,
    limit: int = Query(default=200, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> Dict[str, Any]:
    await _need_session(x_orion_session_id)
    pool = _pool(request)
    try:
        items = await list_crystallizations(pool, status=status, kind=kind, limit=limit, offset=offset)
    except Exception as exc:
        _http_if_missing_schema(exc)
        raise HTTPException(status_code=503, detail="list_failed") from exc
    return {"items": [i.model_dump(mode="json") for i in items], "count": len(items)}


@router.get("/api/memory/crystallizations/{crystallization_id}")
async def crystallization_get(
    request: Request,
    crystallization_id: str,
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    await _need_session(x_orion_session_id)
    pool = _pool(request)
    row = await get_crystallization(pool, crystallization_id)
    if not row:
        raise HTTPException(status_code=404, detail="not_found")
    return row.model_dump(mode="json")


@router.post("/api/memory/crystallizations/{crystallization_id}/project/card")
async def crystallization_project_card(
    request: Request,
    crystallization_id: str,
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    session = await _need_session(x_orion_session_id)
    pool = _pool(request)
    row = await get_crystallization(pool, crystallization_id)
    if not row:
        raise HTTPException(status_code=404, detail="not_found")

    card_create = build_memory_card_projection(row)
    if card_create is None:
        raise HTTPException(status_code=400, detail="projection_not_allowed_for_status")

    card_id = await mc_dal.insert_card(pool, card_create, actor=session, op="crystallization_project")
    updated = row.model_copy(deep=True)
    updated.projection_refs.memory_card_ids = list(updated.projection_refs.memory_card_ids) + [str(card_id)]
    await update_crystallization(pool, updated)
    return {"card_id": str(card_id), "crystallization_id": crystallization_id}


@router.post("/api/memory/crystallizations/{crystallization_id}/project/chroma")
async def crystallization_project_chroma(
    request: Request,
    crystallization_id: str,
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    await _need_session(x_orion_session_id)
    pool = _pool(request)
    row = await get_crystallization(pool, crystallization_id)
    if not row:
        raise HTTPException(status_code=404, detail="not_found")

    upsert = build_chroma_upsert(row)
    if upsert is None:
        raise HTTPException(status_code=400, detail="chroma_projection_not_allowed")

    return {
        "channel": "orion:memory:vector:upsert",
        "kind": "memory.vector.upsert.v1",
        "payload": upsert.model_dump(mode="json"),
    }


@router.post("/api/memory/crystallizations/{crystallization_id}/project/graphiti")
async def crystallization_project_graphiti(
    request: Request,
    crystallization_id: str,
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    await _need_session(x_orion_session_id)
    pool = _pool(request)
    row = await get_crystallization(pool, crystallization_id)
    if not row:
        raise HTTPException(status_code=404, detail="not_found")

    adapter = _graphiti(request)
    result = adapter.sync_crystallization(row)
    updated = adapter.apply_projection_refs(row, result)
    await update_crystallization(pool, updated)
    return {
        "canonical_mutated": result.canonical_mutated,
        "projection": result.__dict__,
        "crystallization_id": crystallization_id,
    }


@router.get("/api/memory/crystallizations/projection/health")
async def crystallization_projection_health(
    request: Request,
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    await _need_session(x_orion_session_id)
    adapter = _graphiti(request)
    return {
        "chroma_collection": "orion_memory_crystallizations",
        "graphiti_enabled": adapter.enabled,
        "rdf_memory_graph": "unchanged_existing_path",
    }


@router.post("/api/memory/active-packet")
async def memory_active_packet(
    request: Request,
    body: Dict[str, Any],
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    await _need_session(x_orion_session_id)
    pool = _pool(request)
    query = str(body.get("query") or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query_required")

    task_type = body.get("task_type")
    project_id = body.get("project_id")
    session_id = body.get("session_id")
    card_refs: List[str] = list(body.get("card_refs") or [])

    try:
        active_items = await list_crystallizations(pool, status="active", limit=100)
    except Exception as exc:
        _http_if_missing_schema(exc)
        raise HTTPException(status_code=503, detail="retrieval_failed") from exc

    packet = build_active_packet(
        query=query,
        crystallizations=active_items,
        card_refs=card_refs,
        task_type=task_type,
        project_id=project_id,
        session_id=session_id,
    )

    event_id = await insert_retrieval_event(
        pool,
        query=query,
        task_type=task_type,
        project_id=project_id,
        session_id=session_id,
        crystallization_ids=packet.crystallization_refs,
        card_refs=card_refs,
        trace=packet.retrieval_trace,
    )
    out = packet.model_dump(mode="json")
    out["retrieval_event_id"] = event_id
    return out


@router.get("/api/memory/retrieval-events/{event_id}")
async def memory_retrieval_event(
    request: Request,
    event_id: str,
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    await _need_session(x_orion_session_id)
    pool = _pool(request)
    try:
        UUID(event_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="invalid_event_id") from e

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM memory_crystallization_retrieval_events WHERE retrieval_event_id = $1::uuid",
            event_id,
        )
    if not row:
        raise HTTPException(status_code=404, detail="not_found")
    return dict(row)


@router.get("/api/memory/graphiti/health")
async def graphiti_health(
    request: Request,
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    await _need_session(x_orion_session_id)
    adapter = _graphiti(request)
    return {"enabled": adapter.enabled, "url_configured": bool(adapter.url)}


@router.get("/api/memory/graphiti/neighborhood/{crystallization_id}")
async def graphiti_neighborhood(
    request: Request,
    crystallization_id: str,
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    await _need_session(x_orion_session_id)
    return _graphiti(request).neighborhood(crystallization_id)
