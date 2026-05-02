from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException, Query, Request
from pydantic import ValidationError

from orion.core.contracts.memory_cards import (
    MemoryCardCreateV1,
    MemoryCardEdgeCreateV1,
    MemoryCardPatchV1,
    MemoryCardStatusChangeV1,
)
from orion.core.storage import memory_cards as mc_dal

from .session import ensure_session

logger = logging.getLogger("orion-hub.memory")

router = APIRouter(tags=["memory-cards"])


def _pool(request: Request):
    pool = getattr(request.app.state, "memory_pg_pool", None)
    if pool is None:
        raise HTTPException(status_code=503, detail="memory_store_unavailable")
    return pool


def _clamp_limit(limit: int, *, default: int = 200, cap: int = 500) -> int:
    if limit <= 0:
        return default
    return min(limit, cap)


def _clamp_offset(offset: int) -> int:
    if offset < 0:
        return 0
    return min(offset, 100_000)


async def _need_session(x_orion_session_id: Optional[str]) -> str:
    from .main import bus

    return await ensure_session(x_orion_session_id, bus)


def _parse_csv(value: Optional[str]) -> Optional[List[str]]:
    if value is None or not str(value).strip():
        return None
    return [p.strip() for p in str(value).split(",") if p.strip()]


@router.post("/api/memory/cards")
async def memory_create_card(
    request: Request,
    body: Dict[str, Any],
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    await _need_session(x_orion_session_id)
    pool = _pool(request)
    try:
        card = MemoryCardCreateV1.model_validate(body)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.errors()) from e
    try:
        cid = await mc_dal.insert_card(pool, card, actor="operator", op="create")
    except Exception as exc:
        logger.warning("memory_create_card_failed error=%s", exc)
        raise HTTPException(status_code=400, detail="create_failed") from exc
    row = await mc_dal.get_card(pool, str(cid))
    if not row:
        raise HTTPException(status_code=500, detail="create_missing_row")
    return row.model_dump(mode="json")


@router.get("/api/memory/cards")
async def memory_list_cards(
    request: Request,
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
    status: Optional[str] = None,
    types: Optional[str] = None,
    anchor_class: Optional[str] = None,
    project: Optional[str] = None,
    priority: Optional[str] = None,
    limit: int = Query(default=200, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> Dict[str, Any]:
    await _need_session(x_orion_session_id)
    pool = _pool(request)
    type_list = _parse_csv(types)
    rows = await mc_dal.list_cards(
        pool,
        status=status,
        types=type_list,
        anchor_class=anchor_class,
        project=project,
        priority=priority,
        limit=_clamp_limit(limit),
        offset=_clamp_offset(offset),
    )
    return {"items": [r.model_dump(mode="json") for r in rows]}


@router.get("/api/memory/cards/{id_or_slug}")
async def memory_get_card(
    request: Request,
    id_or_slug: str,
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    await _need_session(x_orion_session_id)
    pool = _pool(request)
    card = await mc_dal.get_card(pool, id_or_slug)
    if not card:
        raise HTTPException(status_code=404, detail="card_not_found")
    edges_out = await mc_dal.list_edges(pool, card_id=str(card.card_id), direction="out")
    edges_in = await mc_dal.list_edges(pool, card_id=str(card.card_id), direction="in")
    hist = await mc_dal.list_history(pool, card_id=str(card.card_id), limit=50, offset=0)
    return {
        "card": card.model_dump(mode="json"),
        "edges_out": [e.model_dump(mode="json") for e in edges_out],
        "edges_in": [e.model_dump(mode="json") for e in edges_in],
        "history": [h.model_dump(mode="json") for h in hist],
    }


@router.patch("/api/memory/cards/{id_or_slug}")
async def memory_patch_card(
    request: Request,
    id_or_slug: str,
    body: Dict[str, Any],
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    await _need_session(x_orion_session_id)
    pool = _pool(request)
    try:
        patch = MemoryCardPatchV1.model_validate(body)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.errors()) from e
    updated = await mc_dal.update_card(pool, id_or_slug, patch, actor="operator")
    if not updated:
        raise HTTPException(status_code=404, detail="card_not_found")
    return updated.model_dump(mode="json")


@router.post("/api/memory/cards/{card_id}/status")
async def memory_change_status(
    request: Request,
    card_id: str,
    body: Dict[str, Any],
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    await _need_session(x_orion_session_id)
    pool = _pool(request)
    try:
        ch = MemoryCardStatusChangeV1.model_validate(body)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.errors()) from e
    updated = await mc_dal.change_card_status(
        pool, card_id, new_status=ch.status, actor="operator", reason=ch.reason
    )
    if not updated:
        raise HTTPException(status_code=404, detail="card_not_found")
    return updated.model_dump(mode="json")


@router.post("/api/memory/edges")
async def memory_add_edge(
    request: Request,
    body: Dict[str, Any],
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    await _need_session(x_orion_session_id)
    pool = _pool(request)
    try:
        edge = MemoryCardEdgeCreateV1.model_validate(body)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.errors()) from e
    try:
        created = await mc_dal.add_edge(pool, edge, actor="operator")
    except ValueError as exc:
        if str(exc) == "hierarchy_cycle":
            raise HTTPException(status_code=400, detail="hierarchy_cycle") from exc
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return created.model_dump(mode="json")


@router.delete("/api/memory/edges/{edge_id}")
async def memory_delete_edge(
    request: Request,
    edge_id: str,
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    await _need_session(x_orion_session_id)
    pool = _pool(request)
    await mc_dal.remove_edge(pool, edge_id, actor="operator")
    return {"ok": True}


@router.get("/api/memory/history")
async def memory_list_history(
    request: Request,
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
    card_id: Optional[str] = None,
    edge_id: Optional[str] = None,
    limit: int = Query(default=200, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> Dict[str, Any]:
    if not card_id and not edge_id:
        raise HTTPException(status_code=400, detail="card_id_or_edge_id_required")
    await _need_session(x_orion_session_id)
    pool = _pool(request)
    rows = await mc_dal.list_history(
        pool,
        card_id=card_id,
        edge_id=edge_id,
        limit=_clamp_limit(limit),
        offset=_clamp_offset(offset),
    )
    return {"items": [h.model_dump(mode="json") for h in rows]}


@router.post("/api/memory/history/{history_id}/reverse")
async def memory_reverse_history(
    request: Request,
    history_id: str,
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    await _need_session(x_orion_session_id)
    pool = _pool(request)
    try:
        entry = await mc_dal.reverse_history(pool, history_id, actor="operator")
    except LookupError:
        raise HTTPException(status_code=404, detail="history_not_found") from None
    except ValueError as exc:
        msg = str(exc)
        if msg == "reverse_unsupported_op":
            raise HTTPException(status_code=400, detail="reverse_unsupported") from exc
        raise HTTPException(status_code=400, detail=msg) from exc
    return entry.model_dump(mode="json")


@router.get("/api/memory/cards/{id_or_slug}/neighborhood")
async def memory_neighborhood(
    request: Request,
    id_or_slug: str,
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
    hops: int = Query(default=1, ge=1, le=3),
) -> Dict[str, Any]:
    await _need_session(x_orion_session_id)
    pool = _pool(request)
    return await mc_dal.neighborhood(pool, id_or_slug, hops=hops)


@router.post("/api/memory/sessions/{session_id}/distill")
async def memory_distill_stub(
    session_id: str,
    x_orion_session_id: Optional[str] = Header(None, alias="X-Orion-Session-Id"),
) -> Dict[str, Any]:
    await _need_session(x_orion_session_id)
    raise HTTPException(status_code=501, detail="distill_not_implemented")
