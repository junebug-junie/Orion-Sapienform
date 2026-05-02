from __future__ import annotations

import logging
from typing import Optional

import asyncpg
from fastapi import APIRouter, Header, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from orion.core.contracts.memory_cards import MemoryCardV1
import orion.core.storage.memory_cards as mc

from scripts.session import ensure_session

logger = logging.getLogger("orion-hub.memory")

# Hub list paging: avoid unbounded scans from arbitrary query params.
_MEMORY_LIST_MAX_LIMIT = 500
_MEMORY_LIST_MAX_OFFSET = 500_000

router = APIRouter(prefix="/api/memory", tags=["memory"])


def _pool(request: Request) -> asyncpg.Pool:
    pool = getattr(request.app.state, "memory_pg_pool", None)
    if pool is None:
        raise HTTPException(status_code=503, detail="memory_cards_pool_unavailable")
    return pool


class MemoryCardCreateBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slug: str
    types: list[str]
    title: str
    summary: str
    provenance: str = "operator_highlight"
    status: str = "pending_review"
    confidence: str = "likely"
    sensitivity: str = "private"
    priority: str = "episodic_detail"
    visibility_scope: list[str] = Field(default_factory=lambda: ["chat"])
    anchor_class: Optional[str] = None
    tags: Optional[list[str]] = None


@router.post("/cards")
async def create_card(
    request: Request,
    body: MemoryCardCreateBody,
    x_orion_session_id: Optional[str] = Header(None),
):
    from scripts.main import bus

    if not bus:
        raise HTTPException(status_code=503, detail="bus_unavailable")
    await ensure_session(x_orion_session_id, bus)
    card = MemoryCardV1(
        slug=body.slug,
        types=body.types,
        title=body.title,
        summary=body.summary,
        provenance=body.provenance,
        status=body.status,
        confidence=body.confidence,
        sensitivity=body.sensitivity,
        priority=body.priority,
        visibility_scope=body.visibility_scope,
        anchor_class=body.anchor_class,
        tags=body.tags,
        evidence=[],
    )
    cid = await mc.insert_card(_pool(request), card, actor="hub_operator")
    return {"card_id": str(cid)}


@router.get("/cards")
async def list_cards_api(
    request: Request,
    x_orion_session_id: Optional[str] = Header(None),
    status: Optional[str] = None,
    types: Optional[str] = Query(None, description="Comma-separated"),
    anchor_class: Optional[str] = None,
    project: Optional[str] = None,
    priority: Optional[str] = None,
    limit: int = 200,
    offset: int = 0,
):
    from scripts.main import bus

    if not bus:
        raise HTTPException(status_code=503, detail="bus_unavailable")
    await ensure_session(x_orion_session_id, bus)
    type_list = [t.strip() for t in types.split(",")] if types else None
    eff_limit = min(max(limit, 1), _MEMORY_LIST_MAX_LIMIT)
    eff_offset = min(max(offset, 0), _MEMORY_LIST_MAX_OFFSET)
    rows = await mc.list_cards(
        _pool(request),
        status=status,
        types=type_list,
        anchor_class=anchor_class,
        project=project,
        priority=priority,
        limit=eff_limit,
        offset=eff_offset,
    )
    return {"items": [r.model_dump(mode="json") for r in rows]}


@router.get("/cards/{id_or_slug}")
async def get_card_api(
    request: Request,
    id_or_slug: str,
    x_orion_session_id: Optional[str] = Header(None),
):
    from scripts.main import bus

    if not bus:
        raise HTTPException(status_code=503, detail="bus_unavailable")
    await ensure_session(x_orion_session_id, bus)
    card = await mc.get_card(_pool(request), id_or_slug)
    if not card or not card.card_id:
        raise HTTPException(status_code=404, detail="not_found")
    edges = await mc.list_edges(_pool(request), card_id=str(card.card_id), direction="both")
    hist = await mc.list_history(_pool(request), card_id=str(card.card_id), limit=10)
    return {
        "card": card.model_dump(mode="json"),
        "edges": [e.model_dump(mode="json") for e in edges],
        "history": [h.model_dump(mode="json") for h in hist],
    }


@router.post("/sessions/{session_id}/distill")
async def distill_stub(session_id: str, x_orion_session_id: Optional[str] = Header(None)):
    from scripts.main import bus

    if not bus:
        raise HTTPException(status_code=503, detail="bus_unavailable")
    await ensure_session(x_orion_session_id, bus)
    raise HTTPException(status_code=501, detail="distill_cli_phase5")


@router.post("/history/{history_id}/reverse")
async def reverse_history_api(
    request: Request,
    history_id: str,
    x_orion_session_id: Optional[str] = Header(None),
):
    from scripts.main import bus

    if not bus:
        raise HTTPException(status_code=503, detail="bus_unavailable")
    await ensure_session(x_orion_session_id, bus)
    try:
        entry = await mc.reverse_history(_pool(request), history_id, actor="hub_operator")
    except LookupError:
        raise HTTPException(status_code=404, detail="history_not_found") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return entry.model_dump(mode="json")
