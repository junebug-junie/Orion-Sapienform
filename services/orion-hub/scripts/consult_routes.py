"""Consult API: Juniper's answer to a brief, and the conversation behind it.

``GET  /api/consults``                  briefs waiting for her answer (``?all=1`` for every consult)
``GET  /api/consults/{id}``             one consult with every turn (the "Show the conversation" view)
``POST /api/consults/{id}/decision``    {"decision": "yes"|"no"|"no_because", "reason": "..."}
``GET  /api/debug/consult``             the loop's last tick and the meter it reads

Design: docs/superpowers/specs/2026-11-11-consult-claude-before-juniper-design.md.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from orion.schemas.consult import ConsultDecision

from scripts import consult_loop as store

router = APIRouter()


class DecisionBody(BaseModel):
    decision: ConsultDecision
    reason: Optional[str] = None


def _pool(request: Request) -> Any:
    pool = getattr(request.app.state, "memory_pg_pool", None)
    if pool is None:
        raise HTTPException(status_code=503, detail="postgres_pool_unavailable")
    return pool


def _public(row: dict) -> dict:
    out = {k: v for k, v in row.items() if k not in ("brief_json",)}
    for key, value in list(out.items()):
        if hasattr(value, "isoformat"):
            out[key] = value.isoformat()
    return out


@router.get("/api/consults")
async def list_consults(request: Request, all: int = 0, limit: int = 20) -> dict:
    rows = await store.recent(_pool(request), undecided_only=not all, limit=max(1, min(limit, 100)))
    return {"consults": [_public(r) for r in rows]}


@router.get("/api/consults/{consult_id}")
async def get_consult(request: Request, consult_id: str) -> dict:
    pool = _pool(request)
    row = await store.get(pool, consult_id)
    if row is None:
        raise HTTPException(status_code=404, detail="consult_not_found")
    turns = [_public(t) for t in await store.turns(pool, consult_id)]
    return {"consult": _public(row), "turns": turns}


@router.post("/api/consults/{consult_id}/decision")
async def decide_consult(request: Request, consult_id: str, body: DecisionBody) -> dict:
    reason = (body.reason or "").strip() or None
    if body.decision == "no_because" and not reason:
        raise HTTPException(status_code=422, detail="no_because_needs_a_reason")
    row = await store.decide(_pool(request), consult_id, decision=body.decision, reason=reason)
    if row is None:
        raise HTTPException(status_code=409, detail="no_undecided_brief")
    return {"consult": _public(row)}


@router.get("/api/debug/consult")
async def consult_debug() -> dict:
    from scripts import main as hub_main

    loop = getattr(hub_main, "consult_loop", None)
    return loop.status() if loop is not None else {"enabled": False, "reason": "not_started"}
