"""Read-only Mind run introspection (same Postgres pool as Hub memory cards)."""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, Request

from .session import ensure_session

logger = logging.getLogger("orion-hub.mind")

router = APIRouter(tags=["mind"])


async def _need_session(x_orion_session_id: Optional[str]) -> str:
    from .main import bus

    return await ensure_session(x_orion_session_id, bus)


def _pool(request: Request):
    pool = getattr(request.app.state, "memory_pg_pool", None)
    if pool is None:
        raise HTTPException(status_code=503, detail="postgres_pool_unavailable")
    return pool


@router.get("/api/mind/runs/{mind_run_id}")
async def get_mind_run(mind_run_id: str, request: Request, x_orion_session_id: Optional[str] = None) -> dict[str, Any]:
    await _need_session(x_orion_session_id)
    pool = _pool(request)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT mind_run_id, correlation_id, session_id, trigger, ok, error_code,
                   snapshot_hash, router_profile_id, result_jsonb, request_summary_jsonb,
                   redaction_profile_id, created_at_utc
            FROM mind_runs WHERE mind_run_id = $1
            """,
            mind_run_id,
        )
    if row is None:
        raise HTTPException(status_code=404, detail="mind_run_not_found")
    return dict(row)


@router.get("/api/mind/runs")
async def list_mind_runs(
    request: Request,
    correlation_id: str = Query(..., min_length=4),
    limit: int = Query(50, ge=1, le=200),
    x_orion_session_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    await _need_session(x_orion_session_id)
    pool = _pool(request)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT mind_run_id, correlation_id, session_id, trigger, ok, error_code,
                   snapshot_hash, router_profile_id, result_jsonb, request_summary_jsonb,
                   redaction_profile_id, created_at_utc
            FROM mind_runs WHERE correlation_id = $1
            ORDER BY created_at_utc DESC
            LIMIT $2
            """,
            correlation_id,
            limit,
        )
    return [dict(r) for r in rows]
