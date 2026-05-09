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


@router.get("/api/mind/runs/recent")
async def list_recent_mind_runs(
    request: Request,
    hours: int = Query(24, ge=1, le=168),
    limit: int = Query(100, ge=1, le=500),
    ok: Optional[bool] = None,
    trigger: Optional[str] = Query(None, min_length=1, max_length=64),
    error_code: Optional[str] = Query(None, min_length=1, max_length=128),
    router_profile_id: Optional[str] = Query(None, min_length=1, max_length=128),
    x_orion_session_id: Optional[str] = None,
) -> dict[str, Any]:
    session_id = await _need_session(x_orion_session_id)
    pool = _pool(request)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT mind_run_id, correlation_id, created_at_utc, ok, trigger, error_code, router_profile_id
            FROM mind_runs
            WHERE session_id = $1
              AND created_at_utc >= ((NOW() AT TIME ZONE 'UTC') - ($2 * INTERVAL '1 hour'))
              AND ($3::boolean IS NULL OR ok = $3)
              AND ($4::text IS NULL OR trigger = $4)
              AND ($5::text IS NULL OR error_code = $5)
              AND ($6::text IS NULL OR router_profile_id = $6)
            ORDER BY created_at_utc DESC, mind_run_id DESC
            LIMIT $7
            """,
            session_id,
            hours,
            ok,
            trigger,
            error_code,
            router_profile_id,
            limit,
        )
        summary = await conn.fetchrow(
            """
            SELECT
              COUNT(*)::int AS total_runs,
              COALESCE(SUM(CASE WHEN ok THEN 1 ELSE 0 END), 0)::int AS ok_count,
              COALESCE(SUM(CASE WHEN ok THEN 0 ELSE 1 END), 0)::int AS failed_count
            FROM mind_runs
            WHERE session_id = $1
              AND created_at_utc >= ((NOW() AT TIME ZONE 'UTC') - ($2 * INTERVAL '1 hour'))
              AND ($3::boolean IS NULL OR ok = $3)
              AND ($4::text IS NULL OR trigger = $4)
              AND ($5::text IS NULL OR error_code = $5)
              AND ($6::text IS NULL OR router_profile_id = $6)
            """,
            session_id,
            hours,
            ok,
            trigger,
            error_code,
            router_profile_id,
        )
        top_error_codes = await conn.fetch(
            """
            SELECT error_code, COUNT(*)::int AS run_count
            FROM mind_runs
            WHERE session_id = $1
              AND created_at_utc >= ((NOW() AT TIME ZONE 'UTC') - ($2 * INTERVAL '1 hour'))
              AND ($3::boolean IS NULL OR ok = $3)
              AND ($4::text IS NULL OR trigger = $4)
              AND ($5::text IS NULL OR error_code = $5)
              AND ($6::text IS NULL OR router_profile_id = $6)
              AND error_code IS NOT NULL
            GROUP BY error_code
            ORDER BY run_count DESC, error_code ASC
            LIMIT 3
            """,
            session_id,
            hours,
            ok,
            trigger,
            error_code,
            router_profile_id,
        )
        top_router_profiles = await conn.fetch(
            """
            SELECT router_profile_id, COUNT(*)::int AS run_count
            FROM mind_runs
            WHERE session_id = $1
              AND created_at_utc >= ((NOW() AT TIME ZONE 'UTC') - ($2 * INTERVAL '1 hour'))
              AND ($3::boolean IS NULL OR ok = $3)
              AND ($4::text IS NULL OR trigger = $4)
              AND ($5::text IS NULL OR error_code = $5)
              AND ($6::text IS NULL OR router_profile_id = $6)
              AND router_profile_id IS NOT NULL
            GROUP BY router_profile_id
            ORDER BY run_count DESC, router_profile_id ASC
            LIMIT 3
            """,
            session_id,
            hours,
            ok,
            trigger,
            error_code,
            router_profile_id,
        )
        bucket_counts = await conn.fetch(
            """
            SELECT date_trunc('hour', created_at_utc) AS bucket_utc, COUNT(*)::int AS run_count
            FROM mind_runs
            WHERE session_id = $1
              AND created_at_utc >= ((NOW() AT TIME ZONE 'UTC') - ($2 * INTERVAL '1 hour'))
              AND ($3::boolean IS NULL OR ok = $3)
              AND ($4::text IS NULL OR trigger = $4)
              AND ($5::text IS NULL OR error_code = $5)
              AND ($6::text IS NULL OR router_profile_id = $6)
            GROUP BY 1
            ORDER BY bucket_utc DESC
            LIMIT 72
            """,
            session_id,
            hours,
            ok,
            trigger,
            error_code,
            router_profile_id,
        )

    return {
        "items": [dict(r) for r in rows],
        "next_cursor": None,
        "aggregates": {
            **dict(summary or {"total_runs": 0, "ok_count": 0, "failed_count": 0}),
            "top_error_codes": [dict(r) for r in top_error_codes],
            "top_router_profile_ids": [dict(r) for r in top_router_profiles],
            "time_buckets": [dict(r) for r in bucket_counts],
        },
    }


@router.get("/api/mind/runs/{mind_run_id}")
async def get_mind_run(mind_run_id: str, request: Request, x_orion_session_id: Optional[str] = None) -> dict[str, Any]:
    session_id = await _need_session(x_orion_session_id)
    pool = _pool(request)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT mind_run_id, correlation_id, session_id, trigger, ok, error_code,
                   snapshot_hash, router_profile_id, result_jsonb, request_summary_jsonb,
                   redaction_profile_id, created_at_utc
            FROM mind_runs WHERE mind_run_id = $1 AND session_id = $2
            """,
            mind_run_id,
            session_id,
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
    session_id = await _need_session(x_orion_session_id)
    pool = _pool(request)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT mind_run_id, correlation_id, session_id, trigger, ok, error_code,
                   snapshot_hash, router_profile_id, result_jsonb, request_summary_jsonb,
                   redaction_profile_id, created_at_utc
            FROM mind_runs WHERE correlation_id = $1 AND session_id = $2
            ORDER BY created_at_utc DESC
            LIMIT $3
            """,
            correlation_id,
            session_id,
            limit,
        )
    return [dict(r) for r in rows]
