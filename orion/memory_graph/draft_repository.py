from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

import asyncpg

from orion.memory_graph.dto import SuggestDraftV1

DraftStatus = Literal["pending_review", "approved", "rejected"]


async def insert_pending_draft(
    pool: asyncpg.Pool,
    *,
    memory_window_id: str,
    draft: SuggestDraftV1,
    turn_correlation_ids: list[str],
) -> str:
    draft_id = str(uuid4())
    row = await pool.fetchrow(
        """
        INSERT INTO memory_graph_suggest_drafts
          (draft_id, memory_window_id, status, draft, turn_correlation_ids, created_at)
        VALUES ($1, $2, 'pending_review', $3::jsonb, $4::jsonb, $5)
        ON CONFLICT (memory_window_id) DO UPDATE SET
          draft = EXCLUDED.draft,
          turn_correlation_ids = EXCLUDED.turn_correlation_ids,
          status = 'pending_review'
        RETURNING draft_id
        """,
        draft_id,
        memory_window_id,
        json.dumps(draft.model_dump(mode="json")),
        json.dumps(turn_correlation_ids),
        datetime.now(timezone.utc),
    )
    return str(row["draft_id"])


def _row_to_dict(row: asyncpg.Record) -> dict[str, Any]:
    draft_payload = row["draft"]
    if isinstance(draft_payload, str):
        draft_payload = json.loads(draft_payload)
    turn_ids = row["turn_correlation_ids"]
    if isinstance(turn_ids, str):
        turn_ids = json.loads(turn_ids)
    created_at = row["created_at"]
    return {
        "draft_id": str(row["draft_id"]),
        "memory_window_id": str(row["memory_window_id"]),
        "status": str(row["status"]),
        "draft": draft_payload if isinstance(draft_payload, dict) else {},
        "turn_correlation_ids": turn_ids if isinstance(turn_ids, list) else [],
        "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at),
    }


async def list_consolidation_drafts(
    pool: asyncpg.Pool,
    *,
    status: DraftStatus = "pending_review",
    limit: int = 50,
) -> list[dict[str, Any]]:
    lim = max(1, min(int(limit), 200))
    rows = await pool.fetch(
        """
        SELECT draft_id, memory_window_id, status, draft, turn_correlation_ids, created_at
        FROM memory_graph_suggest_drafts
        WHERE status = $1
        ORDER BY created_at DESC
        LIMIT $2
        """,
        status,
        lim,
    )
    return [_row_to_dict(row) for row in rows]


async def get_consolidation_draft(pool: asyncpg.Pool, draft_id: str) -> dict[str, Any] | None:
    row = await pool.fetchrow(
        """
        SELECT draft_id, memory_window_id, status, draft, turn_correlation_ids, created_at
        FROM memory_graph_suggest_drafts
        WHERE draft_id = $1
        """,
        str(draft_id),
    )
    if row is None:
        return None
    return _row_to_dict(row)


async def update_consolidation_draft_status(
    pool: asyncpg.Pool,
    draft_id: str,
    *,
    status: DraftStatus,
) -> dict[str, Any] | None:
    if status == "approved":
        row = await pool.fetchrow(
            """
            UPDATE memory_graph_suggest_drafts
            SET status = $2
            WHERE draft_id = $1 AND status = 'pending_review'
            RETURNING draft_id, memory_window_id, status, draft, turn_correlation_ids, created_at
            """,
            str(draft_id),
            status,
        )
    else:
        row = await pool.fetchrow(
            """
            UPDATE memory_graph_suggest_drafts
            SET status = $2
            WHERE draft_id = $1
            RETURNING draft_id, memory_window_id, status, draft, turn_correlation_ids, created_at
            """,
            str(draft_id),
            status,
        )
    if row is None:
        return None
    return _row_to_dict(row)
