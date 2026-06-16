from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

import asyncpg

from orion.memory_graph.dto import SuggestDraftV1


async def insert_pending_draft(
    pool: asyncpg.Pool,
    *,
    memory_window_id: str,
    draft: SuggestDraftV1,
    turn_correlation_ids: list[str],
) -> str:
    draft_id = str(uuid4())
    await pool.execute(
        """
        INSERT INTO memory_graph_suggest_drafts
          (draft_id, memory_window_id, status, draft, turn_correlation_ids, created_at)
        VALUES ($1, $2, 'pending_review', $3::jsonb, $4::jsonb, $5)
        ON CONFLICT (memory_window_id) DO NOTHING
        """,
        draft_id,
        memory_window_id,
        json.dumps(draft.model_dump(mode="json")),
        json.dumps(turn_correlation_ids),
        datetime.now(timezone.utc),
    )
    return draft_id
