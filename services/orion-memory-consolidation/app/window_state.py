from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import asyncpg

from orion.schemas.memory_consolidation import MemoryTurnPersistedV1


class WindowStore:
    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    async def _get_open_window(self) -> asyncpg.Record | None:
        return await self._pool.fetchrow(
            """
            SELECT * FROM memory_consolidation_windows
            WHERE status = 'open'
            ORDER BY created_at ASC
            LIMIT 1
            """
        )

    async def append_turn(self, turn: MemoryTurnPersistedV1, *, scores: dict[str, Any]) -> None:
        row = await self._get_open_window()
        turn_entry = {
            "correlation_id": turn.correlation_id,
            "prompt": turn.prompt,
            "response": turn.response,
            "memory_significance_score": scores.get("memory_significance_score"),
            "conversation_boundary_score": scores.get("conversation_boundary_score"),
        }
        if row is None:
            window_id = str(uuid4())
            await self._pool.execute(
                """
                INSERT INTO memory_consolidation_windows
                  (memory_window_id, status, turn_correlation_ids, created_at)
                VALUES ($1, 'open', $2::jsonb, $3)
                """,
                window_id,
                json.dumps([turn_entry]),
                datetime.now(timezone.utc),
            )
            return

        turns = json.loads(row["turn_correlation_ids"]) if row["turn_correlation_ids"] else []
        if isinstance(turns, list):
            turns.append(turn_entry)
        else:
            turns = [turn_entry]
        corr_ids = [t.get("correlation_id") for t in turns if isinstance(t, dict)]
        await self._pool.execute(
            """
            UPDATE memory_consolidation_windows
            SET turn_correlation_ids = $2::jsonb
            WHERE memory_window_id = $1
            """,
            row["memory_window_id"],
            json.dumps(turns),
        )

    async def close_current_window(self, closing_correlation_id: str) -> dict[str, Any]:
        row = await self._get_open_window()
        if row is None:
            return {"memory_window_id": str(uuid4()), "turn_correlation_ids": [], "turns": []}

        turns = json.loads(row["turn_correlation_ids"]) if row["turn_correlation_ids"] else []
        if not isinstance(turns, list):
            turns = []
        completed = [t for t in turns if isinstance(t, dict) and t.get("correlation_id") != closing_correlation_id]
        phase = None
        for t in reversed(turns):
            if isinstance(t, dict) and t.get("correlation_id") == closing_correlation_id:
                phase = t.get("phase_change")
                break
        now = datetime.now(timezone.utc)
        await self._pool.execute(
            """
            UPDATE memory_consolidation_windows
            SET status = 'closed', closed_at = $2, phase_change_at_close = $3, consolidation_status = 'pending'
            WHERE memory_window_id = $1
            """,
            row["memory_window_id"],
            now,
            phase,
        )
        new_window_id = str(uuid4())
        closing_turn = next(
            (t for t in turns if isinstance(t, dict) and t.get("correlation_id") == closing_correlation_id),
            None,
        )
        next_turns = [closing_turn] if closing_turn else []
        await self._pool.execute(
            """
            INSERT INTO memory_consolidation_windows
              (memory_window_id, status, turn_correlation_ids, created_at)
            VALUES ($1, 'open', $2::jsonb, $3)
            """,
            new_window_id,
            json.dumps(next_turns),
            now,
        )
        return {
            "memory_window_id": row["memory_window_id"],
            "turn_correlation_ids": [t.get("correlation_id") for t in completed if isinstance(t, dict)],
            "turns": completed,
        }

    async def mark_consolidated(self, memory_window_id: str, *, draft_id: str) -> None:
        await self._pool.execute(
            """
            UPDATE memory_consolidation_windows
            SET status = 'consolidated', consolidation_status = 'ok', draft_id = $2
            WHERE memory_window_id = $1
            """,
            memory_window_id,
            draft_id,
        )

    async def mark_failed(self, memory_window_id: str) -> None:
        await self._pool.execute(
            """
            UPDATE memory_consolidation_windows
            SET consolidation_status = 'failed'
            WHERE memory_window_id = $1
            """,
            memory_window_id,
        )

    async def list_failed_windows(self, *, limit: int = 20) -> list[asyncpg.Record]:
        return await self._pool.fetch(
            """
            SELECT memory_window_id, turn_correlation_ids
            FROM memory_consolidation_windows
            WHERE consolidation_status = 'failed' AND status = 'closed'
            ORDER BY closed_at ASC
            LIMIT $1
            """,
            limit,
        )

    async def get_window_turns(self, memory_window_id: str) -> list[dict[str, Any]]:
        row = await self._pool.fetchrow(
            "SELECT turn_correlation_ids FROM memory_consolidation_windows WHERE memory_window_id = $1",
            memory_window_id,
        )
        if row is None:
            return []
        turns = json.loads(row["turn_correlation_ids"]) if row["turn_correlation_ids"] else []
        return turns if isinstance(turns, list) else []
