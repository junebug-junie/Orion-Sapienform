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

    async def _get_open_window(self, source_platform: str | None = None) -> asyncpg.Record | None:
        """The open window for one platform. NULL platform = direct conversation.

        This used to select the single global open window with no partitioning
        at all, so ai-town NPC turns and Juniper's own turns appended to the same
        window -- 26 windows were confirmed mixed on live data (2026-08-14). An
        episode is meant to be one coherent stretch of conversation, so that was
        a correctness bug independent of the review queue.

        `IS NOT DISTINCT FROM` rather than `=` because the direct-conversation
        partition is keyed by NULL, and `NULL = NULL` is NULL, which would make
        every direct turn open a brand-new window and never find its own.
        """
        return await self._pool.fetchrow(
            """
            SELECT * FROM memory_consolidation_windows
            WHERE status = 'open'
              AND source_platform IS NOT DISTINCT FROM $1
            ORDER BY created_at ASC
            LIMIT 1
            """,
            source_platform,
        )

    async def append_turn(self, turn: MemoryTurnPersistedV1, *, scores: dict[str, Any]) -> None:
        row = await self._get_open_window(turn.source_platform)
        phase_change = (turn.spark_meta.get("conversation_phase") or {}).get("phase_change")
        appraisal = scores.get("turn_change_appraisal")
        spark_meta: dict[str, Any] = {}
        if isinstance(appraisal, dict):
            spark_meta["turn_change_appraisal"] = appraisal
        sig = scores.get("memory_significance_score")
        if isinstance(sig, (int, float)):
            spark_meta["memory_significance_score"] = float(sig)
        turn_entry = {
            "correlation_id": turn.correlation_id,
            "prompt": turn.prompt,
            "response": turn.response,
            "memory_significance_score": scores.get("memory_significance_score"),
            "conversation_boundary_score": scores.get("conversation_boundary_score"),
            "phase_change": phase_change,
            "memory_classify_ts": scores.get("memory_classify_ts"),
            "spark_meta": spark_meta,
            # Still carried per-turn even though the cursor is now partitioned by
            # platform, for two reasons: windows created before that partitioning
            # can genuinely hold turns from more than one platform, and
            # _window_source_platform()'s unanimity rule is what keeps those
            # legacy mixed windows out of the auto-activate path. Belt and braces
            # -- the partitioning prevents new mixing, the unanimity check refuses
            # to trust any window that mixed anyway.
            "source_platform": turn.source_platform,
        }
        if row is None:
            window_id = str(uuid4())
            await self._pool.execute(
                """
                INSERT INTO memory_consolidation_windows
                  (memory_window_id, status, turn_correlation_ids, created_at, source_platform)
                VALUES ($1, 'open', $2::jsonb, $3, $4)
                """,
                window_id,
                json.dumps([turn_entry]),
                datetime.now(timezone.utc),
                turn.source_platform,
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

    async def close_current_window(
        self, closing_correlation_id: str, *, source_platform: str | None
    ) -> dict[str, Any]:
        """`source_platform` is REQUIRED (keyword, no default) even though None is
        a legal value.

        None is a real partition -- Juniper's direct conversation -- not a
        sentinel meaning "any". A default would make a forgotten kwarg silently
        close and reseed HER window instead of the intended one, with nothing in
        the signature, the types, or the runtime to make the omission visible.
        Passing None must be a decision, not an oversight.
        """
        row = await self._get_open_window(source_platform)
        if row is None:
            return {"memory_window_id": str(uuid4()), "turn_correlation_ids": [], "turns": []}

        turns = json.loads(row["turn_correlation_ids"]) if row["turn_correlation_ids"] else []
        if not isinstance(turns, list):
            turns = []
        all_turns = [t for t in turns if isinstance(t, dict)]
        phase = None
        for t in reversed(all_turns):
            if t.get("correlation_id") == closing_correlation_id:
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
        # The closing turn is carried into the next window as its first turn (it
        # is the boundary, and it belongs to both sides of it). Before the cursor
        # was partitioned this was a real leak: a direct turn from Juniper seeded
        # the next window, making it permanently "mixed" no matter how many
        # ai-town turns followed, so a burst of NPC dialogue right after she
        # spoke still landed in the review queue. Now the closing turn is by
        # construction from this partition, so it can only seed its own.
        next_turns = [closing_turn] if closing_turn else []
        # dict(row).get, not row["source_platform"]: asyncpg Records raise
        # KeyError for an absent column, and the closing turn's own platform is
        # the more specific answer anyway -- the row is only the fallback.
        carried_platform = (
            closing_turn.get("source_platform") if isinstance(closing_turn, dict) else None
        ) or dict(row).get("source_platform")
        await self._pool.execute(
            """
            INSERT INTO memory_consolidation_windows
              (memory_window_id, status, turn_correlation_ids, created_at, source_platform)
            VALUES ($1, 'open', $2::jsonb, $3, $4)
            """,
            new_window_id,
            json.dumps(next_turns),
            now,
            carried_platform,
        )
        return {
            "memory_window_id": row["memory_window_id"],
            "turn_correlation_ids": [t.get("correlation_id") for t in all_turns],
            "turns": all_turns,
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

    async def mark_consolidated_skipped(self, memory_window_id: str, *, reasons: list[str]) -> None:
        await self._pool.execute(
            """
            UPDATE memory_consolidation_windows
            SET status = 'consolidated', consolidation_status = 'skipped'
            WHERE memory_window_id = $1
            """,
            memory_window_id,
        )

    async def mark_crystallization_proposed(
        self,
        memory_window_id: str,
        *,
        crystallization_id: str,
    ) -> None:
        await self._pool.execute(
            """
            UPDATE memory_consolidation_windows
            SET status = 'consolidated', consolidation_status = 'ok', draft_id = $2
            WHERE memory_window_id = $1
            """,
            memory_window_id,
            crystallization_id,
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
