from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import asyncpg

from orion.schemas.memory_consolidation import MemoryTurnPersistedV1


def _apply_scores_to_entry(entry: dict[str, Any], scores: dict[str, Any]) -> None:
    for key in ("memory_significance_score", "conversation_boundary_score", "memory_classify_ts"):
        if key in scores:
            entry[key] = scores.get(key)
    appraisal = scores.get("turn_change_appraisal")
    if isinstance(appraisal, dict):
        meta = entry.get("spark_meta") if isinstance(entry.get("spark_meta"), dict) else {}
        meta["turn_change_appraisal"] = appraisal
        sig = scores.get("memory_significance_score")
        if isinstance(sig, (int, float)):
            meta["memory_significance_score"] = float(sig)
        entry["spark_meta"] = meta


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

    async def find_windowed_turn(
        self, correlation_id: str, *, lookback_hours: int = 72
    ) -> dict[str, Any] | None:
        """The window entry already holding this turn, if any (newest window first).

        Boundary Fix 2. orion-sql-writer publishes orion:memory:turn:persisted
        twice per turn (see append_turn's comment). The second copy used to be
        classified again, and because the first copy had already been appended
        (or carried into the next window as its seed), the second pass took the
        turn itself as its "previous turn" baseline. That self-comparison is
        why every turn's persisted conversation_boundary_score read low
        (0.004-0.685 on 2026-09-28) while the score that actually closed the
        window read 0.96-1.00: two different classifications, the later one
        overwriting chat_history_log. A turn already in a window has been
        classified; this is the check that keeps it to once.
        """
        row = await self._pool.fetchrow(
            """
            SELECT memory_window_id, turn_correlation_ids
            FROM memory_consolidation_windows
            WHERE turn_correlation_ids @> $1::jsonb
              AND created_at > now() - make_interval(hours => $2)
            ORDER BY created_at DESC
            LIMIT 1
            """,
            json.dumps([{"correlation_id": correlation_id}]),
            int(lookback_hours),
        )
        if row is None:
            return None
        turns = json.loads(row["turn_correlation_ids"]) if row["turn_correlation_ids"] else []
        for entry in turns if isinstance(turns, list) else []:
            if isinstance(entry, dict) and entry.get("correlation_id") == correlation_id:
                return {"memory_window_id": row["memory_window_id"], **entry}
        return None

    async def update_turn_scores(self, correlation_id: str, *, scores: dict[str, Any]) -> int:
        """Rewrite this turn's scores in every window that holds it. Returns rows updated.

        Used by the degraded-classify retry so the window keeps the same score
        the retry patches into chat_history_log (Fix 2: one score per turn).
        """
        rows = await self._pool.fetch(
            """
            SELECT memory_window_id, turn_correlation_ids
            FROM memory_consolidation_windows
            WHERE turn_correlation_ids @> $1::jsonb
            """,
            json.dumps([{"correlation_id": correlation_id}]),
        )
        updated = 0
        for row in rows:
            turns = json.loads(row["turn_correlation_ids"]) if row["turn_correlation_ids"] else []
            if not isinstance(turns, list):
                continue
            changed = False
            for entry in turns:
                if isinstance(entry, dict) and entry.get("correlation_id") == correlation_id:
                    _apply_scores_to_entry(entry, scores)
                    changed = True
            if changed:
                await self._pool.execute(
                    "UPDATE memory_consolidation_windows SET turn_correlation_ids = $2::jsonb WHERE memory_window_id = $1",
                    row["memory_window_id"],
                    json.dumps(turns),
                )
                updated += 1
        return updated

    async def record_close_audit(
        self,
        memory_window_id: str,
        *,
        close_reason: str | None,
        boundary_score_at_close: float | None,
    ) -> None:
        """Name why the live (legacy) rule closed this window.

        Separate statement from the close itself and best-effort in the caller:
        the columns arrive with manual_migration_memory_episode_v1.sql, and a
        service deployed before that migration must still close windows.
        """
        await self._pool.execute(
            """
            UPDATE memory_consolidation_windows
            SET close_reason = $2, boundary_score_at_close = $3
            WHERE memory_window_id = $1
            """,
            memory_window_id,
            close_reason,
            boundary_score_at_close,
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
            "conversation_phase": turn.spark_meta.get("conversation_phase")
            if isinstance(turn.spark_meta.get("conversation_phase"), dict)
            else None,
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
        if not isinstance(turns, list):
            turns = []
        # orion:memory:turn:persisted can be published more than once for the
        # same correlation_id -- confirmed against the real producer
        # (services/orion-sql-writer/app/worker.py): a "chat.history" envelope
        # write emits it directly, and a separate, independent branch emits it
        # again for the matching "chat.history.message.v1" (assistant-role)
        # envelope via _maybe_emit_memory_turn_from_row(). Neither branch
        # reclassifies anything -- this is a producer-side double-publish, not
        # a legitimate two-phase classify design. A blind append here built up
        # two window entries for one real turn, which then made
        # fetch_grammar_evidence_for_window() query the same grammar trace_id
        # twice and build_crystallization_from_window() mint two evidence rows
        # citing the same source_id. Confirmed live 2026-08-20: 571 duplicate
        # memory_crystallization_sources groups (55 chat_turn + 516
        # grammar_event), every pair sharing one insert timestamp -- i.e. minted
        # together from one already-duplicated evidence list, not two separate
        # writes. This is a consumer-side guard, not a fix for the double
        # publish itself -- any other future consumer of this channel needs its
        # own dedup too; see the PR description for the sql-writer follow-up.
        #
        # Collapse ALL existing entries for this correlation_id (not just the
        # first) into the fresh one, keeping the position of the first
        # occurrence. A window already carrying legacy duplicates from before
        # this fix shipped must not have a still-stale second entry survive
        # sitting after the fresh replacement -- build_crystallization_from_window's
        # own dedup keeps the LAST occurrence it sees, so a leftover stale
        # entry positioned after the fresh one would silently win instead.
        new_turns: list[dict[str, Any]] = []
        replaced = False
        for existing in turns:
            if isinstance(existing, dict) and existing.get("correlation_id") == turn.correlation_id:
                if not replaced:
                    new_turns.append(turn_entry)
                    replaced = True
                continue
            new_turns.append(existing)
        if not replaced:
            new_turns.append(turn_entry)
        turns = new_turns
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
