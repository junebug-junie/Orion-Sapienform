from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from psycopg2.extras import Json
from pydantic import ValidationError
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from orion.schemas.field_attention_frame import FieldAttentionFrameV1
from orion.schemas.field_state import FieldStateV1
from orion.schemas.proposal_frame import ProposalFrameV1

logger = logging.getLogger("orion.proposal_runtime.store")


class ProposalRuntimeStore:
    def __init__(self, postgres_uri: str) -> None:
        self._engine: Engine = create_engine(
            postgres_uri,
            pool_pre_ping=True,
            json_serializer=json.dumps,
            json_deserializer=json.loads,
        )

    def load_latest_field(self) -> FieldStateV1 | None:
        """2026-07-22 (SelfStateV1 burn): replaces load_latest_self_state() as
        the poll-loop's trigger source. FieldStateV1 was always the real
        upstream tick; self_state was a lossy pass-through hop."""
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT field_json FROM substrate_field_state
                        ORDER BY generated_at DESC
                        LIMIT 1
                        """
                    ),
                )
                .mappings()
                .first()
            )
        if not row:
            return None
        payload = row["field_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        try:
            return FieldStateV1.model_validate(payload)
        except ValidationError:
            # Looked up as "latest", not a fixed id, so this can't stall a
            # FIFO queue the way policy/execution-dispatch-runtime's fixed-id
            # lookups could -- but still degrade instead of crash-looping if
            # the very latest row is somehow schema-incompatible.
            logger.warning("field_state_incompatible_schema", exc_info=True)
            return None

    def load_attention_frame_for_field_tick(self, field_tick_id: str) -> FieldAttentionFrameV1 | None:
        """2026-07-22 (SelfStateV1 burn): looks up by source_field_tick_id
        directly rather than by a self-state-provided frame_id -- attention
        frames were always keyed to a field tick underneath.

        2026-08-19: reads the source_field_tick_id COLUMN, not the same key
        extracted out of frame_json. The table has carried
        idx_substrate_attention_frames_source_tick on that column all along, but
        `frame_json ->> 'source_field_tick_id'` is an expression the index cannot
        answer, so the planner fell back to walking
        idx_substrate_attention_frames_generated_at end to end and de-TOASTing
        every JSON blob to evaluate the filter. Measured live with
        EXPLAIN (ANALYZE, BUFFERS): 553,906 buffers touched (~4.3 GB) and
        4,777 ms for a single lookup that returns at most one row.

        Safe to swap: the column and the JSON key agree on all 99,626 rows in
        the live table (`where source_field_tick_id is distinct from
        (frame_json ->> 'source_field_tick_id')` returns 0), and the column is
        NOT NULL-populated for every row.
        """
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT frame_json FROM substrate_attention_frames
                        WHERE source_field_tick_id = :field_tick_id
                        ORDER BY generated_at DESC
                        LIMIT 1
                        """
                    ),
                    {"field_tick_id": field_tick_id},
                )
                .mappings()
                .first()
            )
        if not row:
            return None
        payload = row["frame_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return FieldAttentionFrameV1.model_validate(payload)

    def load_recent_reverie_thought(self, *, max_age_sec: float = 300.0):
        """Latest fresh non-hollow spontaneous thought, or None (Phase B).

        Degrades to None on any error (missing table, bad payload) — a reverie
        read must never break proposal generation.
        """
        from orion.schemas.reverie import SpontaneousThoughtV1

        try:
            with self._engine.connect() as conn:
                row = (
                    conn.execute(
                        text(
                            """
                            SELECT thought_json, created_at FROM substrate_reverie_thought
                            ORDER BY created_at DESC
                            LIMIT 1
                            """
                        ),
                    )
                    .mappings()
                    .first()
                )
            if not row:
                return None
            created_at = row.get("created_at")
            if isinstance(created_at, datetime):
                ts = created_at if created_at.tzinfo else created_at.replace(tzinfo=timezone.utc)
                if (datetime.now(timezone.utc) - ts).total_seconds() > max_age_sec:
                    return None
            else:
                return None  # unknown freshness → treat as stale (defensive)
            payload = row["thought_json"]
            if isinstance(payload, str):
                payload = json.loads(payload)
            thought = SpontaneousThoughtV1.model_validate(payload)
            # Trust the stamped hollow decision persisted at generation (incl.
            # semantic-lift audit-ref grounding); recomputing here would lose that
            # context and falsely drop a valid thought.
            return None if thought.hollow else thought
        except Exception:
            return None

    def load_repair_pressure_readings(self, *, limit: int = 500, window_days: int = 7):
        """Real repair-pressure appraisals, oldest-first, for hop 0's reducer.

        Returns `list[MetacogTrendReading]`. Degrades to `[]` on any error --
        a metacog read must never break proposal generation, same discipline as
        `load_recent_reverie_thought` above.

        **No `confidence > 0` filter, deliberately.** That gate was the right
        discriminator until 2026-07-30, when `repair_pressure_v2`'s confidence
        fix landed: previously a confidently-calm text-fallback reading was
        persisted with `confidence=0.0`, indistinguishable from the appraiser's
        true "no evidence" signal, so gating on confidence separated real
        readings from fallback ones. Post-fix every row carries the real
        `_TEXT_FALLBACK_CONFIDENCE` (0.65), so the gate now filters nothing --
        verified live 2026-07-31: rows on 07-24 were 5/26 confidence>0, rows on
        07-31 are 5/5 at avg confidence 0.650.

        Filtering it back out would discard exactly the readings that establish
        the rest state (a confidently-calm 0.087 IS the calm baseline the
        z-score needs to be anomalous against), so the reducer folds every real
        row and lets the EWMA baseline do the discriminating.

        **Explicit time window, not just a row cap** (review finding
        2026-07-31): a bare `LIMIT 500` means that once history exceeds the cap
        the replay's start point silently slides forward, so the EWMA restarts
        from row N-500 instead of continuing -- neither "full history" nor a
        defined window. `created_at` is a varchar and the table has no index on
        it (only `id`/`correlation_id`), so the cast-and-compare is a scan
        today; fine at ~8.7 rows/day, and an index is the right fix if this
        table ever grows fast. Ordering casts to timestamptz rather than
        relying on lexicographic varchar ordering -- that happens to be correct
        for the current producer's uniform ISO format, but that is a property
        of the producer, not a constraint.
        """
        from orion.metacog.trend_reducer import MetacogTrendReading

        try:
            with self._engine.connect() as conn:
                rows = (
                    conn.execute(
                        text(
                            """
                            SELECT created_at, level, confidence
                            FROM repair_pressure_appraisal_log
                            WHERE created_at::timestamptz
                                  >= NOW() - (:window_days * INTERVAL '1 day')
                            ORDER BY created_at::timestamptz DESC
                            LIMIT :limit
                            """
                        ),
                        {"limit": int(limit), "window_days": int(window_days)},
                    )
                    .mappings()
                    .all()
                )
        except Exception:
            return []

        readings = []
        for row in reversed(rows):  # oldest-first: the reducer folds forward
            created_at = row.get("created_at")
            if isinstance(created_at, str):
                try:
                    created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                except ValueError:
                    continue
            if not isinstance(created_at, datetime):
                continue
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            try:
                level = float(row["level"])
                confidence = float(row["confidence"])
            except (TypeError, ValueError, KeyError):
                continue
            readings.append(
                MetacogTrendReading(at=created_at, level=level, confidence=confidence)
            )
        return readings

    def load_attention_frame(self, frame_id: str) -> FieldAttentionFrameV1 | None:
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT frame_json FROM substrate_attention_frames
                        WHERE frame_id = :frame_id
                        LIMIT 1
                        """
                    ),
                    {"frame_id": frame_id},
                )
                .mappings()
                .first()
            )
        if not row:
            return None
        payload = row["frame_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return FieldAttentionFrameV1.model_validate(payload)

    def load_latest_proposal_frame(self) -> ProposalFrameV1 | None:
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT proposal_frame_json FROM substrate_proposal_frames
                        ORDER BY generated_at DESC
                        LIMIT 1
                        """
                    ),
                )
                .mappings()
                .first()
            )
        if not row:
            return None
        payload = row["proposal_frame_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        try:
            return ProposalFrameV1.model_validate(payload)
        except ValidationError:
            # Looked up as "latest", not a fixed id -- can't stall a FIFO
            # queue the way a fixed-id lookup could, but a schema migration
            # (e.g. 2026-07-22's SelfStateV1 burn, which removed
            # source_self_state_id/added a required source_field_generated_at)
            # can still leave the single latest row incompatible with the
            # currently-running code. Degrade instead of crash-looping.
            logger.warning("proposal_frame_incompatible_schema latest_lookup", exc_info=True)
            return None

    def load_proposal_frame_for_field_tick(self, field_tick_id: str) -> ProposalFrameV1 | None:
        """2026-07-22 (SelfStateV1 burn): replaces load_proposal_frame_for_self_state.
        Dedup key is now the field tick directly, via the already-existing
        source_field_tick_id column -- no self-state hop needed."""
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT proposal_frame_json FROM substrate_proposal_frames
                        WHERE source_field_tick_id = :field_tick_id
                        ORDER BY generated_at DESC
                        LIMIT 1
                        """
                    ),
                    {"field_tick_id": field_tick_id},
                )
                .mappings()
                .first()
            )
        if not row:
            return None
        payload = row["proposal_frame_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        try:
            return ProposalFrameV1.model_validate(payload)
        except ValidationError:
            # Dedup lookup keyed by field_tick_id, which pre-migration rows
            # can also carry (source_field_tick_id predates the SelfStateV1
            # burn) -- a schema-incompatible match here must not crash-loop
            # the tick; treat as "no existing frame for this tick" instead.
            logger.warning(
                "proposal_frame_incompatible_schema field_tick_id=%s", field_tick_id, exc_info=True
            )
            return None

    def save_proposal_frame(self, frame: ProposalFrameV1) -> None:
        now = datetime.now(timezone.utc)
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_proposal_frames (
                        frame_id,
                        source_attention_frame_id,
                        source_field_tick_id,
                        source_field_generated_at,
                        generated_at,
                        policy_id,
                        proposal_frame_json,
                        created_at
                    ) VALUES (
                        :frame_id,
                        :source_attention_frame_id,
                        :source_field_tick_id,
                        :source_field_generated_at,
                        :generated_at,
                        :policy_id,
                        :proposal_frame_json,
                        :created_at
                    )
                    ON CONFLICT (frame_id) DO UPDATE SET
                        source_attention_frame_id = EXCLUDED.source_attention_frame_id,
                        source_field_tick_id = EXCLUDED.source_field_tick_id,
                        source_field_generated_at = EXCLUDED.source_field_generated_at,
                        generated_at = EXCLUDED.generated_at,
                        policy_id = EXCLUDED.policy_id,
                        proposal_frame_json = EXCLUDED.proposal_frame_json
                    """
                ),
                {
                    "frame_id": frame.frame_id,
                    "source_attention_frame_id": frame.source_attention_frame_id,
                    "source_field_tick_id": frame.source_field_tick_id,
                    "source_field_generated_at": frame.source_field_generated_at,
                    "generated_at": frame.generated_at,
                    "policy_id": frame.proposal_policy_id,
                    "proposal_frame_json": Json(frame.model_dump(mode="json")),
                    "created_at": now,
                },
            )
