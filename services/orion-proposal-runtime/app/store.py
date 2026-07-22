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
        frames were always keyed to a field tick underneath."""
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT frame_json FROM substrate_attention_frames
                        WHERE (frame_json ->> 'source_field_tick_id') = :field_tick_id
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
        return ProposalFrameV1.model_validate(payload)

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
        return ProposalFrameV1.model_validate(payload)

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
