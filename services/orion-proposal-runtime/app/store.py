from __future__ import annotations

import json
from datetime import datetime, timezone

from psycopg2.extras import Json
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from orion.schemas.field_attention_frame import FieldAttentionFrameV1
from orion.schemas.field_state import FieldStateV1
from orion.schemas.proposal_frame import ProposalFrameV1
from orion.schemas.self_state import SelfStateV1


class ProposalRuntimeStore:
    def __init__(self, postgres_uri: str) -> None:
        self._engine: Engine = create_engine(
            postgres_uri,
            pool_pre_ping=True,
            json_serializer=json.dumps,
            json_deserializer=json.loads,
        )

    def load_latest_self_state(self) -> SelfStateV1 | None:
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT self_state_json FROM substrate_self_state
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
        payload = row["self_state_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return SelfStateV1.model_validate(payload)

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

    def load_field_for_tick(self, tick_id: str) -> FieldStateV1 | None:
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT field_json FROM substrate_field_state
                        WHERE tick_id = :tick_id
                        ORDER BY generated_at DESC
                        LIMIT 1
                        """
                    ),
                    {"tick_id": tick_id},
                )
                .mappings()
                .first()
            )
        if not row:
            return None
        payload = row["field_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return FieldStateV1.model_validate(payload)

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

    def load_proposal_frame_for_self_state(self, self_state_id: str) -> ProposalFrameV1 | None:
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT proposal_frame_json FROM substrate_proposal_frames
                        WHERE source_self_state_id = :self_state_id
                        ORDER BY generated_at DESC
                        LIMIT 1
                        """
                    ),
                    {"self_state_id": self_state_id},
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
                        source_self_state_id,
                        source_attention_frame_id,
                        source_field_tick_id,
                        generated_at,
                        policy_id,
                        proposal_frame_json,
                        created_at
                    ) VALUES (
                        :frame_id,
                        :source_self_state_id,
                        :source_attention_frame_id,
                        :source_field_tick_id,
                        :generated_at,
                        :policy_id,
                        :proposal_frame_json,
                        :created_at
                    )
                    ON CONFLICT (frame_id) DO UPDATE SET
                        source_self_state_id = EXCLUDED.source_self_state_id,
                        source_attention_frame_id = EXCLUDED.source_attention_frame_id,
                        source_field_tick_id = EXCLUDED.source_field_tick_id,
                        generated_at = EXCLUDED.generated_at,
                        policy_id = EXCLUDED.policy_id,
                        proposal_frame_json = EXCLUDED.proposal_frame_json
                    """
                ),
                {
                    "frame_id": frame.frame_id,
                    "source_self_state_id": frame.source_self_state_id,
                    "source_attention_frame_id": frame.source_attention_frame_id,
                    "source_field_tick_id": frame.source_field_tick_id,
                    "generated_at": frame.generated_at,
                    "policy_id": frame.proposal_policy_id,
                    "proposal_frame_json": Json(frame.model_dump(mode="json")),
                    "created_at": now,
                },
            )
