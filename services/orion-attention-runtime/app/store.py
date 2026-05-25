from __future__ import annotations

import json
from datetime import datetime, timezone

from psycopg2.extras import Json
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from orion.schemas.field_attention_frame import FieldAttentionFrameV1
from orion.schemas.field_state import FieldStateV1


class AttentionRuntimeStore:
    def __init__(self, postgres_uri: str) -> None:
        self._engine: Engine = create_engine(
            postgres_uri,
            pool_pre_ping=True,
            json_serializer=json.dumps,
            json_deserializer=json.loads,
        )

    def load_latest_field(self) -> FieldStateV1 | None:
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
        return FieldStateV1.model_validate(payload)

    def load_latest_attention_frame(self) -> FieldAttentionFrameV1 | None:
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT frame_json FROM substrate_attention_frames
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
        payload = row["frame_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return FieldAttentionFrameV1.model_validate(payload)

    def load_attention_frame_for_field_tick(self, tick_id: str) -> FieldAttentionFrameV1 | None:
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT frame_json FROM substrate_attention_frames
                        WHERE source_field_tick_id = :tick_id
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
        payload = row["frame_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return FieldAttentionFrameV1.model_validate(payload)

    def save_attention_frame(self, frame: FieldAttentionFrameV1) -> None:
        now = datetime.now(timezone.utc)
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_attention_frames (
                        frame_id,
                        source_field_tick_id,
                        source_field_generated_at,
                        generated_at,
                        policy_id,
                        frame_json,
                        created_at
                    ) VALUES (
                        :frame_id,
                        :source_field_tick_id,
                        :source_field_generated_at,
                        :generated_at,
                        :policy_id,
                        :frame_json,
                        :created_at
                    )
                    ON CONFLICT (frame_id) DO UPDATE SET
                        source_field_tick_id = EXCLUDED.source_field_tick_id,
                        source_field_generated_at = EXCLUDED.source_field_generated_at,
                        generated_at = EXCLUDED.generated_at,
                        policy_id = EXCLUDED.policy_id,
                        frame_json = EXCLUDED.frame_json
                    """
                ),
                {
                    "frame_id": frame.frame_id,
                    "source_field_tick_id": frame.source_field_tick_id,
                    "source_field_generated_at": frame.source_field_generated_at,
                    "generated_at": frame.generated_at,
                    "policy_id": frame.attention_policy_id,
                    "frame_json": Json(frame.model_dump(mode="json")),
                    "created_at": now,
                },
            )
