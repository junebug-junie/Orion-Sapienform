from __future__ import annotations

import json
from datetime import datetime, timezone

from psycopg2.extras import Json
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from orion.schemas.field_attention_frame import FieldAttentionFrameV1
from orion.schemas.field_state import FieldStateV1
from orion.schemas.identity_snapshot import IdentitySnapshotV1
from orion.schemas.self_state import SelfStateV1
from orion.schemas.self_state_prediction import SelfStatePredictionV1


class SelfStateRuntimeStore:
    def __init__(self, postgres_uri: str) -> None:
        self._engine: Engine = create_engine(
            postgres_uri,
            pool_pre_ping=True,
            json_serializer=json.dumps,
            json_deserializer=json.loads,
        )

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

    def load_recent_self_states(self, limit: int = 10) -> list[SelfStateV1]:
        with self._engine.connect() as conn:
            rows = (
                conn.execute(
                    text(
                        """
                        SELECT self_state_json FROM substrate_self_state
                        ORDER BY generated_at DESC
                        LIMIT :limit
                        """
                    ),
                    {"limit": limit},
                )
                .mappings()
                .fetchall()
            )
        out: list[SelfStateV1] = []
        for row in rows:
            payload = row["self_state_json"]
            if isinstance(payload, str):
                payload = json.loads(payload)
            out.append(SelfStateV1.model_validate(payload))
        return out

    def load_self_state_for_attention_frame(self, frame_id: str) -> SelfStateV1 | None:
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT self_state_json FROM substrate_self_state
                        WHERE source_attention_frame_id = :frame_id
                        ORDER BY generated_at DESC
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
        payload = row["self_state_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return SelfStateV1.model_validate(payload)

    def save_self_state(self, state: SelfStateV1) -> None:
        now = datetime.now(timezone.utc)
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_self_state (
                        self_state_id,
                        source_field_tick_id,
                        source_attention_frame_id,
                        generated_at,
                        policy_id,
                        self_state_json,
                        created_at
                    ) VALUES (
                        :self_state_id,
                        :source_field_tick_id,
                        :source_attention_frame_id,
                        :generated_at,
                        :policy_id,
                        :self_state_json,
                        :created_at
                    )
                    ON CONFLICT (self_state_id) DO UPDATE SET
                        source_field_tick_id = EXCLUDED.source_field_tick_id,
                        source_attention_frame_id = EXCLUDED.source_attention_frame_id,
                        generated_at = EXCLUDED.generated_at,
                        policy_id = EXCLUDED.policy_id,
                        self_state_json = EXCLUDED.self_state_json
                    """
                ),
                {
                    "self_state_id": state.self_state_id,
                    "source_field_tick_id": state.source_field_tick_id,
                    "source_attention_frame_id": state.source_attention_frame_id,
                    "generated_at": state.generated_at,
                    "policy_id": state.self_state_policy_id,
                    "self_state_json": Json(state.model_dump(mode="json")),
                    "created_at": now,
                },
            )

    def save_identity_snapshot(self, snapshot: IdentitySnapshotV1) -> None:
        now = datetime.now(timezone.utc)
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO identity_snapshots (
                        snapshot_id,
                        source_self_state_id,
                        generated_at,
                        dominant_drive,
                        self_state_condition,
                        snapshot_json,
                        created_at
                    ) VALUES (
                        :snapshot_id,
                        :source_self_state_id,
                        :generated_at,
                        :dominant_drive,
                        :self_state_condition,
                        :snapshot_json,
                        :created_at
                    )
                    ON CONFLICT (snapshot_id) DO NOTHING
                    """
                ),
                {
                    "snapshot_id": snapshot.snapshot_id,
                    "source_self_state_id": snapshot.source_self_state_id,
                    "generated_at": snapshot.generated_at,
                    "dominant_drive": snapshot.dominant_drive,
                    "self_state_condition": snapshot.self_state_condition,
                    "snapshot_json": Json(snapshot.model_dump(mode="json")),
                    "created_at": now,
                },
            )

    def load_latest_identity_snapshot(self) -> IdentitySnapshotV1 | None:
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT snapshot_json FROM identity_snapshots
                        ORDER BY generated_at DESC
                        LIMIT 1
                        """
                    )
                )
                .mappings()
                .first()
            )
        if not row:
            return None
        payload = row["snapshot_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return IdentitySnapshotV1.model_validate(payload)

    def save_self_state_prediction(self, prediction: SelfStatePredictionV1) -> None:
        now = datetime.now(timezone.utc)
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO self_state_predictions (
                        prediction_id,
                        source_self_state_id,
                        generated_at,
                        prediction_json,
                        created_at
                    ) VALUES (
                        :prediction_id,
                        :source_self_state_id,
                        :generated_at,
                        :prediction_json,
                        :created_at
                    )
                    ON CONFLICT (prediction_id) DO NOTHING
                    """
                ),
                {
                    "prediction_id": prediction.prediction_id,
                    "source_self_state_id": prediction.source_self_state_id,
                    "generated_at": prediction.generated_at,
                    "prediction_json": Json(prediction.model_dump(mode="json")),
                    "created_at": now,
                },
            )

    def load_latest_self_state_prediction(self) -> SelfStatePredictionV1 | None:
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT prediction_json FROM self_state_predictions
                        ORDER BY generated_at DESC
                        LIMIT 1
                        """
                    )
                )
                .mappings()
                .first()
            )
        if not row:
            return None
        payload = row["prediction_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return SelfStatePredictionV1.model_validate(payload)
