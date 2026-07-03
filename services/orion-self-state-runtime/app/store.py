from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from psycopg2.extras import Json
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from orion.schemas.attention_frame import AttentionBroadcastProjectionV1
from orion.schemas.field_attention_frame import FieldAttentionFrameV1
from orion.schemas.field_state import FieldStateV1
from orion.schemas.identity_snapshot import IdentitySnapshotV1
from orion.schemas.self_state import SelfStateV1
from orion.schemas.self_state_prediction import SelfStatePredictionV1

logger = logging.getLogger("orion.self_state.runtime.store")

# Observability inputs are best-effort: rows older than these gates are treated
# as absent so self-state degrades to schema defaults instead of stale data.
ATTENTION_BROADCAST_MAX_AGE_SEC = 300.0
HUB_PRESENCE_MAX_AGE_SEC = 600.0


def _age_seconds(generated_at: datetime, now: datetime) -> float:
    if generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=timezone.utc)
    return (now - generated_at).total_seconds()


def _prune_sql(table: str, pk: str) -> str:
    # Batched, guard-railed prune: never deletes the newest row (by
    # generated_at, matching the latest-row readers' ordering).
    return f"""
DELETE FROM {table}
WHERE ctid IN (
    SELECT ctid
    FROM {table}
    WHERE created_at < :cutoff
      AND {pk} <> (
          SELECT {pk} FROM {table}
          ORDER BY generated_at DESC LIMIT 1
      )
    ORDER BY created_at ASC
    LIMIT :batch_size
)
"""


# Trusted module constants (NOT user input) — table/pk names are literals.
PRUNE_HISTORY_SQL: dict[str, str] = {
    "substrate_self_state": _prune_sql("substrate_self_state", "self_state_id"),
    "self_state_predictions": _prune_sql("self_state_predictions", "prediction_id"),
    "identity_snapshots": _prune_sql("identity_snapshots", "snapshot_id"),
}


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

    def load_latest_attention_broadcast(self) -> AttentionBroadcastProjectionV1 | None:
        """Best-effort read of the latest attention broadcast projection.

        Returns None on any failure (missing table — the migration is manual —
        parse errors, connection errors) or when the row is stale.
        """
        try:
            with self._engine.connect() as conn:
                row = (
                    conn.execute(
                        text(
                            """
                            SELECT projection_json, generated_at
                            FROM substrate_attention_broadcast_projection
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
            age = _age_seconds(row["generated_at"], datetime.now(timezone.utc))
            if age > ATTENTION_BROADCAST_MAX_AGE_SEC:
                logger.debug(
                    "attention_broadcast_stale age_sec=%.1f gate_sec=%.1f",
                    age,
                    ATTENTION_BROADCAST_MAX_AGE_SEC,
                )
                return None
            payload = row["projection_json"]
            if isinstance(payload, str):
                payload = json.loads(payload)
            return AttentionBroadcastProjectionV1.model_validate(payload)
        except Exception:
            logger.debug("attention_broadcast_load_failed", exc_info=True)
            return None

    def load_hub_presence(self) -> dict[str, Any] | None:
        """Best-effort read of the single-row hub presence snapshot.

        Returns the parsed presence_json dict with an added "as_of" key
        (generated_at isoformat), or None on staleness or any failure
        (missing table — the migration is manual — parse errors, etc.).
        """
        try:
            with self._engine.connect() as conn:
                row = (
                    conn.execute(
                        text(
                            """
                            SELECT presence_json, generated_at
                            FROM substrate_hub_presence
                            WHERE presence_id = 'hub'
                            LIMIT 1
                            """
                        ),
                    )
                    .mappings()
                    .first()
                )
            if not row:
                return None
            generated_at = row["generated_at"]
            age = _age_seconds(generated_at, datetime.now(timezone.utc))
            if age > HUB_PRESENCE_MAX_AGE_SEC:
                logger.debug(
                    "hub_presence_stale age_sec=%.1f gate_sec=%.1f",
                    age,
                    HUB_PRESENCE_MAX_AGE_SEC,
                )
                return None
            payload = row["presence_json"]
            if isinstance(payload, str):
                payload = json.loads(payload)
            if not isinstance(payload, dict) or not payload:
                return None
            if generated_at.tzinfo is None:
                generated_at = generated_at.replace(tzinfo=timezone.utc)
            payload["as_of"] = generated_at.isoformat()
            return payload
        except Exception:
            logger.debug("hub_presence_load_failed", exc_info=True)
            return None

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

    def prune_history(self, *, retention_hours: float, batch_size: int = 5000) -> int:
        if retention_hours <= 0:
            return 0
        cutoff = datetime.now(timezone.utc) - timedelta(hours=retention_hours)
        total_deleted = 0
        for sql in PRUNE_HISTORY_SQL.values():
            while True:
                with self._engine.begin() as conn:
                    result = conn.execute(
                        text(sql), {"cutoff": cutoff, "batch_size": batch_size}
                    )
                deleted = result.rowcount or 0
                total_deleted += deleted
                if deleted < batch_size:
                    break
        return total_deleted
