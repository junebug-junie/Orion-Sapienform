from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from psycopg2.extras import Json
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from orion.schemas.field_attention_frame import FieldAttentionFrameV1
from orion.schemas.field_state import FieldStateV1

# Batched, guard-railed prune: never deletes the newest frame (by generated_at,
# matching load_latest_attention_frame's ordering).
PRUNE_ATTENTION_FRAMES_SQL = """
DELETE FROM substrate_attention_frames
WHERE ctid IN (
    SELECT ctid
    FROM substrate_attention_frames
    WHERE created_at < :cutoff
      AND frame_id <> (
          SELECT frame_id FROM substrate_attention_frames
          ORDER BY generated_at DESC LIMIT 1
      )
    ORDER BY created_at ASC
    LIMIT :batch_size
)
"""


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

    def load_prediction_error_history(self, *, reducer_key: str, limit: int) -> list[float]:
        """Real, ASC-by-time prediction-error history for one reducer, for
        Candidate A (`orion/attention/field_attention/candidate_precision_
        weighted.py::precision_weighted_salience`) to compute real precision
        (1/variance) from -- same query shape as that module's own docstring
        and `scripts/analysis/measure_precision_weighted_salience_probe.py`.

        `substrate_reduction_receipts` retains success receipts for only
        `ORION_RECEIPT_RETENTION_SUCCESS_MINUTES` (30 min live default) --
        this is always a rolling recent window, not full history, a
        structural property of the source table, not a bug here. Degrades to
        `[]` on any error (missing table, bad row) -- a history-fetch failure
        must never crash the attention tick; `precision_weighted_salience([])`
        already handles the empty case honestly (zero salience, n_samples=0).
        """
        try:
            with self._engine.connect() as conn:
                rows = (
                    conn.execute(
                        text(
                            """
                            SELECT
                                receipt_json -> 'state_deltas' -> 0 -> 'after'
                                    -> 'pressure_hints' ->> 'prediction_error' AS error
                            FROM substrate_reduction_receipts
                            WHERE (receipt_json -> 'state_deltas' -> 0 ->> 'reducer_id')
                                  = :reducer_id
                            ORDER BY created_at ASC
                            LIMIT :limit
                            """
                        ),
                        {"reducer_id": f"substrate.{reducer_key}", "limit": limit},
                    )
                    .mappings()
                    .all()
                )
        except Exception:
            return []

        out: list[float] = []
        for row in rows:
            value = row.get("error")
            if value is None:
                continue
            try:
                out.append(float(value))
            except (TypeError, ValueError):
                continue
        return out

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

    def prune_attention_frames(self, *, retention_hours: float, batch_size: int = 5000) -> int:
        if retention_hours <= 0:
            return 0
        cutoff = datetime.now(timezone.utc) - timedelta(hours=retention_hours)
        total_deleted = 0
        while True:
            with self._engine.begin() as conn:
                result = conn.execute(
                    text(PRUNE_ATTENTION_FRAMES_SQL),
                    {"cutoff": cutoff, "batch_size": batch_size},
                )
            deleted = result.rowcount or 0
            total_deleted += deleted
            if deleted < batch_size:
                break
        return total_deleted

    def attention_frame_oldest_age_hours(self) -> float | None:
        # Keys on created_at -- the same column PRUNE_ATTENTION_FRAMES_SQL's cutoff
        # filters on -- so staleness detection can never disagree with the pruner
        # about which column defines "age".
        with self._engine.connect() as conn:
            row = (
                conn.execute(text("SELECT min(created_at) AS oldest FROM substrate_attention_frames"))
                .mappings()
                .first()
            )
        oldest = row["oldest"] if row else None
        if oldest is None:
            return None
        if oldest.tzinfo is None:
            oldest = oldest.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - oldest).total_seconds() / 3600.0
