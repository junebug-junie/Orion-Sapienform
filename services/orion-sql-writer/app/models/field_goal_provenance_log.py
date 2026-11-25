from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Index, JSON, String, func
from sqlalchemy.dialects.postgresql import JSONB

from app.db import Base

_JSONB = JSON().with_variant(JSONB(), "postgresql")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class FieldGoalProvenanceLogSQL(Base):
    """One row per goal Orion proposes (issue #3924).

    Channel `orion:memory:goals:proposed`, kind `memory.field_goals.proposed.v1`
    (`orion.schemas.field_goal.FieldGoalProvenanceV1`). Producer: orion-attention-runtime
    `_publish_goal`, which logs `field_goal_provenance_published` per publish. Before this
    table that log line was the only record, and a restart erased it.

    `artifact_id` is a uuid the producer sets once per goal, so a re-delivered envelope
    carries the same id: the model is insert-only and the duplicate-key catch skips it.
    Scalar columns keep pydantic field names (generic `_write_row()` path); `ts` is the
    goal's own timestamp, `received_at` is when sql-writer stored it. `payload_json` is the
    validated payload exactly as the schema defines it (no envelope extras).
    The producer's dominance-streak count is NOT on the schema (it lives only in the
    producer log), so there is no streak column.
    """

    __tablename__ = "field_goal_provenance_log"
    __table_args__ = (
        Index("ix_field_goal_provenance_log_received_at", "received_at"),
        Index("ix_field_goal_provenance_log_ts", "ts"),
        Index("ix_field_goal_provenance_log_target_ts", "field_target_id", "ts"),
    )

    artifact_id = Column(String, primary_key=True)
    field_target_id = Column(String, nullable=False)
    target_kind = Column(String, nullable=False)
    salience_score = Column(Float, nullable=False)
    priority = Column(Float, nullable=False)
    proposal_status = Column(String, nullable=False)
    source_field_tick_id = Column(String, nullable=False)
    source_attention_frame_id = Column(String, nullable=False)
    confidence = Column(Float, nullable=True)
    correlation_id = Column(String, nullable=True)
    ts = Column(DateTime(timezone=True), nullable=False)
    envelope_id = Column(String, nullable=True)
    received_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow, server_default=func.now())
    payload_json = Column(_JSONB, nullable=False)
