from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Index, Integer, JSON, String, func
from sqlalchemy.dialects.postgresql import JSONB

from app.db import Base

_JSONB = JSON().with_variant(JSONB(), "postgresql")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class FieldGoalEpisodeSQL(Base):
    """One row per goal EPISODE: a run of Orion wanting the same target (issue #3924).

    Channel `orion:memory:goals:proposed`, kind `memory.field_goals.proposed.v1`
    (`orion.schemas.field_goal.FieldGoalProvenanceV1`), producer orion-attention-runtime
    `_publish_goal`. The producer re-publishes the held goal on every qualifying tick
    (~1,700/h live on 2026-11-25), so one row per message would be mostly repeats. The fold
    rule lives in `app/field_goal_episode.py`. `repeat_count` = goal messages folded in, so
    `sum(repeat_count)` equals the number of goals published.
    """

    __tablename__ = "field_goal_episode"
    __table_args__ = (
        Index("ix_field_goal_episode_first_ts", "first_ts"),
        Index("ix_field_goal_episode_target_last_ts", "field_target_id", "last_ts"),
    )

    episode_id = Column(String, primary_key=True)
    field_target_id = Column(String, nullable=False)
    target_kind = Column(String, nullable=False)
    first_artifact_id = Column(String, nullable=False)
    first_ts = Column(DateTime(timezone=True), nullable=False)
    last_ts = Column(DateTime(timezone=True), nullable=False)
    repeat_count = Column(Integer, nullable=False)
    max_salience = Column(Float, nullable=False)
    last_salience = Column(Float, nullable=False)
    first_payload_json = Column(_JSONB, nullable=False)
    first_envelope_id = Column(String, nullable=True)
    received_at_first = Column(DateTime(timezone=True), nullable=False, default=_utcnow, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow, server_default=func.now())


class FieldGoalEpisodeArtifactSQL(Base):
    """Seen goal artifact_ids, so a re-delivered goal does not bump `repeat_count`.

    PK `artifact_id` is the dedup key. Rows older than the dedup window are pruned on write
    (`app/field_goal_episode.py` ARTIFACT_DEDUP_WINDOW), so this table stays small.
    """

    __tablename__ = "field_goal_episode_artifact"
    __table_args__ = (Index("ix_field_goal_episode_artifact_seen_at", "seen_at"),)

    artifact_id = Column(String, primary_key=True)
    episode_id = Column(String, nullable=False)
    seen_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow, server_default=func.now())
