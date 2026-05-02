from sqlalchemy import Column, DateTime, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from app.db import Base


class WorldPulseRunSQL(Base):
    __tablename__ = "world_pulse_run"
    run_id = Column(String, primary_key=True)
    status = Column(String, nullable=False)
    date = Column(String, nullable=False)
    requested_by = Column(String, nullable=True)
    dry_run = Column(String, nullable=True)
    payload_json = Column(JSONB, nullable=False, default=dict)
    schema_version = Column(String, nullable=False, default="v1")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=True)


class WorldPulseDigestSQL(Base):
    __tablename__ = "world_pulse_digest"
    run_id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    date = Column(String, nullable=False)
    executive_summary = Column(Text, nullable=False)
    payload_json = Column(JSONB, nullable=False, default=dict)
    schema_version = Column(String, nullable=False, default="v1")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class WorldPulseDigestItemSQL(Base):
    __tablename__ = "world_pulse_digest_item"
    item_id = Column(String, primary_key=True)
    run_id = Column(String, nullable=False)
    category = Column(String, nullable=False)
    title = Column(String, nullable=False)
    confidence = Column(String, nullable=True)
    payload_json = Column(JSONB, nullable=False, default=dict)
    schema_version = Column(String, nullable=False, default="v1")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index("idx_world_pulse_digest_item_run_id", "run_id"),
        Index("idx_world_pulse_digest_item_category", "category"),
    )


class WorldPulseArticleSQL(Base):
    __tablename__ = "world_pulse_article"
    article_id = Column(String, primary_key=True)
    run_id = Column(String, nullable=False)
    source_id = Column(String, nullable=False)
    title = Column(String, nullable=False)
    url = Column(Text, nullable=False)
    payload_json = Column(JSONB, nullable=False, default=dict)
    schema_version = Column(String, nullable=False, default="v1")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index("idx_world_pulse_article_run_id", "run_id"),
        Index("idx_world_pulse_article_source_id", "source_id"),
    )


class WorldPulseArticleClusterSQL(Base):
    __tablename__ = "world_pulse_article_cluster"
    cluster_id = Column(String, primary_key=True)
    run_id = Column(String, nullable=False)
    category = Column(String, nullable=False)
    title = Column(String, nullable=False)
    article_count = Column(String, nullable=False, default="0")
    payload_json = Column(JSONB, nullable=False, default=dict)
    schema_version = Column(String, nullable=False, default="v1")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index("idx_world_pulse_cluster_run_id", "run_id"),
        Index("idx_world_pulse_cluster_category", "category"),
    )


class WorldPulseClaimSQL(Base):
    __tablename__ = "world_pulse_claim"
    claim_id = Column(String, primary_key=True)
    run_id = Column(String, nullable=False)
    topic_id = Column(String, nullable=True)
    promotion_status = Column(String, nullable=False)
    payload_json = Column(JSONB, nullable=False, default=dict)
    schema_version = Column(String, nullable=False, default="v1")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (Index("idx_world_pulse_claim_run_id", "run_id"),)


class WorldPulseEventSQL(Base):
    __tablename__ = "world_pulse_event"
    event_id = Column(String, primary_key=True)
    run_id = Column(String, nullable=False)
    event_type = Column(String, nullable=False)
    payload_json = Column(JSONB, nullable=False, default=dict)
    schema_version = Column(String, nullable=False, default="v1")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class WorldPulseEntitySQL(Base):
    __tablename__ = "world_pulse_entity"
    entity_id = Column(String, primary_key=True)
    canonical_name = Column(String, nullable=False)
    entity_type = Column(String, nullable=False)
    payload_json = Column(JSONB, nullable=False, default=dict)
    schema_version = Column(String, nullable=False, default="v1")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class WorldPulseSituationBriefSQL(Base):
    __tablename__ = "world_pulse_situation_brief"
    topic_id = Column(String, primary_key=True)
    run_id = Column(String, nullable=False)
    title = Column(String, nullable=False)
    status = Column(String, nullable=False)
    tracking_status = Column(String, nullable=False)
    current_assessment = Column(Text, nullable=False)
    payload_json = Column(JSONB, nullable=False, default=dict)
    schema_version = Column(String, nullable=False, default="v1")
    created_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (Index("idx_world_pulse_situation_brief_run_id", "run_id"),)


class WorldPulseSituationChangeSQL(Base):
    __tablename__ = "world_pulse_situation_change"
    change_id = Column(String, primary_key=True)
    topic_id = Column(String, nullable=False)
    run_id = Column(String, nullable=False)
    change_type = Column(String, nullable=False)
    payload_json = Column(JSONB, nullable=False, default=dict)
    schema_version = Column(String, nullable=False, default="v1")
    created_at = Column(DateTime(timezone=True), nullable=False)


class WorldPulseLearningDeltaSQL(Base):
    __tablename__ = "world_pulse_learning_delta"
    learning_id = Column(String, primary_key=True)
    run_id = Column(String, nullable=False)
    topic_id = Column(String, nullable=False)
    category = Column(String, nullable=False)
    payload_json = Column(JSONB, nullable=False, default=dict)
    schema_version = Column(String, nullable=False, default="v1")
    created_at = Column(DateTime(timezone=True), nullable=False)


class WorldPulseWorthReadingSQL(Base):
    __tablename__ = "world_pulse_worth_reading"
    reading_id = Column(String, primary_key=True)
    run_id = Column(String, nullable=False)
    category = Column(String, nullable=False)
    payload_json = Column(JSONB, nullable=False, default=dict)
    schema_version = Column(String, nullable=False, default="v1")
    created_at = Column(DateTime(timezone=True), nullable=False)


class WorldPulseWorthWatchingSQL(Base):
    __tablename__ = "world_pulse_worth_watching"
    watch_id = Column(String, primary_key=True)
    run_id = Column(String, nullable=False)
    category = Column(String, nullable=False)
    payload_json = Column(JSONB, nullable=False, default=dict)
    schema_version = Column(String, nullable=False, default="v1")
    created_at = Column(DateTime(timezone=True), nullable=False)


class WorldPulseContextCapsuleSQL(Base):
    __tablename__ = "world_pulse_context_capsule"
    capsule_id = Column(String, primary_key=True)
    run_id = Column(String, nullable=False)
    date = Column(String, nullable=False)
    payload_json = Column(JSONB, nullable=False, default=dict)
    schema_version = Column(String, nullable=False, default="v1")
    created_at = Column(DateTime(timezone=True), nullable=False)


class WorldPulsePublishStatusSQL(Base):
    __tablename__ = "world_pulse_publish_status"
    status_id = Column(String, primary_key=True)
    run_id = Column(String, nullable=False)
    channel = Column(String, nullable=False)
    state = Column(String, nullable=False)
    detail = Column(Text, nullable=True)
    payload_json = Column(JSONB, nullable=False, default=dict)
    schema_version = Column(String, nullable=False, default="v1")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class WorldPulseHubMessageSQL(Base):
    __tablename__ = "world_pulse_hub_message"
    message_id = Column(String, primary_key=True)
    run_id = Column(String, nullable=False)
    title = Column(String, nullable=False)
    date = Column(String, nullable=False)
    executive_summary = Column(Text, nullable=False)
    payload_json = Column(JSONB, nullable=False, default=dict)
    schema_version = Column(String, nullable=False, default="v1")
    created_at = Column(DateTime(timezone=True), nullable=False)
