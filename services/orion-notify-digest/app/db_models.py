from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from sqlalchemy import Column, DateTime, Index, Integer, JSON, String, Text

from orion.core.sql_router.db import Base


class NotificationRequestDB(Base):
    __tablename__ = "notify_requests"

    notification_id = Column(String, primary_key=True)
    source_service = Column(String, nullable=False)
    event_kind = Column(String, nullable=False)
    severity = Column(String, nullable=False)
    title = Column(String, nullable=False)
    body_text = Column(Text, nullable=True)
    body_md = Column(Text, nullable=True)
    context = Column(JSON, nullable=False, default=dict)
    tags = Column(JSON, nullable=False, default=list)
    recipient_group = Column(String, nullable=False)
    channels_requested = Column(JSON, nullable=True)
    dedupe_key = Column(String, nullable=True)
    dedupe_window_seconds = Column(Integer, nullable=True)
    ttl_seconds = Column(Integer, nullable=True)
    correlation_id = Column(String, nullable=True)
    session_id = Column(String, nullable=True)
    created_at = Column(DateTime, nullable=False)
    status = Column(String, nullable=False)
    policy_action = Column(String, nullable=True)
    drop_reason = Column(String, nullable=True)

    __table_args__ = (
        Index("idx_notify_requests_created_at", "created_at"),
        Index("idx_notify_requests_event_kind", "event_kind"),
        Index("idx_notify_requests_severity", "severity"),
    )


class NotificationAttemptDB(Base):
    __tablename__ = "notify_attempts"

    attempt_id = Column(String, primary_key=True)
    notification_id = Column(String, nullable=False)
    channel = Column(String, nullable=False)
    status = Column(String, nullable=False)
    provider_message_id = Column(String, nullable=True)
    error = Column(Text, nullable=True)
    attempted_at = Column(DateTime, nullable=False)

    __table_args__ = (
        Index("idx_notify_attempts_created_at", "attempted_at"),
        Index("idx_notify_attempts_notification_id", "notification_id"),
    )


class DigestRunDB(Base):
    __tablename__ = "notify_digest_runs"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    kind = Column(String, nullable=False)
    window_start = Column(DateTime, nullable=False)
    window_end = Column(DateTime, nullable=False)
    status = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (Index("idx_notify_digest_runs_created_at", "created_at"),)
