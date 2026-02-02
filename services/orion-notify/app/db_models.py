from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from sqlalchemy import Boolean, Column, DateTime, Index, Integer, JSON, String, Text, ForeignKey

from orion.core.sql_router.db import Base


class NotificationRequestDB(Base):
    __tablename__ = "notify_requests"

    notification_id = Column(String, primary_key=True, default=lambda: str(uuid4()))
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
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    status = Column(String, nullable=False)
    policy_action = Column(String, nullable=True)
    drop_reason = Column(String, nullable=True)
    attention_id = Column(String, nullable=True)
    attention_require_ack = Column(Boolean, nullable=False, default=False)
    attention_ack_deadline_minutes = Column(Integer, nullable=True)
    attention_acked_at = Column(DateTime, nullable=True)
    attention_ack_type = Column(String, nullable=True)
    attention_ack_actor = Column(String, nullable=True)
    attention_ack_note = Column(Text, nullable=True)
    attention_escalated_at = Column(DateTime, nullable=True)
    attention_escalation_channels = Column(JSON, nullable=True)
    attention_expires_at = Column(DateTime, nullable=True)
    message_id = Column(String, nullable=True)
    message_session_id = Column(String, nullable=True)
    message_preview_text = Column(Text, nullable=True)
    message_full_text = Column(Text, nullable=True)
    message_require_read_receipt = Column(Boolean, nullable=False, default=False)
    message_first_seen_at = Column(DateTime, nullable=True)
    message_opened_at = Column(DateTime, nullable=True)
    message_dismissed_at = Column(DateTime, nullable=True)
    message_escalated_at = Column(DateTime, nullable=True)
    message_expires_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("idx_notify_requests_created_at", "created_at"),
        Index("idx_notify_requests_severity", "severity"),
        Index("idx_notify_requests_event_kind", "event_kind"),
        Index("idx_notify_requests_dedupe_key", "dedupe_key"),
        Index("idx_notify_requests_correlation_id", "correlation_id"),
        Index("idx_notify_requests_session_id", "session_id"),
        Index("idx_notify_requests_attention_id", "attention_id"),
        Index("idx_notify_requests_attention_acked_at", "attention_acked_at"),
        Index("idx_notify_requests_attention_escalated_at", "attention_escalated_at"),
        Index("idx_notify_requests_message_id", "message_id"),
        Index("idx_notify_requests_message_session_id", "message_session_id"),
        Index("idx_notify_requests_message_opened_at", "message_opened_at"),
    )


class NotificationAttemptDB(Base):
    __tablename__ = "notify_attempts"

    attempt_id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    notification_id = Column(String, nullable=False)
    channel = Column(String, nullable=False)
    status = Column(String, nullable=False)
    provider_message_id = Column(String, nullable=True)
    error = Column(Text, nullable=True)
    attempted_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_notify_attempts_notification_id", "notification_id"),
        Index("idx_notify_attempts_created_at", "attempted_at"),
    )


class RecipientProfileDB(Base):
    __tablename__ = "notify_recipient_profiles"

    recipient_group = Column(String, primary_key=True)
    display_name = Column(String, nullable=True)
    timezone = Column(String, nullable=False, default="America/Denver")
    quiet_hours_enabled = Column(Integer, nullable=False, default=0)
    quiet_start_local = Column(String, nullable=False, default="22:00")
    quiet_end_local = Column(String, nullable=False, default="07:00")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_notify_recipient_profiles_group", "recipient_group"),
    )


class NotificationPreferenceDB(Base):
    __tablename__ = "notify_preferences"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    recipient_group = Column(String, ForeignKey("notify_recipient_profiles.recipient_group"), nullable=False)
    scope_type = Column(String, nullable=False)
    scope_value = Column(String, nullable=False)
    channels_enabled = Column(Text, nullable=False, default="[]")
    escalation_enabled = Column(Integer, nullable=True)
    escalation_delay_minutes = Column(Integer, nullable=True)
    throttle_max_per_window = Column(Integer, nullable=True)
    throttle_window_seconds = Column(Integer, nullable=True)
    dedupe_window_seconds = Column(Integer, nullable=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_notify_preferences_recipient", "recipient_group"),
        Index("idx_notify_preferences_recipient_scope", "recipient_group", "scope_type", "scope_value"),
    )
