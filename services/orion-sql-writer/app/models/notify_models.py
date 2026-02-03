from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from sqlalchemy import Boolean, Column, DateTime, Index, Integer, JSON, String, Text, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB

from app.db import Base


class NotificationRequestDB(Base):
    __tablename__ = "notify_requests"

    notification_id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    source_service = Column(String, nullable=False)
    event_kind = Column(String, nullable=False)
    severity = Column(String, nullable=False)
    title = Column(String, nullable=False)
    body_text = Column(Text, nullable=True)
    body_md = Column(Text, nullable=True)
    context = Column(JSONB, nullable=False, default=dict)
    tags = Column(JSONB, nullable=False, default=list)
    recipient_group = Column(String, nullable=False)
    channels_requested = Column(JSONB, nullable=True)
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
    attention_escalation_channels = Column(JSONB, nullable=True)
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
        # Unique constraints for correctness
        UniqueConstraint("message_id", name="uq_notify_requests_message_id"),
        UniqueConstraint("attention_id", name="uq_notify_requests_attention_id"),
    )


class NotificationReceiptDB(Base):
    __tablename__ = "notify_receipts"

    receipt_id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    message_id = Column(String, nullable=False)
    receipt_type = Column(String, nullable=False)
    session_id = Column(String, nullable=True)
    received_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_notify_receipts_message_id", "message_id"),
        UniqueConstraint("message_id", "receipt_type", name="uq_notify_receipts_message_type"),
    )
