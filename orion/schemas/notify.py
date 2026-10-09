from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Literal
from uuid import UUID, uuid4

import re

from pydantic import BaseModel, Field, field_validator, model_validator

from orion.schemas.cortex.contracts import AgentTraceSummaryV1


class NotificationAttachment(BaseModel):
    filename: str
    content_base64: str
    mime_type: str = Field("application/octet-stream")
    # When set AND the request carries body_html, the attachment is embedded as
    # an inline (multipart/related) part referenced from the HTML as
    # `<img src="cid:{content_id}">` instead of a regular download attachment.
    # Bare id, no angle brackets. Ignored for plain-text-only requests.
    content_id: Optional[str] = None

    @field_validator("content_id")
    @classmethod
    def _normalize_content_id(cls, v: Optional[str]) -> Optional[str]:
        # Bad ids otherwise surface as a broken Content-ID header (image never
        # matches its cid: reference) or a mid-send ValueError that fails the
        # whole letter. Reject at the boundary instead (HTTP 422).
        if v is None:
            return None
        cid = v.strip()
        if cid.lower().startswith("cid:"):
            cid = cid[4:]
        cid = cid.strip().strip("<>").strip()
        if not cid:
            return None
        if not _CONTENT_ID_RE.fullmatch(cid):
            raise ValueError("content_id must match [A-Za-z0-9._@+-]{1,200}")
        return cid


_CONTENT_ID_RE = re.compile(r"[A-Za-z0-9._@+-]{1,200}")


class NotificationRequest(BaseModel):
    notification_id: UUID = Field(default_factory=uuid4)
    source_service: str
    event_kind: str
    severity: str
    title: str
    body_text: Optional[str] = None
    body_md: Optional[str] = None
    # Optional HTML rendering of the email body. Sent as a text/html
    # alternative alongside the plain-text part (body_text or body_md is the
    # fallback). Email-only: not persisted and not sent to the in-app event.
    body_html: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    recipient_group: str = Field("juniper_primary")
    channels_requested: Optional[List[str]] = None
    dedupe_key: Optional[str] = None
    dedupe_window_seconds: Optional[int] = None
    ttl_seconds: Optional[int] = None
    correlation_id: Optional[str] = None
    session_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    attachments: Optional[List[NotificationAttachment]] = None

    @model_validator(mode="after")
    def _unique_content_ids(self) -> "NotificationRequest":
        cids = [a.content_id for a in (self.attachments or []) if a.content_id]
        if len(cids) != len(set(cids)):
            raise ValueError("attachment content_id values must be unique within a request")
        return self


class NotificationAccepted(BaseModel):
    ok: bool
    notification_id: Optional[UUID] = None
    status: Optional[str] = None
    detail: Optional[str] = None
    # What actually happened to the email for this request (EmailOutcome.status:
    # "sent" | "failed" | "skipped" | "deferred"). "sent" means the SMTP server
    # accepted it, not that it reached an inbox. None on endpoints that do not
    # attempt email synchronously.
    email_status: Optional[str] = None


class NotificationRecord(BaseModel):
    notification_id: UUID
    source_service: str
    event_kind: str
    severity: str
    title: str
    body_text: Optional[str] = None
    body_md: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    recipient_group: str
    channels_requested: Optional[List[str]] = None
    dedupe_key: Optional[str] = None
    dedupe_window_seconds: Optional[int] = None
    ttl_seconds: Optional[int] = None
    correlation_id: Optional[str] = None
    session_id: Optional[str] = None
    created_at: datetime
    status: str
    policy_action: Optional[str] = None
    drop_reason: Optional[str] = None


class DeliveryAttempt(BaseModel):
    attempt_id: UUID
    notification_id: UUID
    channel: str
    status: str
    provider_message_id: Optional[str] = None
    error: Optional[str] = None
    attempted_at: datetime


class ChatAttentionRequest(BaseModel):
    attention_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    source_service: str
    reason: str
    severity: str = Field("info")
    message: str
    context: Dict[str, Any] = Field(default_factory=dict)
    correlation_id: Optional[str] = None
    session_id: Optional[str] = None
    expires_at: Optional[datetime] = None
    require_ack: bool = Field(True)


class ChatAttentionAck(BaseModel):
    attention_id: UUID
    acked_at: datetime = Field(default_factory=datetime.utcnow)
    ack_type: Literal["seen", "dismissed", "snooze"] = Field("seen")
    actor: str = Field("juniper")
    note: Optional[str] = None


class ChatAttentionState(BaseModel):
    attention_id: UUID
    notification_id: Optional[UUID] = None
    created_at: datetime
    source_service: str
    reason: str
    severity: str
    message: str
    context: Dict[str, Any] = Field(default_factory=dict)
    correlation_id: Optional[str] = None
    session_id: Optional[str] = None
    expires_at: Optional[datetime] = None
    require_ack: bool
    ack_deadline_minutes: Optional[int] = None
    acked_at: Optional[datetime] = None
    ack_type: Optional[str] = None
    ack_actor: Optional[str] = None
    ack_note: Optional[str] = None
    escalated_at: Optional[datetime] = None
    status: str = Field("pending")


class ChatMessageNotification(BaseModel):
    message_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    source_service: str
    session_id: str
    correlation_id: Optional[str] = None
    title: Optional[str] = Field(default="New message from Orion")
    preview_text: str = Field(max_length=280)
    full_text: Optional[str] = None
    agent_trace: Optional[AgentTraceSummaryV1] = None
    tags: List[str] = Field(default_factory=lambda: ["chat", "message"])
    severity: str = Field("info")
    require_read_receipt: bool = Field(True)
    expires_at: Optional[datetime] = None
    workflow: Dict[str, Any] = Field(default_factory=dict)


class ChatMessageReceipt(BaseModel):
    message_id: UUID
    session_id: str
    receipt_type: Literal["seen", "opened", "dismissed"]
    received_at: datetime = Field(default_factory=datetime.utcnow)
    actor: str = Field("juniper")


class ChatMessageState(BaseModel):
    message_id: UUID
    notification_id: Optional[UUID] = None
    created_at: datetime
    source_service: str
    session_id: str
    correlation_id: Optional[str] = None
    title: Optional[str] = None
    preview_text: str
    full_text: Optional[str] = None
    agent_trace: Optional[AgentTraceSummaryV1] = None
    tags: List[str] = Field(default_factory=list)
    severity: str
    require_read_receipt: bool
    expires_at: Optional[datetime] = None
    first_seen_at: Optional[datetime] = None
    opened_at: Optional[datetime] = None
    dismissed_at: Optional[datetime] = None
    escalated_at: Optional[datetime] = None
    status: str = Field("unread")
    workflow: Dict[str, Any] = Field(default_factory=dict)


class RecipientProfile(BaseModel):
    recipient_group: str
    display_name: Optional[str] = None
    timezone: str = Field(default="America/Denver")
    quiet_hours_enabled: bool = Field(default=False)
    quiet_start_local: str = Field(default="22:00")
    quiet_end_local: str = Field(default="07:00")
    created_at: datetime
    updated_at: datetime


class RecipientProfileUpdate(BaseModel):
    recipient_group: Optional[str] = None
    display_name: Optional[str] = None
    timezone: Optional[str] = None
    quiet_hours_enabled: Optional[bool] = None
    quiet_start_local: Optional[str] = None
    quiet_end_local: Optional[str] = None


class NotificationPreference(BaseModel):
    id: Optional[str] = None
    recipient_group: str
    scope_type: Literal["severity", "event_kind"]
    scope_value: str
    channels_enabled: List[str] = Field(default_factory=list)
    escalation_enabled: Optional[bool] = None
    escalation_delay_minutes: Optional[int] = None
    throttle_max_per_window: Optional[int] = None
    throttle_window_seconds: Optional[int] = None
    dedupe_window_seconds: Optional[int] = None
    updated_at: Optional[datetime] = None


class NotificationPreferencesUpdate(BaseModel):
    preferences: List[NotificationPreference] = Field(default_factory=list)


class PreferenceResolutionRequest(BaseModel):
    recipient_group: str
    event_kind: str
    severity: str


class PreferenceResolutionResponse(BaseModel):
    channels_final: List[str]
    quiet_hours_applied: bool
    escalation_enabled: Optional[bool] = None
    escalation_delay_minutes: Optional[int] = None
    throttle_max_per_window: Optional[int] = None
    throttle_window_seconds: Optional[int] = None
    dedupe_window_seconds: Optional[int] = None
    source_breakdown: Dict[str, Any] = Field(default_factory=dict)


class HubNotificationEvent(BaseModel):
    notification_id: UUID
    created_at: datetime
    severity: str
    event_kind: str
    source_service: str
    title: str
    body_text: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    correlation_id: Optional[str] = None
    session_id: Optional[str] = None
    status: Optional[str] = None
    attention_id: Optional[UUID] = None
    message_id: Optional[UUID] = None
    agent_trace: Optional[AgentTraceSummaryV1] = None
    notification_type: Optional[str] = None
    silent: Optional[bool] = None
    workflow: Dict[str, Any] = Field(default_factory=dict)


class NotificationReceiptEvent(BaseModel):
    receipt_id: UUID = Field(default_factory=uuid4)
    message_id: UUID
    receipt_type: str
    session_id: Optional[str] = None
    received_at: datetime = Field(default_factory=datetime.utcnow)
    created_at: datetime = Field(default_factory=datetime.utcnow)
