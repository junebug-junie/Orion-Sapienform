import json
from datetime import datetime
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, Request
from sqlalchemy import func, and_, or_
from sqlalchemy.orm import Session, aliased

from app.db import get_session, remove_session
from app.models.notify_models import (
    NotificationRequestDB,
    NotificationReceiptDB,
    RecipientProfileDB,
    NotificationPreferenceDB,
    NotificationAttemptDB,
)
from orion.schemas.notify import (
    ChatAttentionState,
    ChatMessageState,
    NotificationRecord,
    DeliveryAttempt,
    RecipientProfile,
    NotificationPreference,
    PreferenceResolutionResponse,
    PreferenceResolutionRequest,
)

router = APIRouter()

# Schema Conversion Helpers (copied/adapted from notify)
def _attention_to_schema(row: NotificationRequestDB) -> ChatAttentionState:
    context = row.context or {}
    reason = context.get("reason") or row.title
    status = "pending" if row.attention_require_ack and row.attention_acked_at is None else "acked"
    return ChatAttentionState(
        attention_id=row.attention_id,
        notification_id=row.notification_id,
        created_at=row.created_at,
        source_service=row.source_service,
        reason=reason,
        severity=row.severity,
        message=row.body_text or "",
        context=context,
        correlation_id=row.correlation_id,
        session_id=row.session_id,
        expires_at=row.attention_expires_at,
        require_ack=row.attention_require_ack,
        ack_deadline_minutes=row.attention_ack_deadline_minutes,
        acked_at=row.attention_acked_at,
        ack_type=row.attention_ack_type,
        ack_actor=row.attention_ack_actor,
        ack_note=row.attention_ack_note,
        escalated_at=row.attention_escalated_at,
        status=status,
    )

def _chat_message_to_schema(row: NotificationRequestDB, receipts: List[NotificationReceiptDB] = None) -> ChatMessageState:
    # Merge receipt data into state
    first_seen_at = row.message_first_seen_at
    opened_at = row.message_opened_at
    dismissed_at = row.message_dismissed_at

    if receipts:
        for r in receipts:
            if r.receipt_type == "seen" and (not first_seen_at or r.received_at < first_seen_at):
                first_seen_at = r.received_at
            elif r.receipt_type == "opened":
                if not opened_at or r.received_at < opened_at:
                    opened_at = r.received_at
                # Opened implies seen
                if not first_seen_at or r.received_at < first_seen_at:
                    first_seen_at = r.received_at
            elif r.receipt_type == "dismissed":
                if not dismissed_at or r.received_at < dismissed_at:
                    dismissed_at = r.received_at
                # Dismissed implies seen
                if not first_seen_at or r.received_at < first_seen_at:
                    first_seen_at = r.received_at

    status = "unread"
    if opened_at or dismissed_at or first_seen_at:
        status = "seen"

    return ChatMessageState(
        message_id=row.message_id,
        notification_id=row.notification_id,
        created_at=row.created_at,
        source_service=row.source_service,
        session_id=row.message_session_id or row.session_id or "",
        correlation_id=row.correlation_id,
        title=row.title,
        preview_text=row.message_preview_text or row.body_text or "",
        full_text=row.message_full_text or row.body_md,
        tags=row.tags or [],
        severity=row.severity,
        require_read_receipt=row.message_require_read_receipt,
        expires_at=row.message_expires_at,
        first_seen_at=first_seen_at,
        opened_at=opened_at,
        dismissed_at=dismissed_at,
        escalated_at=row.message_escalated_at,
        status=status,
    )

def _record_to_schema(row: NotificationRequestDB) -> NotificationRecord:
    return NotificationRecord(
        notification_id=row.notification_id,
        source_service=row.source_service,
        event_kind=row.event_kind,
        severity=row.severity,
        title=row.title,
        body_text=row.body_text,
        body_md=row.body_md,
        context=row.context or {},
        tags=row.tags or [],
        recipient_group=row.recipient_group,
        channels_requested=row.channels_requested,
        dedupe_key=row.dedupe_key,
        dedupe_window_seconds=row.dedupe_window_seconds,
        ttl_seconds=row.ttl_seconds,
        correlation_id=row.correlation_id,
        session_id=row.session_id,
        created_at=row.created_at,
        status=row.status,
        policy_action=row.policy_action,
        drop_reason=row.drop_reason,
    )

def _profile_to_schema(row: RecipientProfileDB) -> RecipientProfile:
    return RecipientProfile(
        recipient_group=row.recipient_group,
        display_name=row.display_name,
        timezone=row.timezone,
        quiet_hours_enabled=bool(row.quiet_hours_enabled),
        quiet_start_local=row.quiet_start_local,
        quiet_end_local=row.quiet_end_local,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )

def _preference_to_schema(row: NotificationPreferenceDB) -> NotificationPreference:
    return NotificationPreference(
        id=row.id,
        recipient_group=row.recipient_group,
        scope_type=row.scope_type,
        scope_value=row.scope_value,
        channels_enabled=_channels_from_json(row.channels_enabled),
        escalation_enabled=_int_to_bool(row.escalation_enabled),
        escalation_delay_minutes=row.escalation_delay_minutes,
        throttle_max_per_window=row.throttle_max_per_window,
        throttle_window_seconds=row.throttle_window_seconds,
        dedupe_window_seconds=row.dedupe_window_seconds,
        updated_at=row.updated_at,
    )

def _channels_from_json(raw):
    if not raw: return []
    try:
        return json.loads(raw)
    except:
        return []

def _int_to_bool(val):
    return bool(val) if val is not None else None

# Endpoints

@router.get("/attention")
async def list_attention(limit: int = 50, status: Optional[str] = None):
    db = get_session()
    try:
        query = db.query(NotificationRequestDB).filter(NotificationRequestDB.attention_id.isnot(None))
        if status == "pending":
            query = query.filter(
                NotificationRequestDB.attention_require_ack.is_(True),
                NotificationRequestDB.attention_acked_at.is_(None),
            )
        elif status == "acked":
            query = query.filter(NotificationRequestDB.attention_acked_at.isnot(None))
        rows = query.order_by(NotificationRequestDB.created_at.desc()).limit(limit).all()
        return [_attention_to_schema(row) for row in rows]
    finally:
        remove_session()

@router.get("/recipients")
async def list_recipients():
    db = get_session()
    try:
        rows = db.query(RecipientProfileDB).order_by(RecipientProfileDB.recipient_group.asc()).all()
        return [_profile_to_schema(row) for row in rows]
    finally:
        remove_session()

@router.get("/recipients/{recipient_group}")
async def get_recipient(recipient_group: str):
    db = get_session()
    try:
        row = db.query(RecipientProfileDB).filter_by(recipient_group=recipient_group).first()
        if not row:
            raise HTTPException(status_code=404, detail="Recipient group not found")
        return _profile_to_schema(row)
    finally:
        remove_session()

@router.get("/recipients/{recipient_group}/preferences")
async def list_preferences(recipient_group: str):
    db = get_session()
    try:
        rows = (
            db.query(NotificationPreferenceDB)
            .filter(NotificationPreferenceDB.recipient_group == recipient_group)
            .order_by(NotificationPreferenceDB.scope_type.asc(), NotificationPreferenceDB.scope_value.asc())
            .all()
        )
        return [_preference_to_schema(row) for row in rows]
    finally:
        remove_session()

@router.get("/chat/messages")
async def list_chat_messages(limit: int = 50, status: Optional[str] = None, session_id: Optional[str] = None):
    db = get_session()
    try:
        # Base query for messages
        query = db.query(NotificationRequestDB).filter(NotificationRequestDB.message_id.isnot(None))

        if session_id:
            query = query.filter(NotificationRequestDB.message_session_id == session_id)

        # Join with receipts to determine status
        # We need to filter based on EXISTENCE of receipt if status is specified.
        # But we also need to fetch receipts to populate the schema timestamps.
        # Efficient approach: Fetch messages then fetch receipts for those messages.
        # OR: Join.
        # Let's filter first.

        if status == "unread":
            # No receipts and legacy columns null
            query = query.filter(
                NotificationRequestDB.message_first_seen_at.is_(None),
                NotificationRequestDB.message_opened_at.is_(None),
                NotificationRequestDB.message_dismissed_at.is_(None),
                ~db.query(NotificationReceiptDB).filter(
                    NotificationReceiptDB.message_id == NotificationRequestDB.message_id
                ).exists()
            )
        elif status == "seen":
            # Receipts exist OR legacy columns not null
            query = query.filter(
                or_(
                    NotificationRequestDB.message_first_seen_at.isnot(None),
                    NotificationRequestDB.message_opened_at.isnot(None),
                    NotificationRequestDB.message_dismissed_at.isnot(None),
                    db.query(NotificationReceiptDB).filter(
                        NotificationReceiptDB.message_id == NotificationRequestDB.message_id
                    ).exists()
                )
            )

        rows = query.order_by(NotificationRequestDB.created_at.desc()).limit(limit).all()

        # Now fetch receipts for these rows to populate timestamps
        if not rows:
            return []

        msg_ids = [r.message_id for r in rows if r.message_id]
        receipts_map = {}
        if msg_ids:
            receipts = db.query(NotificationReceiptDB).filter(NotificationReceiptDB.message_id.in_(msg_ids)).all()
            for r in receipts:
                if r.message_id not in receipts_map:
                    receipts_map[r.message_id] = []
                receipts_map[r.message_id].append(r)

        return [_chat_message_to_schema(row, receipts_map.get(row.message_id)) for row in rows]
    finally:
        remove_session()

@router.get("/notifications")
async def list_notifications(
    limit: int = 50,
    since: Optional[str] = None,
    severity: Optional[str] = None,
    event_kind: Optional[str] = None
):
    db = get_session()
    try:
        query = db.query(NotificationRequestDB)
        if since:
            try:
                since_dt = datetime.fromisoformat(since)
                query = query.filter(NotificationRequestDB.created_at >= since_dt)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid since timestamp")
        if severity:
            query = query.filter(NotificationRequestDB.severity == severity)
        if event_kind:
            query = query.filter(NotificationRequestDB.event_kind == event_kind)
        rows = query.order_by(NotificationRequestDB.created_at.desc()).limit(limit).all()
        return [_record_to_schema(row) for row in rows]
    finally:
        remove_session()

@router.get("/notifications/{notification_id}")
async def get_notification(notification_id: str):
    db = get_session()
    try:
        row = db.query(NotificationRequestDB).filter_by(notification_id=notification_id).first()
        if not row:
            raise HTTPException(status_code=404, detail="Notification not found")
        return _record_to_schema(row)
    finally:
        remove_session()

@router.post("/preferences/resolve")
async def resolve_preferences(payload: PreferenceResolutionRequest):
    # Minimal resolution logic (DB based only, ignoring static policy for now as per stateless req)
    db = get_session()
    try:
        recipient_profile = db.query(RecipientProfileDB).filter_by(recipient_group=payload.recipient_group).first()
        preferences = db.query(NotificationPreferenceDB).filter(NotificationPreferenceDB.recipient_group == payload.recipient_group).all()

        # Simplified resolution (reimplementing minimal logic)
        channels = ["in_app"] # Default fallback

        # Override logic
        pref_event = None
        pref_sev = None
        for p in preferences:
            if p.scope_type == "event_kind" and p.scope_value == payload.event_kind:
                pref_event = p
            if p.scope_type == "severity" and p.scope_value == payload.severity.lower():
                pref_sev = p

        pref = pref_event or pref_sev
        breakdown = {"source": "default"}

        if pref:
            if pref.channels_enabled:
                channels = _channels_from_json(pref.channels_enabled)
                breakdown["source"] = f"preference:{pref.scope_type}"

        return PreferenceResolutionResponse(
            channels_final=channels,
            quiet_hours_applied=False, # TODO: Implement quiet hours logic if needed
            source_breakdown=breakdown
        )
    finally:
        remove_session()
