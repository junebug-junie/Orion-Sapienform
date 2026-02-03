from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter
from sqlalchemy import or_

from app.db import get_session, remove_session
from app.models.notify_models import (
    NotificationRequestDB,
    NotificationReceiptDB,
)
from orion.schemas.notify import (
    ChatAttentionState,
    ChatMessageState,
)

router = APIRouter()

# Schema Conversion Helpers

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

@router.get("/chat/messages")
async def list_chat_messages(limit: int = 50, status: Optional[str] = None, session_id: Optional[str] = None):
    db = get_session()
    try:
        query = db.query(NotificationRequestDB).filter(NotificationRequestDB.message_id.isnot(None))

        if session_id:
            query = query.filter(NotificationRequestDB.message_session_id == session_id)

        if status == "unread":
            query = query.filter(
                NotificationRequestDB.message_first_seen_at.is_(None),
                NotificationRequestDB.message_opened_at.is_(None),
                NotificationRequestDB.message_dismissed_at.is_(None),
                ~db.query(NotificationReceiptDB).filter(
                    NotificationReceiptDB.message_id == NotificationRequestDB.message_id
                ).exists()
            )
        elif status == "seen":
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
