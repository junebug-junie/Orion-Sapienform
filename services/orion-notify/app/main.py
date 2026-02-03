import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import requests
from fastapi import FastAPI, Header, HTTPException, Query, Request
from sqlalchemy import func
from sqlalchemy.orm import Session

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.sql_router.db import SessionLocal, init_models
from orion.notify.transport import EmailTransport
from orion.schemas.notify import (
    ChatAttentionAck,
    ChatAttentionRequest,
    ChatAttentionState,
    ChatMessageNotification,
    ChatMessageReceipt,
    ChatMessageState,
    DeliveryAttempt,
    HubNotificationEvent,
    NotificationAccepted,
    NotificationPreference,
    NotificationPreferencesUpdate,
    NotificationRecord,
    NotificationReceiptEvent,
    NotificationRequest,
    PreferenceResolutionRequest,
    PreferenceResolutionResponse,
    RecipientProfile,
    RecipientProfileUpdate,
)

from .db_models import (
    NotificationAttemptDB,
    NotificationPreferenceDB,
    NotificationRequestDB,
    RecipientProfileDB,
    NotificationReceiptDB,
)
from .policy import Policy, PolicyDecision, ThrottleRule
from .settings import settings

logger = logging.getLogger("orion-notify")

app = FastAPI(
    title="Orion Notify",
    version=settings.SERVICE_VERSION,
)

ATTENTION_EVENT_KIND = "orion.chat.attention"
ATTENTION_ESCALATION_EVENT_KIND = "orion.chat.attention.escalation"
CHAT_MESSAGE_EVENT_KIND = "orion.chat.message"
CHAT_MESSAGE_ESCALATION_EVENT_KIND = "orion.chat.message.escalation"

THROTTLE_WINDOW_MIN_SECONDS = 10
THROTTLE_WINDOW_MAX_SECONDS = 86400
THROTTLE_MAX_MIN = 1
THROTTLE_MAX_MAX = 1000
DEDUPE_WINDOW_MIN_SECONDS = 0
DEDUPE_WINDOW_MAX_SECONDS = 86400


@app.on_event("startup")
async def on_startup() -> None:
    init_models([NotificationRequestDB, NotificationAttemptDB, RecipientProfileDB, NotificationPreferenceDB, NotificationReceiptDB])
    app.state.transport = EmailTransport(
        smtp_host=settings.NOTIFY_EMAIL_SMTP_HOST,
        smtp_port=settings.NOTIFY_EMAIL_SMTP_PORT,
        smtp_username=settings.NOTIFY_EMAIL_SMTP_USERNAME,
        smtp_password=settings.NOTIFY_EMAIL_SMTP_PASSWORD,
        use_tls=settings.NOTIFY_EMAIL_USE_TLS,
        default_from=settings.NOTIFY_EMAIL_FROM,
        default_to=settings.notify_email_to,
    )
    app.state.policy = Policy.load(settings.POLICY_RULES_PATH)
    app.state.env_lookup = dict(os.environ)
    app.state.bus = await _init_bus()
    if settings.NOTIFY_ESCALATION_POLL_SECONDS > 0:
        app.state.escalation_task = asyncio.create_task(_run_escalation_loop(app))
    _ensure_bootstrap_defaults()


@app.on_event("shutdown")
async def on_shutdown() -> None:
    bus = getattr(app.state, "bus", None)
    if bus is not None:
        try:
            await bus.close()
        except Exception as exc:
            logger.warning("Failed to close Orion bus: %s", exc)
    task = getattr(app.state, "escalation_task", None)
    if task is not None and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


@app.get("/health")
async def health() -> dict:
    return {
        "ok": True,
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "smtp_host_configured": bool(settings.NOTIFY_EMAIL_SMTP_HOST),
    }


def _check_token(token: Optional[str]) -> None:
    if settings.API_TOKEN and token != settings.API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid API token")


def _get_transport(request: Request) -> EmailTransport:
    return request.app.state.transport  # type: ignore[return-value]


@app.post("/notify")
async def notify(
    payload: NotificationRequest,
    request: Request,
    x_orion_notify_token: Optional[str] = Header(default=None, alias="X-Orion-Notify-Token"),
) -> NotificationAccepted:
    _check_token(x_orion_notify_token)
    transport = _get_transport(request)
    policy: Policy = request.app.state.policy
    env_lookup = request.app.state.env_lookup
    bus: OrionBusAsync | None = request.app.state.bus

    accepted, _ = await _process_notification(
        payload=payload,
        transport=transport,
        policy=policy,
        env_lookup=env_lookup,
        bus=bus,
    )
    return accepted


@app.post("/attention/request")
async def attention_request(
    payload: ChatAttentionRequest,
    request: Request,
    x_orion_notify_token: Optional[str] = Header(default=None, alias="X-Orion-Notify-Token"),
) -> ChatAttentionState:
    _check_token(x_orion_notify_token)
    notify_payload = _attention_request_to_notification(payload)
    transport = _get_transport(request)
    policy: Policy = request.app.state.policy
    env_lookup = request.app.state.env_lookup
    bus: OrionBusAsync | None = request.app.state.bus
    accepted, record = await _process_notification(
        payload=notify_payload,
        transport=transport,
        policy=policy,
        env_lookup=env_lookup,
        bus=bus,
    )
    if not accepted.ok:
        raise HTTPException(status_code=500, detail=accepted.detail or "Failed to store attention request")
    if not record:
        raise HTTPException(status_code=500, detail="Attention record not stored")
    return _attention_to_schema(record)


@app.post("/attention/{attention_id}/ack")
async def attention_ack(
    attention_id: str,
    payload: ChatAttentionAck,
    x_orion_notify_token: Optional[str] = Header(default=None, alias="X-Orion-Notify-Token"),
) -> ChatAttentionState:
    _check_token(x_orion_notify_token)
    if str(payload.attention_id) != attention_id:
        raise HTTPException(status_code=400, detail="Attention ID mismatch")
    with SessionLocal() as db:
        record = _apply_attention_ack(db, payload)
        if not record:
            raise HTTPException(status_code=404, detail="Attention request not found")
        return _attention_to_schema(record)


@app.get("/attention")
async def list_attention(
    limit: int = Query(default=50, ge=1, le=200),
    status: Optional[str] = None,
    x_orion_notify_token: Optional[str] = Header(default=None, alias="X-Orion-Notify-Token"),
) -> List[ChatAttentionState]:
    _check_token(x_orion_notify_token)
    with SessionLocal() as db:
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


@app.get("/recipients")
async def list_recipients(
    x_orion_notify_token: Optional[str] = Header(default=None, alias="X-Orion-Notify-Token"),
) -> List[RecipientProfile]:
    _check_token(x_orion_notify_token)
    with SessionLocal() as db:
        rows = db.query(RecipientProfileDB).order_by(RecipientProfileDB.recipient_group.asc()).all()
        return [_profile_to_schema(row) for row in rows]


@app.get("/recipients/{recipient_group}")
async def get_recipient(
    recipient_group: str,
    x_orion_notify_token: Optional[str] = Header(default=None, alias="X-Orion-Notify-Token"),
) -> RecipientProfile:
    _check_token(x_orion_notify_token)
    with SessionLocal() as db:
        row = db.query(RecipientProfileDB).filter_by(recipient_group=recipient_group).first()
        if not row:
            raise HTTPException(status_code=404, detail="Recipient group not found")
        return _profile_to_schema(row)


@app.put("/recipients/{recipient_group}")
async def update_recipient(
    recipient_group: str,
    payload: RecipientProfileUpdate,
    x_orion_notify_token: Optional[str] = Header(default=None, alias="X-Orion-Notify-Token"),
    ) -> RecipientProfile:
    _check_token(x_orion_notify_token)
    with SessionLocal() as db:
        row = db.query(RecipientProfileDB).filter_by(recipient_group=recipient_group).first()
        if not row:
            row = RecipientProfileDB(recipient_group=recipient_group)
            db.add(row)
        if payload.display_name is not None:
            row.display_name = payload.display_name
        if payload.timezone is not None:
            row.timezone = payload.timezone
        if payload.quiet_hours_enabled is not None:
            row.quiet_hours_enabled = 1 if payload.quiet_hours_enabled else 0
        if payload.quiet_start_local is not None:
            row.quiet_start_local = payload.quiet_start_local
        if payload.quiet_end_local is not None:
            row.quiet_end_local = payload.quiet_end_local
        row.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(row)
        return _profile_to_schema(row)


@app.get("/recipients/{recipient_group}/preferences")
async def list_preferences(
    recipient_group: str,
    x_orion_notify_token: Optional[str] = Header(default=None, alias="X-Orion-Notify-Token"),
) -> List[NotificationPreference]:
    _check_token(x_orion_notify_token)
    with SessionLocal() as db:
        rows = (
            db.query(NotificationPreferenceDB)
            .filter(NotificationPreferenceDB.recipient_group == recipient_group)
            .order_by(NotificationPreferenceDB.scope_type.asc(), NotificationPreferenceDB.scope_value.asc())
            .all()
        )
        return [_preference_to_schema(row) for row in rows]


@app.put("/recipients/{recipient_group}/preferences")
async def update_preferences(
    recipient_group: str,
    payload: NotificationPreferencesUpdate,
    x_orion_notify_token: Optional[str] = Header(default=None, alias="X-Orion-Notify-Token"),
) -> List[NotificationPreference]:
    _check_token(x_orion_notify_token)
    _validate_preferences(payload.preferences)
    with SessionLocal() as db:
        profile = db.query(RecipientProfileDB).filter_by(recipient_group=recipient_group).first()
        if not profile:
            profile = RecipientProfileDB(recipient_group=recipient_group)
            db.add(profile)
            db.commit()
            db.refresh(profile)
        updated_rows: List[NotificationPreferenceDB] = []
        for pref in payload.preferences:
            if pref.recipient_group and pref.recipient_group != recipient_group:
                raise HTTPException(status_code=400, detail="Recipient group mismatch in preference payload")
            row = None
            if pref.id:
                row = db.query(NotificationPreferenceDB).filter_by(id=pref.id, recipient_group=recipient_group).first()
            if not row:
                row = (
                    db.query(NotificationPreferenceDB)
                    .filter_by(
                        recipient_group=recipient_group,
                        scope_type=pref.scope_type,
                        scope_value=pref.scope_value,
                    )
                    .first()
                )
            if not row:
                row = NotificationPreferenceDB(
                    id=pref.id or str(uuid4()),
                    recipient_group=recipient_group,
                    scope_type=pref.scope_type,
                    scope_value=pref.scope_value,
                )
                db.add(row)
            row.channels_enabled = _channels_to_json(pref.channels_enabled)
            row.escalation_enabled = (
                None if pref.escalation_enabled is None else (1 if pref.escalation_enabled else 0)
            )
            row.escalation_delay_minutes = pref.escalation_delay_minutes
            row.throttle_max_per_window = pref.throttle_max_per_window
            row.throttle_window_seconds = pref.throttle_window_seconds
            row.dedupe_window_seconds = pref.dedupe_window_seconds
            row.updated_at = datetime.utcnow()
            updated_rows.append(row)
        db.commit()
        return [_preference_to_schema(row) for row in updated_rows]


@app.delete("/recipients/{recipient_group}/preferences/{pref_id}")
async def delete_preference(
    recipient_group: str,
    pref_id: str,
    x_orion_notify_token: Optional[str] = Header(default=None, alias="X-Orion-Notify-Token"),
) -> dict:
    _check_token(x_orion_notify_token)
    with SessionLocal() as db:
        row = db.query(NotificationPreferenceDB).filter_by(id=pref_id, recipient_group=recipient_group).first()
        if not row:
            raise HTTPException(status_code=404, detail="Preference not found")
        db.delete(row)
        db.commit()
        return {"ok": True}


@app.post("/preferences/resolve")
async def resolve_preferences(
    payload: PreferenceResolutionRequest,
    x_orion_notify_token: Optional[str] = Header(default=None, alias="X-Orion-Notify-Token"),
) -> PreferenceResolutionResponse:
    _check_token(x_orion_notify_token)
    with SessionLocal() as db:
        profile = _get_recipient_profile(db, payload.recipient_group)
        preferences = _get_preferences(db, payload.recipient_group)
    policy: Policy = app.state.policy
    decision, breakdown = _resolve_policy_with_preferences(
        policy=policy,
        recipient_profile=profile,
        preferences=preferences,
        event_kind=payload.event_kind,
        severity=payload.severity,
    )
    return PreferenceResolutionResponse(
        channels_final=decision.channels,
        quiet_hours_applied=breakdown.get("quiet_hours_applied", False),
        escalation_enabled=breakdown.get("escalation_enabled"),
        escalation_delay_minutes=breakdown.get("escalation_delay_minutes"),
        throttle_max_per_window=breakdown.get("throttle_max_per_window"),
        throttle_window_seconds=breakdown.get("throttle_window_seconds"),
        dedupe_window_seconds=breakdown.get("dedupe_window_seconds"),
        source_breakdown=breakdown,
    )


@app.post("/chat/message")
async def chat_message(
    payload: ChatMessageNotification,
    request: Request,
    x_orion_notify_token: Optional[str] = Header(default=None, alias="X-Orion-Notify-Token"),
) -> ChatMessageState:
    _check_token(x_orion_notify_token)
    notify_payload = _chat_message_to_notification(payload)
    transport = _get_transport(request)
    policy: Policy = request.app.state.policy
    env_lookup = request.app.state.env_lookup
    bus: OrionBusAsync | None = request.app.state.bus
    accepted, record = await _process_notification(
        payload=notify_payload,
        transport=transport,
        policy=policy,
        env_lookup=env_lookup,
        bus=bus,
    )
    if not accepted.ok:
        raise HTTPException(status_code=500, detail=accepted.detail or "Failed to store chat message")
    if not record:
        raise HTTPException(status_code=500, detail="Chat message record not stored")
    return _chat_message_to_schema(record)


@app.post("/chat/message/{message_id}/receipt")
async def chat_message_receipt(
    message_id: str,
    payload: ChatMessageReceipt,
    x_orion_notify_token: Optional[str] = Header(default=None, alias="X-Orion-Notify-Token"),
) -> ChatMessageState:
    _check_token(x_orion_notify_token)
    if str(payload.message_id) != message_id:
        raise HTTPException(status_code=400, detail="Message ID mismatch")
    with SessionLocal() as db:
        record = _apply_chat_message_receipt(db, payload)
        if not record:
            raise HTTPException(status_code=404, detail="Chat message not found")
        return _chat_message_to_schema(record)


@app.get("/chat/messages")
async def list_chat_messages(
    limit: int = Query(default=50, ge=1, le=200),
    status: Optional[str] = None,
    session_id: Optional[str] = None,
    x_orion_notify_token: Optional[str] = Header(default=None, alias="X-Orion-Notify-Token"),
) -> List[ChatMessageState]:
    _check_token(x_orion_notify_token)
    with SessionLocal() as db:
        query = db.query(NotificationRequestDB).filter(NotificationRequestDB.message_id.isnot(None))
        if session_id:
            query = query.filter(NotificationRequestDB.message_session_id == session_id)
        if status == "unread":
            query = query.filter(
                NotificationRequestDB.message_first_seen_at.is_(None),
                NotificationRequestDB.message_opened_at.is_(None),
                NotificationRequestDB.message_dismissed_at.is_(None),
            )
        elif status == "seen":
            query = query.filter(
                (NotificationRequestDB.message_first_seen_at.isnot(None))
                | (NotificationRequestDB.message_opened_at.isnot(None))
                | (NotificationRequestDB.message_dismissed_at.isnot(None))
            )
        rows = query.order_by(NotificationRequestDB.created_at.desc()).limit(limit).all()
        return [_chat_message_to_schema(row) for row in rows]


@app.get("/notifications")
async def list_notifications(
    limit: int = Query(default=50, ge=1, le=200),
    since: Optional[str] = None,
    severity: Optional[str] = None,
    event_kind: Optional[str] = None,
    x_orion_notify_token: Optional[str] = Header(default=None, alias="X-Orion-Notify-Token"),
) -> List[NotificationRecord]:
    _check_token(x_orion_notify_token)
    with SessionLocal() as db:
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


@app.get("/notifications/{notification_id}")
async def get_notification(
    notification_id: str,
    x_orion_notify_token: Optional[str] = Header(default=None, alias="X-Orion-Notify-Token"),
) -> NotificationRecord:
    _check_token(x_orion_notify_token)
    with SessionLocal() as db:
        row = db.query(NotificationRequestDB).filter_by(notification_id=notification_id).first()
        if not row:
            raise HTTPException(status_code=404, detail="Notification not found")
        return _record_to_schema(row)


@app.get("/notifications/{notification_id}/attempts")
async def get_attempts(
    notification_id: str,
    x_orion_notify_token: Optional[str] = Header(default=None, alias="X-Orion-Notify-Token"),
) -> List[DeliveryAttempt]:
    _check_token(x_orion_notify_token)
    with SessionLocal() as db:
        rows = db.query(NotificationAttemptDB).filter_by(notification_id=notification_id).all()
        return [_attempt_to_schema(row) for row in rows]


async def _process_notification(
    *,
    payload: NotificationRequest,
    transport: EmailTransport,
    policy: Policy,
    env_lookup: Dict[str, str],
    bus: OrionBusAsync | None,
) -> tuple[NotificationAccepted, Optional[NotificationRequestDB]]:
    now = datetime.utcnow()
    with SessionLocal() as db:
        recipient_profile = _get_recipient_profile(db, payload.recipient_group)
        preferences = _get_preferences(db, payload.recipient_group)
        decision, breakdown = _resolve_policy_with_preferences(
            policy=policy,
            recipient_profile=recipient_profile,
            preferences=preferences,
            event_kind=payload.event_kind,
            severity=payload.severity,
            payload=payload,
            now=now,
        )
        if payload.event_kind == ATTENTION_EVENT_KIND:
            _ensure_attention_context(payload, decision)
        if payload.event_kind == CHAT_MESSAGE_EVENT_KIND:
            _ensure_chat_message_context(payload, decision)
            presence_active = await _check_presence()
            payload.context["hub_active"] = presence_active
            if not presence_active:
                payload.context["silent"] = True
        payload.context["policy_breakdown"] = breakdown
        dedupe_window = payload.dedupe_window_seconds or decision.dedupe_window_seconds

        logger.info(
            "[NOTIFY] Received notification %s event=%s severity=%s",
            payload.notification_id,
            payload.event_kind,
            payload.severity,
        )

        attention_meta = _extract_attention_meta(payload)
        message_meta: Dict[str, Any] = {}
        if payload.event_kind == CHAT_MESSAGE_EVENT_KIND:
            message_meta = _extract_chat_message_meta(payload)
        if payload.dedupe_key and dedupe_window:
            dedupe_cutoff = now - timedelta(seconds=dedupe_window)
            existing = (
                db.query(NotificationRequestDB)
                .filter(
                    NotificationRequestDB.dedupe_key == payload.dedupe_key,
                    NotificationRequestDB.created_at >= dedupe_cutoff,
                )
                .first()
            )
            if existing:
                record = _store_request(
                    db,
                    payload,
                    status="deduped",
                    policy_action=decision.action,
                    drop_reason="dedupe_window",
                    **attention_meta,
                    **message_meta,
                )
                return (
                    NotificationAccepted(
                        ok=True,
                        notification_id=UUID(record.notification_id),
                        status="deduped",
                    ),
                    record,
                )

        if decision.throttle and decision.throttle.max_per_window > 0:
            throttle_cutoff = now - timedelta(seconds=decision.throttle.window_seconds)
            throttled_count = (
                db.query(func.count(NotificationRequestDB.notification_id))
                .filter(
                    NotificationRequestDB.event_kind == payload.event_kind,
                    NotificationRequestDB.recipient_group == decision.recipient_group,
                    NotificationRequestDB.created_at >= throttle_cutoff,
                    NotificationRequestDB.status != "dropped",
                )
                .scalar()
            )
            if throttled_count and throttled_count >= decision.throttle.max_per_window:
                record = _store_request(
                    db,
                    payload,
                    status="throttled",
                    policy_action=decision.action,
                    drop_reason="throttled",
                    **attention_meta,
                    **message_meta,
                )
                return (
                    NotificationAccepted(
                        ok=True,
                        notification_id=UUID(record.notification_id),
                        status="throttled",
                    ),
                    record,
                )

        if not decision.allowed:
            record = _store_request(
                db,
                payload,
                status="dropped",
                policy_action=decision.action,
                drop_reason=decision.reason or "policy_drop",
                **attention_meta,
                **message_meta,
            )
            return (
                NotificationAccepted(ok=True, notification_id=UUID(record.notification_id), status="dropped"),
                record,
            )

        if not decision.channels:
            record = _store_request(
                db,
                payload,
                status="stored",
                policy_action=decision.action,
                drop_reason=decision.reason or "digest_only",
                **attention_meta,
                **message_meta,
            )
            return (
                NotificationAccepted(ok=True, notification_id=UUID(record.notification_id), status="stored"),
                record,
            )

        record = _store_request(
            db,
            payload,
            status="pending",
            policy_action=decision.action,
            drop_reason=None,
            **attention_meta,
            **message_meta,
        )

        recipients = policy.resolve_recipient_emails(decision.recipient_group, env_lookup)
        if not recipients:
            recipients = settings.notify_email_to

        if "email" in decision.channels:
            try:
                transport.send(payload, to_override=recipients, from_override=settings.NOTIFY_EMAIL_FROM)
                _store_attempt(db, record.notification_id, channel="email", status="sent")
                record.status = "sent"
                db.commit()
            except Exception as exc:
                logger.error("[NOTIFY] Failed to send email: %s", exc)
                _store_attempt(db, record.notification_id, channel="email", status="failed", error=str(exc))
                record.status = "failed"
                db.commit()
                raise HTTPException(status_code=500, detail=str(exc))

        if "in_app" in decision.channels:
            if settings.NOTIFY_IN_APP_ENABLED and bus is not None:
                asyncio.create_task(
                    _publish_in_app_event(
                        bus=bus,
                        notification_id=record.notification_id,
                        payload=payload,
                        status="delivered",
                    )
                )
            else:
                logger.warning("[NOTIFY] In-app channel disabled or bus unavailable.")
                if "email" not in decision.channels:
                    record.status = "stored"
                    record.drop_reason = "in_app_disabled"
                    db.commit()

        return (
            NotificationAccepted(ok=True, notification_id=UUID(record.notification_id), status=record.status),
            record,
        )


def _store_request(
    db: Session,
    payload: NotificationRequest,
    *,
    status: str,
    policy_action: Optional[str],
    drop_reason: Optional[str],
    attention_id: Optional[str] = None,
    attention_require_ack: bool = False,
    attention_ack_deadline_minutes: Optional[int] = None,
    attention_expires_at: Optional[datetime] = None,
    attention_escalation_channels: Optional[List[str]] = None,
    message_id: Optional[str] = None,
    message_session_id: Optional[str] = None,
    message_preview_text: Optional[str] = None,
    message_full_text: Optional[str] = None,
    message_require_read_receipt: bool = False,
    message_expires_at: Optional[datetime] = None,
) -> NotificationRequestDB:
    # Explicitly set created_at if not present in payload, though payload.created_at usually has a default.
    # However, if payload.created_at is None (unlikely given schema default), we need a value.
    # More importantly, SQLAlchemy default=datetime.utcnow only fires on flush.
    # We must ensure record.created_at is set for the bus event and API return.
    # payload.created_at comes from the API request model which has default_factory=datetime.utcnow.
    # So payload.created_at SHOULD be set.
    # But let's be safe and ensure we use the same timestamp for everything.
    ts = payload.created_at or datetime.utcnow()

    record = NotificationRequestDB(
        notification_id=str(payload.notification_id),
        source_service=payload.source_service,
        event_kind=payload.event_kind,
        severity=payload.severity,
        title=payload.title,
        body_text=payload.body_text,
        body_md=payload.body_md,
        context=payload.context,
        tags=payload.tags,
        recipient_group=payload.recipient_group,
        channels_requested=payload.channels_requested,
        dedupe_key=payload.dedupe_key,
        dedupe_window_seconds=payload.dedupe_window_seconds,
        ttl_seconds=payload.ttl_seconds,
        correlation_id=payload.correlation_id,
        session_id=payload.session_id,
        created_at=ts,
        status=status,
        policy_action=policy_action,
        drop_reason=drop_reason,
        attention_id=attention_id,
        attention_require_ack=attention_require_ack,
        attention_ack_deadline_minutes=attention_ack_deadline_minutes,
        attention_expires_at=attention_expires_at,
        attention_escalation_channels=attention_escalation_channels,
        message_id=message_id,
        message_session_id=message_session_id,
        message_preview_text=message_preview_text,
        message_full_text=message_full_text,
        message_require_read_receipt=message_require_read_receipt,
        message_expires_at=message_expires_at,
    )

    # Use existing local DB write (works for SQLite and if we connect to Postgres for reads)
    # But if we are in Postgres mode (PROD), we ALSO publish to bus for durable persistence via SQL writer
    # The 'db' session passed here is SessionLocal, which is bound to settings.POSTGRES_URI

    # If in dev (sqlite), write locally.
    # If in prod (postgres), we might be reading from postgres, so we can't write if we don't own the table/permissions or if architecture forbids it.
    # User said: "persistence should flow bus -> sql-writer".
    # And "If notify serves reads, notify should query via sql-writer read path or direct DB read"

    # Check URI scheme
    is_sqlite = str(settings.POSTGRES_URI).startswith("sqlite")

    if is_sqlite:
        # Local dev: write to local DB
        db.add(record)
        db.commit()
        db.refresh(record)
    else:
        # Prod/Shared DB: Publish to bus
        # Note: We return the record object, but it is transient (not attached to session or refreshed from DB)
        # This means generated fields (like defaults if any, though we set most) might be missing if relying on DB.
        # But we set most fields explicitly.
        bus = getattr(app.state, "bus", None)
        if bus:
            event = NotificationRecord(
                notification_id=UUID(record.notification_id),
                source_service=record.source_service,
                event_kind=record.event_kind,
                severity=record.severity,
                title=record.title,
                body_text=record.body_text,
                body_md=record.body_md,
                context=record.context,
                tags=record.tags,
                recipient_group=record.recipient_group,
                channels_requested=record.channels_requested,
                dedupe_key=record.dedupe_key,
                dedupe_window_seconds=record.dedupe_window_seconds,
                ttl_seconds=record.ttl_seconds,
                correlation_id=record.correlation_id,
                session_id=record.session_id,
                created_at=record.created_at,
                status=record.status,
                policy_action=record.policy_action,
                drop_reason=record.drop_reason,
            )

            # Fire and forget persistence
            asyncio.create_task(_publish_persistence_event(bus, "orion:notify:persistence:request", event))
        else:
            logger.warning("Orion Bus not available for persistence in non-SQLite mode")

    return record


def _store_attempt(
    db: Session,
    notification_id: str,
    *,
    channel: str,
    status: str,
    error: Optional[str] = None,
) -> NotificationAttemptDB:
    attempt = NotificationAttemptDB(
        notification_id=notification_id,
        channel=channel,
        status=status,
        error=error,
    )
    db.add(attempt)
    db.commit()
    db.refresh(attempt)
    return attempt


async def _init_bus() -> OrionBusAsync | None:
    if not settings.ORION_BUS_ENABLED:
        return None
    if not settings.NOTIFY_IN_APP_ENABLED:
        return None
    bus = OrionBusAsync(
        url=settings.ORION_BUS_URL,
        enabled=settings.ORION_BUS_ENABLED,
        enforce_catalog=settings.ORION_BUS_ENFORCE_CATALOG,
    )
    try:
        await bus.connect()
    except Exception as exc:
        logger.error("Failed to connect Orion bus for in-app notifications: %s", exc)
        return None
    return bus


async def _publish_in_app_event(
    *,
    bus: OrionBusAsync,
    notification_id: str,
    payload: NotificationRequest,
    status: str,
) -> None:
    try:
        attention_id = _parse_attention_id(payload.context)
        message_id = _parse_message_id(payload.context)
        notification_type = _resolve_notification_type(payload.event_kind)
        silent = bool(payload.context.get("silent")) if payload.context else None
        event = HubNotificationEvent(
            notification_id=UUID(notification_id),
            created_at=payload.created_at,
            severity=payload.severity,
            event_kind=payload.event_kind,
            source_service=payload.source_service,
            title=payload.title,
            body_text=_truncate_text(payload.body_text),
            tags=payload.tags,
            correlation_id=payload.correlation_id,
            session_id=payload.session_id,
            status=status,
            attention_id=attention_id,
            message_id=message_id,
            notification_type=notification_type,
            silent=silent,
        )
        env = BaseEnvelope(
            kind="notify.in_app.v1",
            source=ServiceRef(
                name=settings.SERVICE_NAME,
                node=settings.NODE_NAME,
                version=settings.SERVICE_VERSION,
            ),
            payload=event.model_dump(mode="json"),
        )
        await bus.publish(settings.NOTIFY_IN_APP_CHANNEL, env)
        with SessionLocal() as db:
            _store_attempt(db, notification_id, channel="in_app", status="sent")
            _update_request_status(db, notification_id, "sent")
    except Exception as exc:
        logger.error("[NOTIFY] Failed to publish in-app notification: %s", exc)
        with SessionLocal() as db:
            _store_attempt(db, notification_id, channel="in_app", status="failed", error=str(exc))
            _update_request_status(db, notification_id, "failed")


def _update_request_status(db: Session, notification_id: str, status: str) -> None:
    record = db.query(NotificationRequestDB).filter_by(notification_id=notification_id).first()
    if not record:
        return
    if record.status == "sent" and status != "sent":
        return
    record.status = status
    db.commit()


def _truncate_text(value: Optional[str], max_len: int = 500) -> Optional[str]:
    if value is None:
        return None
    if len(value) <= max_len:
        return value
    return value[: max_len - 1].rstrip() + "…"


def _record_to_schema(row: NotificationRequestDB) -> NotificationRecord:
    return NotificationRecord(
        notification_id=UUID(row.notification_id),
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


def _attempt_to_schema(row: NotificationAttemptDB) -> DeliveryAttempt:
    return DeliveryAttempt(
        attempt_id=UUID(row.attempt_id),
        notification_id=UUID(row.notification_id),
        channel=row.channel,
        status=row.status,
        provider_message_id=row.provider_message_id,
        error=row.error,
        attempted_at=row.attempted_at,
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


def _get_recipient_profile(db: Session, recipient_group: str) -> Optional[RecipientProfileDB]:
    return db.query(RecipientProfileDB).filter_by(recipient_group=recipient_group).first()


def _get_preferences(db: Session, recipient_group: str) -> List[NotificationPreferenceDB]:
    return (
        db.query(NotificationPreferenceDB)
        .filter(NotificationPreferenceDB.recipient_group == recipient_group)
        .all()
    )


def _validate_preferences(preferences: List[NotificationPreference]) -> None:
    for pref in preferences:
        if pref.throttle_window_seconds is not None:
            if not (THROTTLE_WINDOW_MIN_SECONDS <= pref.throttle_window_seconds <= THROTTLE_WINDOW_MAX_SECONDS):
                raise HTTPException(
                    status_code=400,
                    detail=f"throttle_window_seconds must be between {THROTTLE_WINDOW_MIN_SECONDS} and {THROTTLE_WINDOW_MAX_SECONDS}",
                )
        if pref.throttle_max_per_window is not None:
            if not (THROTTLE_MAX_MIN <= pref.throttle_max_per_window <= THROTTLE_MAX_MAX):
                raise HTTPException(
                    status_code=400,
                    detail=f"throttle_max_per_window must be between {THROTTLE_MAX_MIN} and {THROTTLE_MAX_MAX}",
                )
        if pref.dedupe_window_seconds is not None:
            if not (DEDUPE_WINDOW_MIN_SECONDS <= pref.dedupe_window_seconds <= DEDUPE_WINDOW_MAX_SECONDS):
                raise HTTPException(
                    status_code=400,
                    detail=f"dedupe_window_seconds must be between {DEDUPE_WINDOW_MIN_SECONDS} and {DEDUPE_WINDOW_MAX_SECONDS}",
                )


def _resolve_policy_with_preferences(
    *,
    policy: Policy,
    recipient_profile: Optional[RecipientProfileDB],
    preferences: List[NotificationPreferenceDB],
    event_kind: str,
    severity: str,
    payload: Optional[NotificationRequest] = None,
    now: Optional[datetime] = None,
) -> tuple[PolicyDecision, Dict[str, Any]]:
    quiet_hours_enabled = False
    quiet_hours_override = None
    if recipient_profile:
        quiet_hours_enabled = bool(recipient_profile.quiet_hours_enabled)
        quiet_hours_override = {
            "timezone": recipient_profile.timezone,
            "start": recipient_profile.quiet_start_local,
            "end": recipient_profile.quiet_end_local,
        }
    request_payload = payload or NotificationRequest(
        source_service="resolve",
        event_kind=event_kind,
        severity=severity,
        title="resolve",
    )
    now_ts = now or datetime.utcnow()
    decision = policy.evaluate(
        request_payload,
        now_ts,
        quiet_hours_enabled=False,
        quiet_hours_override=quiet_hours_override,
    )
    breakdown: Dict[str, Any] = {
        "policy_action": decision.action,
        "quiet_hours_applied": False,
        "quiet_hours_in_effect": False,
        "preference_scope": None,
        "quiet_hours": quiet_hours_override or {},
        "sources": {
            "channels": "policy",
            "dedupe_window_seconds": "policy",
            "throttle": "policy",
            "escalation_enabled": "policy",
            "escalation_delay_minutes": "policy",
            "quiet_hours": "recipient_profile" if quiet_hours_override else "policy",
        },
    }
    breakdown["dedupe_window_seconds"] = decision.dedupe_window_seconds
    if decision.throttle:
        breakdown["throttle_max_per_window"] = decision.throttle.max_per_window
        breakdown["throttle_window_seconds"] = decision.throttle.window_seconds
    escalation_enabled = bool(decision.escalation_channels) if decision.escalation_channels is not None else None
    if escalation_enabled is not None:
        breakdown["escalation_enabled"] = escalation_enabled
    if event_kind == ATTENTION_EVENT_KIND and decision.ack_deadline_minutes is not None:
        breakdown["escalation_delay_minutes"] = decision.ack_deadline_minutes
    if event_kind == CHAT_MESSAGE_EVENT_KIND and decision.read_receipt_deadline_minutes is not None:
        breakdown["escalation_delay_minutes"] = decision.read_receipt_deadline_minutes
    severity_key = severity.lower()
    pref_event = _select_preference(preferences, "event_kind", event_kind)
    pref_severity = None if pref_event else _select_preference(preferences, "severity", severity_key)
    pref = pref_event or pref_severity
    if pref:
        breakdown["preference_scope"] = f"{pref.scope_type}:{pref.scope_value}"
        decision = _apply_preference_overrides(decision, pref, event_kind, breakdown)

    quiet_hours_active = quiet_hours_enabled and policy.is_quiet_hours(now_ts, quiet_hours_override)
    breakdown["quiet_hours_in_effect"] = quiet_hours_active
    if quiet_hours_active and severity_key not in {"critical", "error"}:
        pref_allows_in_app = bool(pref and "in_app" in _channels_from_json(pref.channels_enabled))
        if not pref_allows_in_app and "in_app" in decision.channels:
            decision.channels = [channel for channel in decision.channels if channel != "in_app"]
            breakdown["quiet_hours_applied"] = True
            breakdown["sources"]["channels"] = "quiet_hours"
        elif not decision.channels:
            breakdown["quiet_hours_applied"] = True
    breakdown["channels_final"] = decision.channels
    return decision, breakdown


def _select_preference(
    preferences: List[NotificationPreferenceDB],
    scope_type: str,
    scope_value: str,
) -> Optional[NotificationPreferenceDB]:
    for pref in preferences:
        if pref.scope_type == scope_type and pref.scope_value == scope_value:
            return pref
    return None


def _apply_preference_overrides(
    decision: PolicyDecision,
    pref: NotificationPreferenceDB,
    event_kind: str,
    breakdown: Dict[str, Any],
) -> PolicyDecision:
    channels_override = _channels_from_json(pref.channels_enabled)
    if channels_override is not None:
        decision.channels = list(channels_override)
        breakdown["sources"]["channels"] = f"preference:{pref.scope_type}"
    if pref.dedupe_window_seconds is not None:
        decision.dedupe_window_seconds = pref.dedupe_window_seconds
        breakdown["dedupe_window_seconds"] = pref.dedupe_window_seconds
        breakdown["sources"]["dedupe_window_seconds"] = f"preference:{pref.scope_type}"
    if pref.throttle_max_per_window is not None or pref.throttle_window_seconds is not None:
        max_per = pref.throttle_max_per_window or (decision.throttle.max_per_window if decision.throttle else 0)
        window_sec = pref.throttle_window_seconds or (decision.throttle.window_seconds if decision.throttle else 0)
        decision.throttle = ThrottleRule(max_per_window=max_per, window_seconds=window_sec)
        breakdown["throttle_max_per_window"] = max_per
        breakdown["throttle_window_seconds"] = window_sec
        breakdown["sources"]["throttle"] = f"preference:{pref.scope_type}"
    if pref.escalation_enabled is not None:
        escalation_enabled = bool(pref.escalation_enabled)
        breakdown["escalation_enabled"] = escalation_enabled
        breakdown["sources"]["escalation_enabled"] = f"preference:{pref.scope_type}"
        if not escalation_enabled:
            decision.escalation_channels = []
    if pref.escalation_delay_minutes is not None:
        breakdown["escalation_delay_minutes"] = pref.escalation_delay_minutes
        breakdown["sources"]["escalation_delay_minutes"] = f"preference:{pref.scope_type}"
        if event_kind == ATTENTION_EVENT_KIND:
            decision.ack_deadline_minutes = pref.escalation_delay_minutes
        elif event_kind == CHAT_MESSAGE_EVENT_KIND:
            decision.read_receipt_deadline_minutes = pref.escalation_delay_minutes
    return decision


def _attention_request_to_notification(payload: ChatAttentionRequest) -> NotificationRequest:
    context = dict(payload.context)
    context.update(
        {
            "attention_id": str(payload.attention_id),
            "reason": payload.reason,
            "require_ack": payload.require_ack,
            "expires_at": payload.expires_at.isoformat() if payload.expires_at else None,
        }
    )
    return NotificationRequest(
        source_service=payload.source_service,
        event_kind=ATTENTION_EVENT_KIND,
        severity=payload.severity,
        title=payload.reason,
        body_text=payload.message,
        context=context,
        tags=["chat", "attention"],
        correlation_id=payload.correlation_id,
        session_id=payload.session_id,
        created_at=payload.created_at,
    )


def _ensure_attention_context(payload: NotificationRequest, decision: PolicyDecision) -> None:
    payload.context = payload.context or {}
    if "attention_id" not in payload.context:
        payload.context["attention_id"] = str(payload.notification_id)
    if "require_ack" not in payload.context:
        payload.context["require_ack"] = decision.require_ack
    if decision.ack_deadline_minutes is not None and "ack_deadline_minutes" not in payload.context:
        payload.context["ack_deadline_minutes"] = decision.ack_deadline_minutes
    if decision.escalation_channels and "escalation_channels" not in payload.context:
        payload.context["escalation_channels"] = decision.escalation_channels


def _parse_attention_id(context: Dict[str, Any]) -> Optional[UUID]:
    raw = context.get("attention_id") if context else None
    if not raw:
        return None
    if isinstance(raw, UUID):
        return raw
    try:
        return UUID(str(raw))
    except (ValueError, TypeError):
        return None


def _parse_message_id(context: Dict[str, Any]) -> Optional[UUID]:
    raw = context.get("message_id") if context else None
    if not raw:
        return None
    if isinstance(raw, UUID):
        return raw
    try:
        return UUID(str(raw))
    except (ValueError, TypeError):
        return None


def _resolve_notification_type(event_kind: str) -> Optional[str]:
    if event_kind == CHAT_MESSAGE_EVENT_KIND:
        return "chat_message"
    if event_kind == ATTENTION_EVENT_KIND:
        return "chat_attention"
    return None


def _extract_attention_meta(payload: NotificationRequest) -> Dict[str, Any]:
    context = payload.context or {}
    attention_id = _parse_attention_id(context)
    require_ack = bool(context.get("require_ack", False))
    ack_deadline = context.get("ack_deadline_minutes")
    try:
        ack_deadline_minutes = int(ack_deadline) if ack_deadline is not None else None
    except (TypeError, ValueError):
        ack_deadline_minutes = None
    escalation_channels = context.get("escalation_channels")
    if escalation_channels is None:
        escalation_channels_list = None
    elif isinstance(escalation_channels, list):
        escalation_channels_list = [str(item) for item in escalation_channels]
    else:
        escalation_channels_list = [str(escalation_channels)]
    expires_at = context.get("expires_at")
    attention_expires_at = None
    if isinstance(expires_at, datetime):
        attention_expires_at = expires_at
    elif isinstance(expires_at, str):
        try:
            attention_expires_at = datetime.fromisoformat(expires_at)
        except ValueError:
            attention_expires_at = None
    return {
        "attention_id": str(attention_id) if attention_id else None,
        "attention_require_ack": require_ack,
        "attention_ack_deadline_minutes": ack_deadline_minutes,
        "attention_expires_at": attention_expires_at,
        "attention_escalation_channels": escalation_channels_list,
    }


def _chat_message_to_notification(payload: ChatMessageNotification) -> NotificationRequest:
    context = {
        "message_id": str(payload.message_id),
        "preview_text": payload.preview_text,
        "full_text": payload.full_text,
        "require_read_receipt": payload.require_read_receipt,
        "expires_at": payload.expires_at.isoformat() if payload.expires_at else None,
    }
    return NotificationRequest(
        source_service=payload.source_service,
        event_kind=CHAT_MESSAGE_EVENT_KIND,
        severity=payload.severity,
        title=payload.title or "New message from Orion",
        body_text=payload.preview_text,
        body_md=payload.full_text,
        context=context,
        tags=payload.tags or ["chat", "message"],
        recipient_group="juniper_primary",
        correlation_id=payload.correlation_id,
        session_id=payload.session_id,
        created_at=payload.created_at,
    )


def _ensure_chat_message_context(payload: NotificationRequest, decision: PolicyDecision) -> None:
    payload.context = payload.context or {}
    if "message_id" not in payload.context:
        payload.context["message_id"] = str(payload.notification_id)
    if "preview_text" not in payload.context:
        payload.context["preview_text"] = payload.body_text
    if "full_text" not in payload.context:
        payload.context["full_text"] = payload.body_md
    if "require_read_receipt" not in payload.context:
        payload.context["require_read_receipt"] = decision.require_read_receipt
    if decision.read_receipt_deadline_minutes is not None and "read_receipt_deadline_minutes" not in payload.context:
        payload.context["read_receipt_deadline_minutes"] = decision.read_receipt_deadline_minutes
    if decision.escalation_channels and "escalation_channels" not in payload.context:
        payload.context["escalation_channels"] = decision.escalation_channels


def _extract_chat_message_meta(payload: NotificationRequest) -> Dict[str, Any]:
    context = payload.context or {}
    message_id = _parse_message_id(context)
    preview_text = context.get("preview_text") or payload.body_text
    full_text = context.get("full_text") or payload.body_md
    require_read_receipt = bool(context.get("require_read_receipt", False))
    expires_at = context.get("expires_at")
    message_expires_at = None
    if isinstance(expires_at, datetime):
        message_expires_at = expires_at
    elif isinstance(expires_at, str):
        try:
            message_expires_at = datetime.fromisoformat(expires_at)
        except ValueError:
            message_expires_at = None
    return {
        "message_id": str(message_id) if message_id else None,
        "message_session_id": payload.session_id,
        "message_preview_text": preview_text,
        "message_full_text": full_text,
        "message_require_read_receipt": require_read_receipt,
        "message_expires_at": message_expires_at,
    }


def _attention_to_schema(row: NotificationRequestDB) -> ChatAttentionState:
    context = row.context or {}
    reason = context.get("reason") or row.title
    status = "pending" if row.attention_require_ack and row.attention_acked_at is None else "acked"
    return ChatAttentionState(
        attention_id=UUID(row.attention_id),
        notification_id=UUID(row.notification_id) if row.notification_id else None,
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


def _chat_message_to_schema(row: NotificationRequestDB) -> ChatMessageState:
    status = "unread"
    if row.message_opened_at or row.message_dismissed_at or row.message_first_seen_at:
        status = "seen"
    return ChatMessageState(
        message_id=UUID(row.message_id),
        notification_id=UUID(row.notification_id) if row.notification_id else None,
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
        first_seen_at=row.message_first_seen_at,
        opened_at=row.message_opened_at,
        dismissed_at=row.message_dismissed_at,
        escalated_at=row.message_escalated_at,
        status=status,
    )


def _apply_attention_ack(db: Session, payload: ChatAttentionAck) -> Optional[NotificationRequestDB]:
    record = db.query(NotificationRequestDB).filter_by(attention_id=str(payload.attention_id)).first()
    if not record:
        return None
    record.attention_acked_at = payload.acked_at
    record.attention_ack_type = payload.ack_type
    record.attention_ack_actor = payload.actor
    record.attention_ack_note = payload.note
    db.commit()
    db.refresh(record)
    return record


def _apply_chat_message_receipt(db: Session, payload: ChatMessageReceipt) -> Optional[NotificationRequestDB]:
    record = db.query(NotificationRequestDB).filter_by(message_id=str(payload.message_id)).first()
    if not record:
        return None
    if record.message_session_id and record.message_session_id != payload.session_id:
        return None

    # Persist receipt
    is_sqlite = str(settings.POSTGRES_URI).startswith("sqlite")

    if is_sqlite:
        existing_receipt = (
            db.query(NotificationReceiptDB)
            .filter_by(
                message_id=str(payload.message_id),
                receipt_type=payload.receipt_type,
            )
            .first()
        )
        if not existing_receipt:
            receipt = NotificationReceiptDB(
                message_id=str(payload.message_id),
                receipt_type=payload.receipt_type,
                session_id=payload.session_id,
                received_at=payload.received_at,
            )
            db.add(receipt)
            # Local update to request record
            if payload.receipt_type == "seen" and record.message_first_seen_at is None:
                record.message_first_seen_at = payload.received_at
            elif payload.receipt_type == "opened":
                record.message_opened_at = payload.received_at
                if record.message_first_seen_at is None:
                    record.message_first_seen_at = payload.received_at
            elif payload.receipt_type == "dismissed":
                record.message_dismissed_at = payload.received_at
                if record.message_first_seen_at is None:
                    record.message_first_seen_at = payload.received_at
            db.commit()
            db.refresh(record)
    else:
        # Prod: Publish receipt to bus
        bus = getattr(app.state, "bus", None)
        if bus:
            receipt_event = NotificationReceiptEvent(
                message_id=payload.message_id,
                receipt_type=payload.receipt_type,
                session_id=payload.session_id,
                received_at=payload.received_at,
                created_at=datetime.utcnow()
            )
            asyncio.create_task(_publish_persistence_event(bus, "orion:notify:persistence:receipt", receipt_event))

            # Optimistically update the record in memory (record is attached to a session that might be read-only or we shouldn't write to)
            # If we are reading from Postgres, we shouldn't write.
            # But the caller expects the updated state.
            if payload.receipt_type == "seen" and record.message_first_seen_at is None:
                record.message_first_seen_at = payload.received_at
            elif payload.receipt_type == "opened":
                record.message_opened_at = payload.received_at
                if record.message_first_seen_at is None:
                    record.message_first_seen_at = payload.received_at
            elif payload.receipt_type == "dismissed":
                record.message_dismissed_at = payload.received_at
                if record.message_first_seen_at is None:
                    record.message_first_seen_at = payload.received_at
            # Do NOT commit db.

    return record # Returns updated object (either from DB refresh or optimistic update)

async def _publish_persistence_event(bus: OrionBusAsync, channel: str, payload: Any) -> None:
    try:
        env = BaseEnvelope(
            kind=payload.model_extra.get("message_kind") if hasattr(payload, "model_extra") and payload.model_extra else "notify.persistence.event", # fallback
            source=ServiceRef(
                name=settings.SERVICE_NAME,
                node=settings.NODE_NAME,
                version=settings.SERVICE_VERSION,
            ),
            payload=payload.model_dump(mode="json"),
        )
        # Fix kind lookup since we are using explicit mapping in channels.yaml but need kind in envelope
        if channel == "orion:notify:persistence:request":
             env.kind = "notify.notification.request.v1"
        elif channel == "orion:notify:persistence:receipt":
             env.kind = "notify.notification.receipt.v1"

        await bus.publish(channel, env)
    except Exception as exc:
        logger.error(f"Failed to publish persistence event to {channel}: {exc}")


# Helper to avoid indentation mess above - we used inline logic but need to be careful with existing logic flow
# The above search/replace block for receipt assumes we are replacing the *entire* receipt logic block.
# Let's verify the search block covers what we want to replace.
# The search block starts with "# Persist receipt" and ends with "elif payload.receipt_type == 'opened':"
# This is risky if the code continues.
# Better to implement a cleaner block replacement.
        record.message_first_seen_at = payload.received_at
    elif payload.receipt_type == "opened":
        record.message_opened_at = payload.received_at
        if record.message_first_seen_at is None:
            record.message_first_seen_at = payload.received_at
    elif payload.receipt_type == "dismissed":
        record.message_dismissed_at = payload.received_at
        if record.message_first_seen_at is None:
            record.message_first_seen_at = payload.received_at
    db.commit()
    db.refresh(record)
    return record


async def _run_escalation_loop(app: FastAPI) -> None:
    poll_seconds = max(settings.NOTIFY_ESCALATION_POLL_SECONDS, 5)
    while True:
        try:
            await _check_attention_escalations(app)
            await _check_chat_message_escalations(app)
        except Exception as exc:
            logger.error("Attention escalation loop failed: %s", exc, exc_info=True)
        await asyncio.sleep(poll_seconds)


async def _check_attention_escalations(app: FastAPI) -> None:
    transport = app.state.transport
    policy: Policy = app.state.policy
    env_lookup = app.state.env_lookup
    bus: OrionBusAsync | None = app.state.bus
    now = datetime.utcnow()

    with SessionLocal() as db:
        rows = (
            db.query(NotificationRequestDB)
            .filter(
                NotificationRequestDB.attention_id.isnot(None),
                NotificationRequestDB.attention_require_ack.is_(True),
                NotificationRequestDB.attention_acked_at.is_(None),
                NotificationRequestDB.attention_ack_deadline_minutes.isnot(None),
                NotificationRequestDB.attention_escalated_at.is_(None),
            )
            .all()
        )

        for row in rows:
            if row.attention_ack_deadline_minutes is None:
                continue
            deadline = row.created_at + timedelta(minutes=row.attention_ack_deadline_minutes)
            if now < deadline:
                continue
            escalation_channels = row.attention_escalation_channels or []
            if not escalation_channels:
                row.attention_escalated_at = now
                db.commit()
                continue
            context = dict(row.context or {})
            context.update(
                {
                    "attention_id": row.attention_id,
                    "escalation_for": row.notification_id,
                    "escalation_reason": "ack_deadline_elapsed",
                }
            )
            escalation_payload = NotificationRequest(
                source_service=row.source_service,
                event_kind=ATTENTION_ESCALATION_EVENT_KIND,
                severity=row.severity,
                title=f"Attention escalation: {row.title}",
                body_text=(
                    f"Attention request was not acknowledged in {row.attention_ack_deadline_minutes} minutes.\n"
                    f"Message: {row.body_text or ''}"
                ),
                context=context,
                tags=["chat", "attention", "escalation"],
                recipient_group=row.recipient_group,
                channels_requested=escalation_channels,
                correlation_id=row.correlation_id,
                session_id=row.session_id,
            )
            accepted, _ = await _process_notification(
                payload=escalation_payload,
                transport=transport,
                policy=policy,
                env_lookup=env_lookup,
                bus=bus,
            )
            if accepted.ok:
                row.attention_escalated_at = now
                db.commit()


async def _check_chat_message_escalations(app: FastAPI) -> None:
    transport = app.state.transport
    policy: Policy = app.state.policy
    env_lookup = app.state.env_lookup
    bus: OrionBusAsync | None = app.state.bus
    now = datetime.utcnow()

    with SessionLocal() as db:
        rows = (
            db.query(NotificationRequestDB)
            .filter(
                NotificationRequestDB.message_id.isnot(None),
                NotificationRequestDB.message_require_read_receipt.is_(True),
                NotificationRequestDB.message_first_seen_at.is_(None),
                NotificationRequestDB.message_opened_at.is_(None),
                NotificationRequestDB.message_dismissed_at.is_(None),
                NotificationRequestDB.message_escalated_at.is_(None),
            )
            .all()
        )

        for row in rows:
            context = row.context or {}
            deadline_minutes = context.get("read_receipt_deadline_minutes")
            try:
                deadline_minutes = int(deadline_minutes) if deadline_minutes is not None else None
            except (TypeError, ValueError):
                deadline_minutes = None
            if not deadline_minutes:
                continue
            deadline = row.created_at + timedelta(minutes=deadline_minutes)
            if now < deadline:
                continue
            escalation_channels = context.get("escalation_channels") or []
            if not escalation_channels:
                row.message_escalated_at = now
                db.commit()
                continue
            hub_url = settings.NOTIFY_HUB_URL.strip().rstrip("/")
            hub_hint = f"{hub_url}/?session_id={row.message_session_id}" if hub_url else ""
            context = dict(context)
            context.update(
                {
                    "message_id": row.message_id,
                    "escalation_for": row.notification_id,
                    "escalation_reason": "read_receipt_deadline_elapsed",
                    "hub_url": hub_hint or None,
                }
            )
            escalation_payload = NotificationRequest(
                source_service=row.source_service,
                event_kind=CHAT_MESSAGE_ESCALATION_EVENT_KIND,
                severity=row.severity,
                title=f"Chat message unread: {row.title}",
                body_text=(
                    "Chat message was not opened within the expected window.\n"
                    f"Preview: {row.message_preview_text or row.body_text or ''}\n"
                    f"Session: {row.message_session_id or 'unknown'}\n"
                    f"{'Hub: ' + hub_hint if hub_hint else ''}"
                ),
                context=context,
                tags=["chat", "message", "escalation"],
                recipient_group=row.recipient_group,
                channels_requested=escalation_channels,
                correlation_id=row.correlation_id,
                session_id=row.message_session_id,
            )
            accepted, _ = await _process_notification(
                payload=escalation_payload,
                transport=transport,
                policy=policy,
                env_lookup=env_lookup,
                bus=bus,
            )
            if accepted.ok:
                row.message_escalated_at = now
                db.commit()


async def _check_presence() -> bool:
    if not settings.NOTIFY_PRESENCE_URL:
        return True
    try:
        resp = requests.get(
            settings.NOTIFY_PRESENCE_URL,
            timeout=settings.NOTIFY_PRESENCE_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()
        return bool(data.get("active", False))
    except Exception as exc:
        logger.warning("Presence check failed: %s", exc)
        return False


def _ensure_bootstrap_defaults() -> None:
    with SessionLocal() as db:
        profile = db.query(RecipientProfileDB).filter_by(recipient_group="juniper_primary").first()
        if not profile:
            profile = RecipientProfileDB(
                recipient_group="juniper_primary",
                display_name="Juniper",
                timezone="America/Denver",
                quiet_hours_enabled=0,
                quiet_start_local="22:00",
                quiet_end_local="07:00",
            )
            db.add(profile)
            db.commit()
        existing = (
            db.query(NotificationPreferenceDB)
            .filter(NotificationPreferenceDB.recipient_group == "juniper_primary")
            .count()
        )
        if existing == 0:
            seed = [
                NotificationPreferenceDB(
                    recipient_group="juniper_primary",
                    scope_type="event_kind",
                    scope_value=ATTENTION_EVENT_KIND,
                    channels_enabled=_channels_to_json(["in_app"]),
                    escalation_enabled=1,
                    escalation_delay_minutes=60,
                ),
                NotificationPreferenceDB(
                    recipient_group="juniper_primary",
                    scope_type="event_kind",
                    scope_value=CHAT_MESSAGE_EVENT_KIND,
                    channels_enabled=_channels_to_json(["in_app"]),
                    escalation_enabled=1,
                    escalation_delay_minutes=60,
                ),
                NotificationPreferenceDB(
                    recipient_group="juniper_primary",
                    scope_type="severity",
                    scope_value="error",
                    channels_enabled=_channels_to_json(["email", "in_app"]),
                ),
                NotificationPreferenceDB(
                    recipient_group="juniper_primary",
                    scope_type="severity",
                    scope_value="warning",
                    channels_enabled=_channels_to_json(["in_app"]),
                ),
                NotificationPreferenceDB(
                    recipient_group="juniper_primary",
                    scope_type="severity",
                    scope_value="info",
                    channels_enabled=_channels_to_json([]),
                ),
            ]
            db.add_all(seed)
            db.commit()


def _channels_from_json(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item) for item in raw if str(item)]
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return [str(item) for item in data if str(item)]
    return []


def _channels_to_json(channels: Optional[List[str]]) -> str:
    if not channels:
        return "[]"
    return json.dumps([str(item) for item in channels])


def _int_to_bool(value: Optional[int]) -> Optional[bool]:
    if value is None:
        return None
    return bool(value)
