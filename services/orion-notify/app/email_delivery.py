from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional, Tuple

from orion.notify.transport import EmailTransport
from orion.schemas.notify import NotificationRequest

from .policy import Policy

logger = logging.getLogger("orion.notify.email_delivery")


def should_send_email(payload: NotificationRequest) -> Tuple[bool, Optional[str]]:
    channels = payload.channels_requested or []
    if any(channel.lower() == "email" for channel in channels):
        return True, "channels_requested=email"
    severity = (payload.severity or "").lower()
    if severity in {"error", "critical"}:
        return True, f"severity={severity}"
    return False, None


def enrich_with_policy(payload: NotificationRequest, policy: Policy, now: datetime) -> NotificationRequest:
    decision = policy.evaluate(payload, now)
    context = dict(payload.context or {})
    if decision.ack_deadline_minutes is not None:
        context["ack_deadline_minutes"] = decision.ack_deadline_minutes
    if decision.escalation_channels:
        context["escalation_channels"] = decision.escalation_channels
    return payload.model_copy(update={"context": context})


def maybe_send_email(
    transport: EmailTransport | None,
    payload: NotificationRequest,
    *,
    immediate_critical_only: bool = False,
) -> None:
    if transport is None:
        return
    severity = (payload.severity or "").lower()
    if immediate_critical_only and severity != "critical":
        return
    should, reason = should_send_email(payload)
    if not should:
        return
    try:
        transport.send(payload)
        logger.info(
            "[NOTIFY] email_send_succeeded notification_id=%s event_kind=%s reason=%s",
            payload.notification_id,
            payload.event_kind,
            reason,
        )
    except Exception as exc:
        logger.error(
            "[NOTIFY] email_send_failed notification_id=%s event_kind=%s error_class=%s error=%s",
            payload.notification_id,
            payload.event_kind,
            exc.__class__.__name__,
            str(exc),
        )
