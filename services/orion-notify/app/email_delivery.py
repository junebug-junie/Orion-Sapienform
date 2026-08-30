from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class EmailOutcome:
    """What actually happened to the email for one notification.

    `status` is one of:
      "sent"    -- the SMTP transport accepted the message WITHOUT raising.
                   This is not proof it reached an inbox; nothing downstream of
                   SMTP reports back here. Do not read it as "Juniper saw it".
      "failed"  -- the send raised.
      "skipped"  -- no email was attempted and none will be: policy declined, or
                    no SMTP transport is configured.
      "deferred" -- not sent NOW, but a later path may still send it. Used for
                    the attention endpoint's `immediate_critical_only` skip:
                    `attention_escalation.py` emails exactly `severity=="error"`
                    attentions past their ack deadline. Live 2026-08-30, 37 of
                    46 error attentions escalated and every other severity
                    escalated zero times -- so stamping those "no email" would
                    make the column confidently wrong about the ONLY class that
                    reliably emails.
    """

    status: str
    reason: Optional[str]


def maybe_send_email(
    transport: EmailTransport | None,
    payload: NotificationRequest,
    *,
    immediate_critical_only: bool = False,
) -> EmailOutcome:
    """Attempt the email and RETURN what happened.

    Previously returned None: the outcome was logged and discarded, and the
    caller then wrote a hardcoded `status="pending"` onto the persisted record.
    That is why all 10,900 rows in `notify_requests` -- every one since
    2026-07-24, including emails Juniper confirmed receiving -- read "pending".
    A column that always says pending is worse than no column: a reader gets a
    wrong answer instead of no answer.
    """
    if transport is None:
        return EmailOutcome("skipped", "smtp_transport_unconfigured")
    severity = (payload.severity or "").lower()
    if immediate_critical_only and severity != "critical":
        return EmailOutcome("deferred", f"immediate_critical_only_severity_{severity or 'unset'}")
    should, reason = should_send_email(payload)
    if not should:
        return EmailOutcome("skipped", reason)
    try:
        transport.send(payload)
        logger.info(
            "[NOTIFY] email_send_succeeded notification_id=%s event_kind=%s reason=%s",
            payload.notification_id,
            payload.event_kind,
            reason,
        )
        return EmailOutcome("sent", reason)
    except Exception as exc:
        logger.error(
            "[NOTIFY] email_send_failed notification_id=%s event_kind=%s error_class=%s error=%s",
            payload.notification_id,
            payload.event_kind,
            exc.__class__.__name__,
            str(exc),
        )
        return EmailOutcome("failed", f"{exc.__class__.__name__}: {exc}"[:500])
