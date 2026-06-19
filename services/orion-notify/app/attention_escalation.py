from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Dict, Optional
from uuid import uuid4

from orion.schemas.notify import NotificationRequest

from .email_delivery import maybe_send_email
from .policy import Policy
from .settings import settings as notify_settings

logger = logging.getLogger("orion.notify.attention_escalation")

ATTENTION_ESCALATION_EVENT_KIND = "orion.chat.attention.escalation"

ProxyGet = Callable[..., Awaitable[Any]]
ProxyPost = Callable[..., Awaitable[Any]]


def _parse_ts(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        ts = value
    elif isinstance(value, str):
        try:
            ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts


def _hub_link(hub_url_base: str, attention_id: str) -> Optional[str]:
    base = hub_url_base.strip().rstrip("/")
    if not base:
        return None
    return f"{base}/#attention"


def _build_escalation_request(row: Dict[str, Any], *, hub_url_base: str) -> NotificationRequest:
    attention_id = str(row.get("attention_id") or "")
    context = dict(row.get("context") or {})
    hub_url = _hub_link(hub_url_base, attention_id)
    body_lines = [
        "Attention request was not acknowledged before the escalation deadline.",
        "",
        str(row.get("message") or ""),
    ]
    if hub_url:
        body_lines.extend(["", f"Hub: {hub_url}"])
    return NotificationRequest(
        notification_id=uuid4(),
        source_service=str(row.get("source_service") or "orion-notify"),
        event_kind=ATTENTION_ESCALATION_EVENT_KIND,
        severity="error",
        title=f"Escalation: {row.get('reason') or 'attention_request'}",
        body_text="\n".join(body_lines),
        context={**context, "attention_id": attention_id, "escalated": True},
        tags=["chat", "attention", "escalation"],
        recipient_group="juniper_primary",
        channels_requested=["email"],
        dedupe_key=f"attention-escalation:{attention_id}",
        dedupe_window_seconds=3600,
        correlation_id=row.get("correlation_id"),
        session_id=row.get("session_id"),
    )


async def run_attention_escalation_once(
    *,
    email_transport,
    policy: Policy,
    proxy_get: ProxyGet,
    proxy_post: ProxyPost,
    hub_url_base: str,
) -> int:
    rows = await proxy_get("/attention", params={"status": "pending", "limit": 100})
    if not isinstance(rows, list):
        return 0

    now = datetime.now(timezone.utc)
    sent = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        if (row.get("severity") or "").lower() != "error":
            continue
        if row.get("escalated_at"):
            continue
        if not row.get("require_ack") or row.get("acked_at"):
            continue

        deadline_min = row.get("ack_deadline_minutes")
        if deadline_min is None:
            notify_payload = NotificationRequest(
                source_service=str(row.get("source_service") or "unknown"),
                event_kind="orion.chat.attention",
                severity="error",
                title=str(row.get("reason") or "attention_request"),
                body_text=str(row.get("message") or ""),
                context=dict(row.get("context") or {}),
            )
            deadline_min = policy.evaluate(notify_payload, now.replace(tzinfo=None)).ack_deadline_minutes or 60

        created = _parse_ts(row.get("created_at"))
        if created is None:
            continue
        if now < created + timedelta(minutes=int(deadline_min)):
            continue

        channels = (row.get("context") or {}).get("escalation_channels") or ["email"]
        if "email" not in [str(c).lower() for c in channels]:
            continue

        attention_id = str(row.get("attention_id") or "")
        if not attention_id:
            continue

        escalation = _build_escalation_request(row, hub_url_base=hub_url_base)
        maybe_send_email(email_transport, escalation)
        await proxy_post(f"/attention/{attention_id}/escalate", {})
        sent += 1
        logger.info("[NOTIFY] attention_escalated attention_id=%s", attention_id)

    return sent


async def attention_escalation_loop(app) -> None:
    poll_seconds = max(int(notify_settings.NOTIFY_ESCALATION_POLL_SECONDS), 10)
    hub_url_base = notify_settings.NOTIFY_HUB_URL or ""
    while True:
        try:
            count = await run_attention_escalation_once(
                email_transport=getattr(app.state, "email_transport", None),
                policy=app.state.policy,
                proxy_get=app.state.proxy_get,
                proxy_post=app.state.proxy_post,
                hub_url_base=hub_url_base,
            )
            if count:
                logger.info("[NOTIFY] attention_escalation_tick sent=%s", count)
        except Exception as exc:
            logger.warning("[NOTIFY] attention_escalation_tick failed: %s", exc, exc_info=True)
        await asyncio.sleep(poll_seconds)
