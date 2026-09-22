"""Hold Hub chat while Juniper has lent her chat lane to the durable burst queue.

The `chat-burst` gateway route shares one physical worker (circe-worker-1) with `chat`.
When the operator opens its gate from the Hub ("Lend chat lane" button), that worker is
busy serving queued durable runs, so a message typed into the Hub chat box must not be
sent to cortex. Instead the message stays in chat history as usual and is emailed to
Juniper through orion-notify so nothing is lost.

Two entry points, used by both the WebSocket and HTTP chat paths:

- `chat_lane_is_lent()` -- is the gate open right now? Cached ~3 s so a burst of
  messages does not hammer the gateway. Any failure reads as "not lent" (normal chat).
- `hold_chat_message_for_email(...)` -- email the held message; returns whether notify
  accepted it.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone

from orion.notify.client import NotifyClient
from orion.schemas.notify import NotificationRequest

from scripts.llm_gateway_client import fetch_route_gate
from scripts.settings import settings

logger = logging.getLogger("orion-hub.chat-lane-lend")

GATE_ROUTE_ID = "chat-burst"
CACHE_TTL_SEC = 3.0

#: Single source of the user-facing notice for both the WebSocket and HTTP transports.
HELD_NOTICE_TEXT = (
    "Orion's chat lane is currently lent to the burst queue. "
    "Your message was saved and emailed to Juniper; it was not sent to Orion."
)
EMAIL_FAILED_SUFFIX = " (email delivery failed)"

# Module-level cache: last answer plus the monotonic time it was read.
_cached_lent: bool = False
_cached_at: float | None = None
# Counts consecutive gate-read failures so the warning is logged once per streak, not
# once per message.
_failure_streak: int = 0


def reset_cache() -> None:
    """Forget the cached gate answer (tests, and after the operator flips the gate)."""
    global _cached_lent, _cached_at, _failure_streak
    _cached_lent = False
    _cached_at = None
    _failure_streak = 0


def held_notice_text(emailed: bool) -> str:
    """The notice to show in the transcript, with a suffix when the email did not go out."""
    return HELD_NOTICE_TEXT if emailed else HELD_NOTICE_TEXT + EMAIL_FAILED_SUFFIX


async def chat_lane_is_lent() -> bool:
    """True while the chat-burst gate is open. False on any error (gateway unreachable,
    route not gated, bad payload) so a gateway outage never blocks normal chat."""
    global _cached_lent, _cached_at, _failure_streak
    now = time.monotonic()
    if _cached_at is not None and (now - _cached_at) < CACHE_TTL_SEC:
        return _cached_lent
    try:
        gate = await fetch_route_gate(GATE_ROUTE_ID)
        lent = bool(gate.get("open"))
        if _failure_streak:
            logger.info("chat_lane_gate_read_recovered after %s failures", _failure_streak)
        _failure_streak = 0
    except Exception as exc:  # any failure means "treat the lane as not lent"
        if _failure_streak == 0:
            logger.warning("chat_lane_gate_read_failed route=%s err=%s", GATE_ROUTE_ID, exc)
        _failure_streak += 1
        lent = False
    _cached_lent = lent
    _cached_at = now
    return lent


def build_held_message_request(
    *,
    text: str,
    session_id: str,
    correlation_id: str,
    mode: str,
    speaker: str,
    now: datetime | None = None,
) -> NotificationRequest:
    """The email body: when, which session/mode/speaker, then the message verbatim."""
    stamp = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    body = (
        f"Held at: {stamp}\n"
        f"Session: {session_id}\n"
        f"Mode: {mode}\n"
        f"Speaker: {speaker}\n"
        f"Correlation: {correlation_id}\n"
        "\n"
        "Message:\n"
        f"{text}\n"
    )
    return NotificationRequest(
        source_service="orion-hub",
        event_kind="orion.hub.chat.lane_lent",
        severity="info",
        title="Hub chat held: chat lane is lent to burst",
        body_text=body,
        body_md=body,
        recipient_group="juniper_primary",
        channels_requested=["email"],
        dedupe_key=f"hub-chat-lent:{correlation_id}",
        correlation_id=correlation_id,
        session_id=session_id,
    )


async def hold_chat_message_for_email(
    *,
    text: str,
    session_id: str,
    correlation_id: str,
    mode: str,
    speaker: str,
) -> bool:
    """Email a held chat message to Juniper via orion-notify. Returns True only if
    notify accepted it. NotifyClient is blocking `requests`, so it runs in a thread."""
    base_url = str(getattr(settings, "NOTIFY_BASE_URL", "") or "").strip()
    if not base_url:
        logger.warning("chat_lane_hold_email_skipped corr=%s reason=NOTIFY_BASE_URL_unset", correlation_id)
        return False
    request = build_held_message_request(
        text=text,
        session_id=session_id,
        correlation_id=correlation_id,
        mode=mode,
        speaker=speaker,
    )
    client = NotifyClient(base_url, getattr(settings, "NOTIFY_API_TOKEN", None) or None)
    try:
        accepted = await asyncio.to_thread(client.send, request)
    except Exception as exc:
        logger.warning("chat_lane_hold_email_failed corr=%s err=%s", correlation_id, exc)
        return False
    ok = bool(getattr(accepted, "ok", False))
    logger.info(
        "chat_lane_hold_email_result corr=%s session=%s ok=%s detail=%s",
        correlation_id,
        session_id,
        ok,
        getattr(accepted, "detail", None),
    )
    return ok
