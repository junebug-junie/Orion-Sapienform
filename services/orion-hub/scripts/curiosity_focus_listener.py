"""Hub's introspect responder for the `queue_curiosity_focus` tool.

Juniper told Orion "it's your choice what you focus on; queue it up in your curiosity run" and
Orion had no way to do it (2026-10-16). This responder is that way: one
``curiosity_focus_request`` row (orion/curiosity/focus.py), offered once to the next regular
investigate run. Design: docs/superpowers/specs/2026-10-16-orion-queue-curiosity-focus-design.md.

Before anything is written:
* the request is a chat turn (curiosity runs already carry their own continuation note);
* the focus queue is on (HUB_CURIOSITY_FOCUS_ENABLED);
* a pending focus Juniper asked for is never replaced by Orion's own choice.

Any refusal is ok=false and nothing is written; the queue is one transaction, so ok=false
always means "not queued". Same reply rules as the other introspect responders: only to exactly
orion:introspect:result:<corr>, and only for the request kind. Orion's graph is never touched.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.curiosity.focus import FOCUS_TTL, MIGRATION, FocusRefused, FocusRow, queue_focus
from orion.introspect.redact import safe_exception_detail
from orion.introspect.transport import (
    QUEUE_CURIOSITY_FOCUS_REQUEST_CHANNEL,
    REQUEST_KIND,
    RESULT_KIND,
    RESULT_PREFIX,
)
from orion.schemas.introspect import (
    IntrospectItemV1,
    IntrospectRequestV1,
    IntrospectResultV1,
    QueueCuriosityFocusArguments,
)

logger = logging.getLogger("orion-hub.curiosity_focus")

OPERATION = "queue_curiosity_focus"
# Not "not queued": a connection lost after the commit lands here too.
UNAVAILABLE = "queue_curiosity_focus_unavailable; answer unknown"
DISABLED = "the curiosity focus queue is switched off (HUB_CURIOSITY_FOCUS_ENABLED=false)"
_INVALID_CAP = 400
_UNDEFINED_TABLE = "42P01"


def _failed(now: datetime, error: str) -> IntrospectResultV1:
    return IntrospectResultV1(ok=False, operation=OPERATION, as_of=now, error=error)


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value is not None else None


def next_run_words(schedule: Optional[dict[str, Any]]) -> dict[str, Any]:
    """When the next regular investigate run can start, from the Curiosity tab's own schedule
    read (cooldown stamp and daily counter in Redis). Never claims "now" for a line at cap."""
    line = ((schedule or {}).get("lines") or {}).get("investigate") or {}
    if not (schedule or {}).get("available") or not line:
        return {"known": False, "words": "the schedule could not be read; the next run time is unknown"}
    out: dict[str, Any] = {
        "known": True,
        "enabled": line.get("enabled"),
        "runs_today": line.get("runs_today"),
        "daily_cap": line.get("daily_cap"),
        "next_eligible_at": line.get("next_eligible_at"),
        "tz": schedule.get("tz"),
    }
    runs, cap = line.get("runs_today"), line.get("daily_cap")
    if line.get("enabled") is False:
        out["words"] = "the investigate line is switched off; no run will take this until it is on"
    elif isinstance(runs, int) and isinstance(cap, int) and cap > 0 and runs >= cap:
        out["words"] = (
            f"today's investigate runs are used up ({runs} of {cap}); the next run can start after "
            f"midnight {schedule.get('tz') or 'local time'}"
        )
    elif line.get("next_eligible_at"):
        out["words"] = (
            f"the next run can start no earlier than {line['next_eligible_at']} (cooldown), "
            "if the other gates pass"
        )
    else:
        out["words"] = "the next run can start at the next scheduler tick, if the gates pass"
    return out


def _focus_extra(focus: FocusRow) -> dict[str, Any]:
    return {
        "focus_id": focus.focus_id,
        "status": focus.status,
        "origin": focus.origin,
        "question": focus.question,
        "why": focus.why,
        "queued_at": _iso(focus.created_at),
        "expires_at": _iso(focus.expires_at),
        "offered_run_id": focus.offered_run_id,
    }


class CuriosityFocusListener:
    def __init__(
        self, *, pool_provider: Callable[[], Any], source_ref: Any,
        enabled: Callable[[], bool] | bool = True,
        schedule_provider: Optional[Callable[[], Awaitable[dict[str, Any]]]] = None,
    ):
        self.pool_provider = pool_provider
        self.source_ref = source_ref
        self._enabled = enabled
        self.schedule_provider = schedule_provider
        self.bus: Any = None
        self.task: asyncio.Task | None = None

    @property
    def enabled(self) -> bool:
        return bool(self._enabled() if callable(self._enabled) else self._enabled)

    async def handle(self, envelope: BaseEnvelope) -> None:
        reply = f"{RESULT_PREFIX}{envelope.correlation_id}"
        if envelope.reply_to != reply or envelope.kind != REQUEST_KIND:
            return
        now = datetime.now(timezone.utc)
        try:
            request = IntrospectRequestV1.model_validate(envelope.payload)
            if request.operation != OPERATION:
                raise ValueError(f"operation {request.operation!r} is not answered here")
            args = QueueCuriosityFocusArguments.model_validate(request.args)
        except ValueError as exc:
            await self._publish(reply, envelope, _failed(now, f"invalid queue_curiosity_focus request: {str(exc)[:_INVALID_CAP]}"))
            return
        try:
            result = await self.queue(request, args, now)
        except FocusRefused as exc:
            logger.info("curiosity_focus_refused corr=%s origin=%s detail=%s",
                        envelope.correlation_id, args.origin, str(exc)[:_INVALID_CAP])
            result = _failed(now, str(exc)[:_INVALID_CAP])
        except Exception as exc:
            if getattr(exc, "sqlstate", None) == _UNDEFINED_TABLE:
                logger.warning("curiosity_focus_table_missing corr=%s -- apply %s", envelope.correlation_id, MIGRATION)
            logger.warning(
                "introspect_failed op=queue_curiosity_focus corr=%s exc_type=%s detail=%s",
                envelope.correlation_id, type(exc).__name__, safe_exception_detail(exc),
            )
            result = _failed(now, UNAVAILABLE)
        else:
            extra = result.items[0].extra
            logger.info(
                "curiosity_focus_queued focus_id=%s corr=%s chat_corr=%s origin=%s replaced=%s already_queued=%s",
                extra["focus_id"], envelope.correlation_id, extra.get("chat_correlation_id"), args.origin,
                (extra.get("replaced") or {}).get("focus_id"), extra.get("already_queued"),
            )
        await self._publish(reply, envelope, result)

    async def queue(
        self, request: IntrospectRequestV1, args: QueueCuriosityFocusArguments, now: datetime,
    ) -> IntrospectResultV1:
        binding = request.binding
        if binding.invocation_context != "unified_chat":
            raise FocusRefused(
                "a focus is queued from a conversation with Juniper; this is a "
                f"{binding.invocation_context} turn (a curiosity run already has its continuation note)"
            )
        if not self.enabled:
            raise FocusRefused(DISABLED)
        pool = self.pool_provider()
        if pool is None:
            raise RuntimeError("no memory pool")
        chat_corr = binding.parent_trace_id or None
        queued = await queue_focus(
            pool, question=args.question, why=args.why, origin=args.origin, chat_correlation_id=chat_corr,
        )
        schedule = None
        if self.schedule_provider is not None:
            try:
                schedule = await self.schedule_provider()
            except Exception as exc:  # noqa: BLE001 -- the focus is queued either way
                logger.warning("curiosity_focus_schedule_unreadable err=%s", exc)
        focus = queued.focus
        replaced = queued.replaced
        extra = {
            **_focus_extra(focus),
            "chat_correlation_id": chat_corr,
            "already_queued": queued.already_queued,
            "replaced": None if replaced is None else {
                "focus_id": replaced.focus_id, "question": replaced.question, "origin": replaced.origin,
            },
            "next_run": next_run_words(schedule),
            "what_happens_next": (
                "nothing starts now. The next regular investigate run that passes its normal gates "
                "is shown this focus first and decides whether to explore it or set it aside; that "
                f"decision is recorded on the focus. Unless a run takes it, it expires after "
                f"{int(FOCUS_TTL.total_seconds() // 3600)} hours. curiosity(kind=focus) shows where it is."
            ),
        }
        if queued.already_queued:
            extra["what_happens_next"] = (
                f"this exact focus was already queued in this conversation; its status now is "
                f"{focus.status}. Nothing new was written."
            )
        item = IntrospectItemV1(
            id=focus.focus_id,
            occurred_at=focus.created_at or now,
            kind="curiosity_focus",
            epistemic_status="record",
            text=focus.question,
            extra=extra,
        )
        return IntrospectResultV1(ok=True, operation=OPERATION, as_of=now, total_available=1, items=[item])

    async def _publish(self, reply: str, envelope: BaseEnvelope, result: IntrospectResultV1) -> None:
        await self.bus.publish(reply, BaseEnvelope(
            kind=RESULT_KIND, correlation_id=envelope.correlation_id,
            source=self.source_ref, payload=result.model_dump(mode="json"),
        ))

    async def _run(self) -> None:
        while True:
            try:
                async with self.bus.subscribe(QUEUE_CURIOSITY_FOCUS_REQUEST_CHANNEL) as pubsub:
                    logger.info("curiosity_focus_listening channel=%s", QUEUE_CURIOSITY_FOCUS_REQUEST_CHANNEL)
                    async for raw in self.bus.iter_messages(pubsub):
                        try:
                            decoded = self.bus.codec.decode(raw.get("data"))
                            if decoded.ok:
                                await self.handle(decoded.envelope)
                        except Exception:
                            logger.warning("curiosity_focus_message_failed", exc_info=True)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("curiosity_focus_reconnecting", exc_info=True)
                await asyncio.sleep(1)

    async def start(self, bus: Any) -> None:
        self.bus = bus
        self.task = asyncio.create_task(self._run(), name="hub-curiosity-focus")

    async def stop(self) -> None:
        if self.task:
            self.task.cancel()
            with suppress(asyncio.CancelledError):
                await self.task
            self.task = None
