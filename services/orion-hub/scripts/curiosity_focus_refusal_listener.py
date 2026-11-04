"""Hub's responder for the curiosity replica's focus refusals (design 2026-11-04, section 9, patch 4).

The replica (orion-mind, services/orion-mind/app/replica_loop.py) may refuse a focus question PRIME queued,
with a one-sentence reason. It cannot write Hub's focus table itself; it asks here:

    orion:curiosity:replica:focus_refusal:request   CuriosityFocusRefusalRequestV1
    orion:curiosity:replica:focus_refusal:result:<correlation_id>   CuriosityFocusRefusalResultV1

This applies orion/curiosity/focus.py ``refuse_focus``: only a still-pending, unexpired focus Prime queued
leaves the slot (read as ``refused_by_organ``, the reason and the replica's answer act stored on the row), so no
run is shown it. A focus Juniper asked for, or one Orion chose in a conversation, is never refused: that rule is
in ``organ_refusal_blocker``, in the UPDATE's own WHERE, and in the table's CHECK. ``ok`` false means nothing
was written. Same reply rule as the introspect responders: only to exactly ``...:result:<correlation_id>``.

Only APPLIED refusals reach here; the replica keeps its held-back ones (the holdback arm) on its own act.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any, Callable

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.curiosity.focus import REFUSED_MIGRATION, OrganRefusalNotAllowed, refuse_focus
from orion.introspect.redact import safe_exception_detail
from orion.schemas.curiosity_replica import (
    FOCUS_REFUSAL_REQUEST_CHANNEL,
    FOCUS_REFUSAL_REQUEST_KIND,
    FOCUS_REFUSAL_RESULT_CHANNEL_PREFIX,
    FOCUS_REFUSAL_RESULT_KIND,
    REPLICA_ACTOR,
    CuriosityFocusRefusalRequestV1,
    CuriosityFocusRefusalResultV1,
)

logger = logging.getLogger("orion-hub.curiosity_focus_refusal")

UNAVAILABLE = "the focus table could not be written; nothing was refused"
DISABLED = "the curiosity focus queue is switched off (HUB_CURIOSITY_FOCUS_ENABLED=false)"
_UNDEFINED_COLUMN = "42703"
_UNDEFINED_TABLE = "42P01"
_ERROR_CAP = 400


class CuriosityFocusRefusalListener:
    def __init__(self, *, pool_provider: Callable[[], Any], source_ref: Any,
                 enabled: Callable[[], bool] | bool = True):
        self.pool_provider = pool_provider
        self.source_ref = source_ref
        self._enabled = enabled
        self.bus: Any = None
        self.task: asyncio.Task | None = None

    @property
    def enabled(self) -> bool:
        return bool(self._enabled() if callable(self._enabled) else self._enabled)

    async def handle(self, envelope: BaseEnvelope) -> None:
        reply = f"{FOCUS_REFUSAL_RESULT_CHANNEL_PREFIX}{envelope.correlation_id}"
        if envelope.reply_to != reply or envelope.kind != FOCUS_REFUSAL_REQUEST_KIND:
            return
        try:
            req = CuriosityFocusRefusalRequestV1.model_validate(envelope.payload)
        except ValueError as exc:
            focus_id = str((envelope.payload or {}).get("focus_id") or "?") if isinstance(envelope.payload, dict) else "?"
            await self._publish(reply, envelope, CuriosityFocusRefusalResultV1(
                focus_id=focus_id or "?", ok=False, error=f"invalid refusal request: {str(exc)[:300]}"))
            return
        result = await self.refuse(req)
        logger.info("curiosity_focus_refusal focus=%s ok=%s act=%s prime_act=%s err=%s", req.focus_id, result.ok,
                    req.act_id, req.prime_act_id, (result.error or "-")[:200])
        await self._publish(reply, envelope, result)

    async def refuse(self, req: CuriosityFocusRefusalRequestV1) -> CuriosityFocusRefusalResultV1:
        if not self.enabled:
            return CuriosityFocusRefusalResultV1(focus_id=req.focus_id, ok=False, error=DISABLED)
        pool = self.pool_provider()
        if pool is None:
            return CuriosityFocusRefusalResultV1(focus_id=req.focus_id, ok=False, error=UNAVAILABLE)
        try:
            row = await refuse_focus(pool, focus_id=req.focus_id, refused_by=REPLICA_ACTOR, reason=req.reason,
                                     act_id=req.act_id)
        except OrganRefusalNotAllowed as exc:
            return CuriosityFocusRefusalResultV1(focus_id=req.focus_id, ok=False, error=str(exc)[:_ERROR_CAP])
        except Exception as exc:  # noqa: BLE001 -- one transaction: an error means nothing was written
            if getattr(exc, "sqlstate", None) in (_UNDEFINED_COLUMN, _UNDEFINED_TABLE):
                logger.warning("curiosity_focus_refusal_columns_missing -- apply %s", REFUSED_MIGRATION)
            logger.warning("curiosity_focus_refusal_failed focus=%s exc_type=%s detail=%s", req.focus_id,
                           type(exc).__name__, safe_exception_detail(exc))
            return CuriosityFocusRefusalResultV1(focus_id=req.focus_id, ok=False, error=UNAVAILABLE)
        return CuriosityFocusRefusalResultV1(focus_id=req.focus_id, ok=True,
                                             refused_at=row.status_at or datetime.now(timezone.utc))

    async def _publish(self, reply: str, envelope: BaseEnvelope, result: CuriosityFocusRefusalResultV1) -> None:
        await self.bus.publish(reply, BaseEnvelope(
            kind=FOCUS_REFUSAL_RESULT_KIND, correlation_id=envelope.correlation_id, source=self.source_ref,
            payload=result.model_dump(mode="json"),
        ))

    async def _run(self) -> None:
        while True:
            try:
                async with self.bus.subscribe(FOCUS_REFUSAL_REQUEST_CHANNEL) as pubsub:
                    logger.info("curiosity_focus_refusal_listening channel=%s", FOCUS_REFUSAL_REQUEST_CHANNEL)
                    async for raw in self.bus.iter_messages(pubsub):
                        try:
                            decoded = self.bus.codec.decode(raw.get("data"))
                            if decoded.ok:
                                await self.handle(decoded.envelope)
                        except Exception:  # noqa: BLE001
                            logger.warning("curiosity_focus_refusal_message_failed", exc_info=True)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.warning("curiosity_focus_refusal_reconnecting", exc_info=True)
            await asyncio.sleep(1)  # never a hot loop, whether the subscription failed or just ended

    async def start(self, bus: Any) -> None:
        self.bus = bus
        self.task = asyncio.create_task(self._run(), name="hub-curiosity-focus-refusal")

    async def stop(self) -> None:
        if self.task:
            self.task.cancel()
            with suppress(asyncio.CancelledError):
                await self.task
            self.task = None


__all__ = ["CuriosityFocusRefusalListener", "DISABLED", "UNAVAILABLE"]
