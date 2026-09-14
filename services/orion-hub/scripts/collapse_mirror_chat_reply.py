"""Hub: juniper Collapse Mirror → live chat You bubble + chat-lane reply.

Design: docs/superpowers/specs/2026-09-14-collapse-mirror-chat-lane-reply-design.md
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.collapse_mirror_chat_reply import (
    COLLAPSE_MIRROR_CHAT_REPLY_CHANNEL,
    CollapseMirrorChatReplyRequestV1,
)
from orion.schemas.notify import HubNotificationEvent

logger = logging.getLogger("orion-hub.collapse_mirror_chat_reply")

YOU_KIND = "collapse_mirror_you"
SOURCE_TAG = "collapse_mirror_reply"
DEFAULT_TURN_TIMEOUT_SEC = 120.0


class CollapseMirrorChatReplyHandler:
    def __init__(
        self,
        *,
        outreach: Any,
        bus: Any = None,
        channel: str = COLLAPSE_MIRROR_CHAT_REPLY_CHANNEL,
        dedupe_ttl_seconds: int = 86400,
        turn_timeout_sec: float = DEFAULT_TURN_TIMEOUT_SEC,
    ) -> None:
        self._outreach = outreach
        self._bus = bus
        self._channel = channel
        self._dedupe_ttl = int(dedupe_ttl_seconds)
        self._turn_timeout_sec = float(turn_timeout_sec)
        self._done_expiry: dict[str, float] = {}
        self._task: Optional[asyncio.Task] = None
        self._stopping = False
        # `_deliver` persists via outreach._bus; keep it aligned when tests/wiring
        # pass the bus only to this handler.
        if bus is not None and getattr(outreach, "_bus", None) is None:
            outreach._bus = bus

    def _prune(self, now: float) -> None:
        expired = [k for k, exp in self._done_expiry.items() if exp <= now]
        for k in expired:
            self._done_expiry.pop(k, None)

    def _try_claim(self, event_id: str) -> bool:
        now = time.time()
        self._prune(now)
        if event_id in self._done_expiry:
            return False
        self._done_expiry[event_id] = now + self._dedupe_ttl
        return True

    async def start(self, bus: Any) -> None:
        self._bus = bus
        if getattr(self._outreach, "_bus", None) is None:
            self._outreach._bus = bus
        self._stopping = False
        self._task = asyncio.create_task(self._consume_loop(), name="collapse_mirror_chat_reply")

    async def stop(self) -> None:
        self._stopping = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _consume_loop(self) -> None:
        assert self._bus is not None
        while not self._stopping:
            try:
                async with self._bus.subscribe(self._channel) as pubsub:
                    async for msg in self._bus.iter_messages(pubsub):
                        if self._stopping:
                            break
                        try:
                            decoded = self._bus.codec.decode(msg.get("data"))
                            env = (
                                BaseEnvelope.model_validate(decoded)
                                if not isinstance(decoded, BaseEnvelope)
                                else decoded
                            )
                            await self.handle(env)
                        except Exception:  # noqa: BLE001
                            logger.exception("collapse_mirror_chat_reply_handle_failed")
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.exception("collapse_mirror_chat_reply_subscribe_failed")
                await asyncio.sleep(1.0)

    async def handle(self, env: BaseEnvelope) -> dict:
        try:
            req = CollapseMirrorChatReplyRequestV1.model_validate(env.payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning("collapse_mirror_chat_reply_invalid_payload err=%s", exc)
            return {"status": "skipped", "reason": "invalid_payload"}

        event_id = req.event_id
        if not self._try_claim(event_id):
            logger.info("collapse_mirror_chat_reply_deduped event_id=%s", event_id)
            return {"status": "skipped", "reason": "deduped", "event_id": event_id}

        session_id = self._outreach.live_session_id()
        if not session_id:
            # Keep the claim — redelivery stays skipped for this event_id until TTL.
            await self._optional_held_back_notify(
                event_id=event_id, correlation_id=str(env.correlation_id)
            )
            logger.info(
                "collapse_mirror_chat_reply_skipped reason=no_live_session event_id=%s",
                event_id,
            )
            return {"status": "skipped", "reason": "no_live_session", "event_id": event_id}

        correlation_id = str(env.correlation_id)
        await self._inject_you(
            text=req.mirror_text,
            session_id=session_id,
            correlation_id=correlation_id,
        )

        try:
            text, debug = await self._run_chat_lane_turn(
                user_message=req.mirror_text,
                session_id=session_id,
                correlation_id=correlation_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "collapse_mirror_chat_reply_turn_failed event_id=%s corr=%s",
                event_id,
                correlation_id,
            )
            return {
                "status": "failed",
                "reason": str(exc),
                "event_id": event_id,
                "session_id": session_id,
                "you_written": True,
            }

        if not text or not str(text).strip():
            logger.info(
                "collapse_mirror_chat_reply_empty_reply event_id=%s corr=%s",
                event_id,
                correlation_id,
            )
            return {
                "status": "failed",
                "reason": "empty_reply",
                "event_id": event_id,
                "session_id": session_id,
                "you_written": True,
            }

        await self._outreach._deliver(
            text=str(text).strip(),
            session_id=session_id,
            correlation_id=correlation_id,
            model=debug.get("fcc_model_label") if isinstance(debug, dict) else None,
            source_tag=SOURCE_TAG,
        )
        return {
            "status": "delivered",
            "event_id": event_id,
            "session_id": session_id,
            "you_written": True,
        }

    async def _inject_you(self, *, text: str, session_id: str, correlation_id: str) -> None:
        message_id = str(uuid4())
        payload = {
            "kind": YOU_KIND,
            "text": text,
            "correlation_id": correlation_id,
            "message_id": message_id,
            "session_id": session_id,
        }
        for connection_id, entry in list(self._outreach._connections.items()):
            queue = entry.get("queue")
            if queue is None:
                continue
            try:
                queue.put_nowait(dict(payload))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "collapse_mirror_you_push_failed connection=%s err=%s",
                    connection_id,
                    exc,
                )
        try:
            from scripts.chat_history import build_chat_history_envelope, publish_chat_history

            env = build_chat_history_envelope(
                content=text,
                role="user",
                session_id=session_id,
                correlation_id=correlation_id,
                speaker="Juniper",
                tags=[SOURCE_TAG],
                message_id=message_id,
                client_meta={"collapse_mirror_you": True},
            )
            await publish_chat_history(self._bus or self._outreach._bus, [env])
        except Exception as exc:  # noqa: BLE001
            logger.warning("collapse_mirror_you_history_failed corr=%s err=%s", correlation_id, exc)

    async def _run_chat_lane_turn(
        self,
        *,
        user_message: str,
        session_id: str,
        correlation_id: str,
    ) -> tuple[str, dict]:
        # Lazy import matches endogenous_outreach._attempt_unified_turn so tests
        # can patch orion.hub.turn_orchestrator.execute_unified_turn.
        from orion.hub.turn_orchestrator import execute_unified_turn

        request_payload: dict[str, Any] = {
            "no_write": True,
            "source": SOURCE_TAG,
        }
        # Intentionally omit fcc_model_label → chat lane.
        bus = self._bus or self._outreach._bus
        frames = await asyncio.wait_for(
            execute_unified_turn(
                bus=bus,
                correlation_id=correlation_id,
                session_id=session_id,
                user_message=user_message,
                payload=request_payload,
                continuity_messages=None,
                harness_rpc_bus=getattr(self._outreach, "_harness_rpc_bus", None) or bus,
                harness_step_relay=None,
                harness_step_queue=None,
            ),
            timeout=self._turn_timeout_sec,
        )
        final = frames[-1] if frames else {}
        text = str((final or {}).get("llm_response") or "").strip()
        debug = {
            "fcc_model_label": (final or {}).get("fcc_model_label"),
        }
        return text, debug

    async def _optional_held_back_notify(self, *, event_id: str, correlation_id: str) -> None:
        """Quiet note that a reply was held back — never the reply body itself."""
        try:
            bus = self._bus or getattr(self._outreach, "_bus", None)
            channel = getattr(self._outreach, "notify_channel", None)
            if not bus or not channel:
                return
            notification = HubNotificationEvent(
                notification_id=uuid4(),
                created_at=datetime.now(timezone.utc),
                severity="info",
                event_kind="orion.chat.message",
                source_service="orion-hub",
                title="Collapse Mirror reply held back",
                body_text=(
                    "No live Hub chat session was connected, so Orion did not reply in chat."
                ),
                tags=[SOURCE_TAG, "held_back"],
                correlation_id=correlation_id,
                notification_type="collapse_mirror_held_back",
                silent=True,
            )
            env = BaseEnvelope(
                kind="notify.in_app.v1",
                source=ServiceRef(name="orion-hub"),
                correlation_id=correlation_id,
                payload=notification.model_dump(mode="json"),
            )
            await bus.publish(channel, env)
        except Exception:  # noqa: BLE001
            logger.debug(
                "collapse_mirror_held_back_notify_failed event_id=%s",
                event_id,
                exc_info=True,
            )
