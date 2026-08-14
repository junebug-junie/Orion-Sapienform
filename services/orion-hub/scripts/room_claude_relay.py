"""Hub side of the three-way room: invite Claude, relay what it says.

Hub never spawns `claude` and never holds the Claude credential -- that lives
in `orion-room-companion`, deliberately, because Hub runs as root with SSH
keys, a gh token and the docker socket, all readable by Orion's own FCC turns.
See `services/orion-room-companion/README.md` for why, and for what that
separation does and does not buy.

So this module is only two halves of a bus round-trip:

    invite()  -> publish RoomClaudeRequestV1  on orion:room:claude:request
    _consume  <- consume RoomClaudeUtteranceV1 on orion:room:claude:utterance
                 -> push a frame into every live socket + persist the turn

The socket-injection half is modelled on `endogenous_outreach.py`, which
already solves the same problem (get an unsolicited message into an open
chat) including the busy/idle guards that stop a bubble landing mid-turn.

v1 is explicit-invite only: the only caller of `invite()` is an operator
route. Orion inviting Claude itself is v2 and is precisely the change that
makes budget enforcement matter -- see the README's phase-2 watchdog section.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.room_claude import (
    RoomClaudeRequestV1,
    RoomClaudeUtteranceV1,
    RoomTranscriptEntryV1,
)

logger = logging.getLogger("orion-hub.room_claude_relay")

# Client dispatches on this, same as OUTREACH_KIND. A distinct kind (rather
# than reusing the outreach one) so the UI can render Claude as its own
# speaker instead of another Orion bubble.
ROOM_CLAUDE_KIND = "room_claude_utterance"
ROOM_CLAUDE_TAG = "room_claude"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RoomClaudeRelay:
    def __init__(
        self,
        *,
        request_channel: str,
        utterance_channel: str,
        participant_name: str = "Claude",
        service_name: str = "orion-hub",
        service_version: str = "0.1.0",
        node_name: str = "athena",
        enabled: bool = False,
    ) -> None:
        self.request_channel = request_channel
        self.utterance_channel = utterance_channel
        self.participant_name = participant_name
        self.enabled = enabled
        self._service_name = service_name
        self._service_version = service_version
        self._node_name = node_name

        self._bus: Any = None
        self._task: Optional[asyncio.Task] = None
        self._connections: Dict[str, Dict[str, Any]] = {}
        # request_id -> the session that asked, so a reply lands back in the
        # conversation it belongs to rather than being broadcast blindly.
        self._pending: Dict[str, Dict[str, Any]] = {}

    # -- connection registry (called from websocket_handler) ---------------

    def register_connection(self, connection_id: str, queue: asyncio.Queue) -> None:
        self._connections[connection_id] = {"queue": queue, "session_id": None}

    def unregister_connection(self, connection_id: str) -> None:
        self._connections.pop(connection_id, None)

    def note_session(self, connection_id: str, session_id: Optional[str]) -> None:
        entry = self._connections.get(connection_id)
        if entry is not None and session_id:
            entry["session_id"] = session_id

    # -- lifecycle ---------------------------------------------------------

    async def start(self, bus: Any) -> None:
        self._bus = bus
        if not self.enabled:
            logger.info("room_claude_relay_disabled")
            return
        self._task = asyncio.create_task(self._consume_loop())
        logger.info(
            "room_claude_relay_started request=%s utterance=%s",
            self.request_channel, self.utterance_channel,
        )

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None

    # -- outbound: invite --------------------------------------------------

    async def invite(
        self,
        *,
        prompt: str,
        invited_by: str,
        session_id: Optional[str],
        room_id: str,
        correlation_id: Optional[str] = None,
        transcript: Optional[List[Dict[str, Any]]] = None,
        social_memory_summary: Optional[Dict[str, Any]] = None,
    ) -> RoomClaudeRequestV1:
        """Ask the companion for one Claude turn. Returns the request so the
        caller can correlate; the reply arrives asynchronously on the bus."""
        entries: List[RoomTranscriptEntryV1] = []
        for raw in transcript or []:
            try:
                entries.append(RoomTranscriptEntryV1.model_validate(raw))
            except Exception:
                # One malformed history line must not cost the whole invite --
                # the transcript is first-turn seeding, not the memory itself.
                logger.debug("room_claude_relay_bad_transcript_entry", exc_info=True)

        request = RoomClaudeRequestV1(
            room_id=room_id,
            session_id=session_id,
            invited_by=invited_by,
            correlation_id=correlation_id or str(uuid4()),
            prompt=prompt,
            transcript=entries,
            social_memory_summary=social_memory_summary or {},
        )
        self._pending[request.request_id] = {
            "session_id": session_id,
            "invited_at": _utcnow_iso(),
        }
        await self._publish(self.request_channel, "room.claude.request.v1", request)
        logger.info(
            "room_claude_invite room=%s request=%s session=%s by=%s",
            request.room_id, request.request_id, session_id, invited_by,
        )
        return request

    # -- inbound: relay ----------------------------------------------------

    async def _consume_loop(self) -> None:
        while True:
            try:
                async with self._bus.subscribe(self.utterance_channel) as pubsub:
                    async for msg in self._bus.iter_messages(pubsub):
                        try:
                            raw = msg.get("data")
                            body = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8"))
                        except Exception:
                            logger.warning("room_claude_relay_bad_envelope")
                            continue
                        try:
                            await self._handle_utterance(body)
                        except Exception:
                            logger.exception("room_claude_relay_handle_failed")
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                # A dropped bus connection must not silently end the room.
                logger.warning("room_claude_relay_subscribe_failed err=%s; retrying in 5s", exc)
                await asyncio.sleep(5)

    async def _handle_utterance(self, body: Any) -> None:
        # A malformed envelope can parse to a bare string or list, not just a
        # wrong-shaped dict -- guard the type before reaching for .get().
        if not isinstance(body, dict):
            logger.warning("room_claude_relay_non_dict_envelope type=%s", type(body).__name__)
            return
        payload = body.get("payload") if isinstance(body.get("payload"), dict) else body
        try:
            utterance = RoomClaudeUtteranceV1.model_validate(payload)
        except Exception:
            logger.warning("room_claude_relay_bad_utterance_payload")
            return

        pending = self._pending.pop(utterance.request_id, None)
        session_id = utterance.session_id or (pending or {}).get("session_id")

        if not utterance.ok:
            # A failed turn is surfaced, not swallowed. Silence here is
            # indistinguishable from Claude choosing not to speak, which is
            # exactly what makes an outage invisible.
            logger.warning(
                "room_claude_relay_turn_failed request=%s error=%s",
                utterance.request_id, utterance.error,
            )
            self._push(
                text=f"[Claude could not reply: {utterance.error or 'unknown error'}]",
                utterance=utterance,
                session_id=session_id,
                ok=False,
            )
            return

        logger.info(
            "room_claude_relay_utterance request=%s model=%s cost_usd=%.6f chars=%d",
            utterance.request_id, utterance.model, utterance.cost_usd, len(utterance.text),
        )
        self._push(text=utterance.text, utterance=utterance, session_id=session_id, ok=True)
        await self._publish_history(utterance, session_id=session_id)

    def _push(
        self, *, text: str, utterance: RoomClaudeUtteranceV1, session_id: Optional[str], ok: bool
    ) -> None:
        """Fan out to live sockets.

        Deliberately omits `state` and the recall/routing debug keys, same as
        endogenous_outreach: a Claude bubble must not stomp the panels showing
        the last real Orion turn.
        """
        frame = {
            "kind": ROOM_CLAUDE_KIND,
            "speaker": utterance.responder.participant_name or self.participant_name,
            "speaker_kind": utterance.responder.participant_kind,
            "llm_response": text,
            "ok": ok,
            "correlation_id": utterance.correlation_id,
            "request_id": utterance.request_id,
            "message_id": utterance.utterance_id,
            "session_id": session_id,
            "model": utterance.model,
            "cost_usd": utterance.cost_usd,
            "duration_ms": utterance.duration_ms,
        }
        for connection_id, entry in list(self._connections.items()):
            # Only the session that invited Claude sees the reply. A room is a
            # conversation, not a broadcast; without this every open tab would
            # get someone else's answer.
            if session_id and entry.get("session_id") and entry["session_id"] != session_id:
                continue
            queue = entry.get("queue")
            if queue is None:
                continue
            try:
                queue.put_nowait(dict(frame))
            except asyncio.QueueFull:
                logger.warning("room_claude_relay_queue_full connection=%s", connection_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("room_claude_relay_push_failed connection=%s err=%s", connection_id, exc)

    async def _publish_history(
        self, utterance: RoomClaudeUtteranceV1, *, session_id: Optional[str]
    ) -> None:
        """Persist Claude's turn with the responder identity attached.

        `external_responder` is the field this feature adds: every stored room
        turn until now named who spoke *to* Orion and assumed Orion answered,
        which cannot express a third participant replying directly.
        """
        try:
            from scripts.chat_history import build_chat_history_envelope, publish_chat_history

            env = build_chat_history_envelope(
                content=utterance.text,
                role="assistant",
                session_id=session_id,
                correlation_id=utterance.correlation_id,
                speaker=utterance.responder.participant_name or self.participant_name,
                model=utterance.model,
                tags=[ROOM_CLAUDE_TAG],
                message_id=utterance.utterance_id,
                client_meta={
                    "external_responder": utterance.responder.model_dump(mode="json"),
                    "external_room": {"platform": utterance.platform, "room_id": utterance.room_id},
                    "room_claude": {
                        "request_id": utterance.request_id,
                        "claude_session_id": utterance.claude_session_id,
                        "cost_usd": utterance.cost_usd,
                        "duration_ms": utterance.duration_ms,
                    },
                },
            )
            await publish_chat_history(self._bus, [env])
        except Exception as exc:  # noqa: BLE001
            logger.warning("room_claude_relay_history_failed request=%s err=%s", utterance.request_id, exc)

    # -- plumbing ----------------------------------------------------------

    async def _publish(self, channel: str, kind: str, payload: Any) -> None:
        if self._bus is None or not getattr(self._bus, "enabled", False):
            logger.warning("room_claude_relay_bus_unavailable channel=%s", channel)
            return
        envelope = BaseEnvelope(
            kind=kind,
            source=ServiceRef(
                name=self._service_name, version=self._service_version, node=self._node_name
            ),
            payload=payload.model_dump(mode="json") if hasattr(payload, "model_dump") else payload,
        )
        await self._bus.publish(channel, envelope)
