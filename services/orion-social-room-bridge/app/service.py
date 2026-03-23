from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.social_autonomy import SocialTurnPolicyDecisionV1
from orion.schemas.social_bridge import (
    CallSyneRoomMessageV1,
    ExternalRoomMessageV1,
    ExternalRoomParticipantV1,
    ExternalRoomPostRequestV1,
    ExternalRoomPostResultV1,
    ExternalRoomTurnSkippedV1,
)

from .clients import CallSyneClient, HubClient, SocialMemoryClient
from .policy import PolicyContext, SocialTurnPolicyEvaluator
from .settings import Settings

logger = logging.getLogger("orion-social-room-bridge")


@dataclass
class RoomState:
    last_outbound_at: float | None = None
    consecutive_orion_turns: int = 0


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SocialRoomBridgeService:
    def __init__(
        self,
        *,
        settings: Settings,
        hub_client: HubClient | None = None,
        callsyne_client: CallSyneClient | None = None,
        social_memory_client: SocialMemoryClient | None = None,
        bus: OrionBusAsync | None = None,
        clock: callable | None = None,
    ) -> None:
        self.settings = settings
        self.hub_client = hub_client or HubClient(
            base_url=settings.hub_base_url,
            chat_path=settings.hub_chat_path,
            timeout_sec=settings.hub_timeout_sec,
        )
        self.callsyne_client = callsyne_client or CallSyneClient(
            base_url=settings.callsyne_base_url,
            api_token=settings.callsyne_api_token,
            timeout_sec=settings.callsyne_timeout_sec,
            post_path_template=settings.callsyne_post_path_template,
        )
        self.social_memory_client = social_memory_client or SocialMemoryClient(
            base_url=settings.social_memory_base_url,
            timeout_sec=settings.social_memory_timeout_sec,
        )
        self.bus = bus
        self._clock = clock or time.time
        self._seen: dict[str, float] = {}
        self._room_state: dict[str, RoomState] = {}
        self.policy = SocialTurnPolicyEvaluator(settings=settings)

    async def start(self) -> None:
        if self.bus is None and self.settings.orion_bus_enabled:
            self.bus = OrionBusAsync(
                self.settings.orion_bus_url,
                enabled=self.settings.orion_bus_enabled,
                enforce_catalog=self.settings.orion_bus_enforce_catalog,
            )
            await self.bus.connect()

    async def stop(self) -> None:
        if self.bus is not None:
            await self.bus.close()
            self.bus = None

    def normalize_callsyne_message(self, payload: Dict[str, Any]) -> CallSyneRoomMessageV1:
        room = payload.get("room") if isinstance(payload.get("room"), dict) else {}
        sender = payload.get("sender") if isinstance(payload.get("sender"), dict) else {}
        metadata = dict(payload.get("metadata") or {})
        room_id = str(payload.get("room_id") or room.get("id") or "").strip()
        thread_id = payload.get("thread_id") or payload.get("thread") or metadata.get("thread_id")
        message_id = str(payload.get("message_id") or payload.get("id") or "").strip()
        sender_id = str(payload.get("sender_id") or sender.get("id") or "").strip()
        sender_name = payload.get("sender_name") or sender.get("name")
        sender_kind = payload.get("sender_kind") or sender.get("kind") or metadata.get("sender_kind") or "peer_ai"
        text = str(payload.get("text") or payload.get("content") or "").strip()
        reply_to_message_id = payload.get("reply_to_message_id") or metadata.get("reply_to_message_id")
        reply_to_sender_id = payload.get("reply_to_sender_id") or metadata.get("reply_to_sender_id")
        target_participant_id = payload.get("target_participant_id") or metadata.get("target_participant_id") or reply_to_sender_id
        target_participant_name = payload.get("target_participant_name") or metadata.get("target_participant_name")
        mentioned_participant_ids = payload.get("mentioned_participant_ids") or metadata.get("mentioned_participant_ids") or []
        mentioned_participant_names = payload.get("mentioned_participant_names") or metadata.get("mentioned_participant_names") or []
        mentions_orion = bool(
            payload.get("mentions_orion")
            or metadata.get("mentions_orion")
            or self.settings.social_bridge_self_name.lower() in text.lower()
        )
        created_at = (
            payload.get("created_at")
            or payload.get("timestamp")
            or payload.get("sent_at")
            or _utcnow_iso()
        )
        normalized = CallSyneRoomMessageV1(
            room_id=room_id,
            thread_id=str(thread_id).strip() or None if thread_id is not None else None,
            message_id=message_id,
            sender_id=sender_id,
            sender_name=str(sender_name).strip() or None if sender_name is not None else None,
            sender_kind=str(sender_kind).strip() or "peer_ai",
            text=text,
            created_at=str(created_at),
            reply_to_message_id=str(reply_to_message_id).strip() or None if reply_to_message_id is not None else None,
            mentions_orion=mentions_orion,
            reply_to_sender_id=str(reply_to_sender_id).strip() or None if reply_to_sender_id is not None else None,
            target_participant_id=str(target_participant_id).strip() or None if target_participant_id is not None else None,
            target_participant_name=str(target_participant_name).strip() or None if target_participant_name is not None else None,
            mentioned_participant_ids=[str(item).strip() for item in mentioned_participant_ids if str(item).strip()],
            mentioned_participant_names=[str(item).strip() for item in mentioned_participant_names if str(item).strip()],
            raw_payload=dict(payload),
            metadata=metadata,
        )
        return normalized

    async def process_callsyne_message(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        message = self.normalize_callsyne_message(payload)
        logger.info(
            "room_inbound_received platform=%s room_id=%s message_id=%s sender_id=%s sender_kind=%s",
            message.platform,
            message.room_id,
            message.message_id,
            message.sender_id,
            message.sender_kind,
        )
        await self._publish(self.settings.room_participant_channel, "external.room.participant.v1", self._participant(message))
        await self._publish(self.settings.room_intake_channel, "external.room.message.v1", self._intake_event(message))

        social_memory = await self._load_social_memory(message)
        decision = self._policy_decision(message, social_memory=social_memory)
        logger.info(
            "room_turn_policy_decided room_id=%s message_id=%s should_speak=%s decision=%s audience=%s route=%s reasons=%s",
            message.room_id,
            message.message_id,
            decision.should_speak,
            decision.decision,
            decision.thread_routing.audience_scope if decision.thread_routing else "none",
            decision.thread_routing.routing_decision if decision.thread_routing else "none",
            " | ".join(decision.reasons),
        )
        await self._publish(
            self.settings.room_turn_policy_channel,
            "social.turn.policy.v1",
            decision,
        )
        if decision.repair_signal is not None:
            logger.info(
                "social_repair_signal_detected room_id=%s message_id=%s type=%s confidence=%.2f rationale=%s",
                message.room_id,
                message.message_id,
                decision.repair_signal.repair_type,
                decision.repair_signal.confidence,
                decision.repair_signal.rationale,
            )
            await self._publish(
                self.settings.room_repair_signal_channel,
                "social.repair.signal.v1",
                decision.repair_signal,
            )
        if decision.repair_decision is not None:
            logger.info(
                "social_repair_decision_chosen room_id=%s message_id=%s decision=%s confidence=%.2f rationale=%s",
                message.room_id,
                message.message_id,
                decision.repair_decision.decision,
                decision.repair_decision.confidence,
                decision.repair_decision.rationale,
            )
            await self._publish(
                self.settings.room_repair_decision_channel,
                "social.repair.decision.v1",
                decision.repair_decision,
            )
        if decision.epistemic_signal is not None:
            logger.info(
                "social_epistemic_signal_detected room_id=%s message_id=%s claim_kind=%s confidence=%s ambiguity=%s rationale=%s",
                message.room_id,
                message.message_id,
                decision.epistemic_signal.claim_kind,
                decision.epistemic_signal.confidence_level,
                decision.epistemic_signal.ambiguity_level,
                decision.epistemic_signal.rationale,
            )
            await self._publish(
                self.settings.room_epistemic_signal_channel,
                "social.epistemic.signal.v1",
                decision.epistemic_signal,
            )
        if decision.epistemic_decision is not None:
            logger.info(
                "social_epistemic_decision_chosen room_id=%s message_id=%s decision=%s claim_kind=%s rationale=%s",
                message.room_id,
                message.message_id,
                decision.epistemic_decision.decision,
                decision.epistemic_decision.claim_kind,
                decision.epistemic_decision.rationale,
            )
            await self._publish(
                self.settings.room_epistemic_decision_channel,
                "social.epistemic.decision.v1",
                decision.epistemic_decision,
            )

        if not decision.should_speak:
            skip_reason = self._policy_skip_reason(decision)
            if skip_reason not in {"duplicate_message", "self_message"}:
                self._room_state.setdefault(self._room_key(message), RoomState()).consecutive_orion_turns = 0
            logger.info(
                "room_turn_skipped platform=%s room_id=%s message_id=%s reason=%s",
                message.platform,
                message.room_id,
                message.message_id,
                skip_reason,
            )
            await self._publish(
                self.settings.room_skipped_channel,
                "external.room.turn.skipped.v1",
                self._skip_event(message, skip_reason, decision=decision),
            )
            return {"status": "skipped", "reason": skip_reason, "message_id": message.message_id}

        dedupe_key = self._dedupe_key(message)
        self._mark_seen(dedupe_key)

        session_id = self._session_id(message)
        hub_payload = self._hub_payload(message, decision=decision)
        if social_memory:
            open_commitments = [
                item
                for item in ((social_memory.get("room") or {}).get("active_commitments") or [])
                if isinstance(item, dict) and str(item.get("state") or "open") == "open"
            ][:2]
            if decision.repair_decision is not None and decision.repair_decision.decision in {"yield", "reset_thread"}:
                logger.info(
                    "social_repair_commitments_suppressed room_id=%s message_id=%s decision=%s",
                    message.room_id,
                    message.message_id,
                    decision.repair_decision.decision,
                )
                open_commitments = []
            hub_payload.update(
                {
                    "social_peer_continuity": social_memory.get("participant"),
                    "social_room_continuity": social_memory.get("room"),
                    "social_stance_snapshot": social_memory.get("stance"),
                    "social_peer_style_hint": social_memory.get("peer_style"),
                    "social_room_ritual_summary": social_memory.get("room_ritual"),
                    "social_open_commitments": open_commitments or None,
                    "social_thread_routing": decision.thread_routing.model_dump(mode="json") if decision.thread_routing else None,
                    "social_handoff_signal": decision.handoff_signal.model_dump(mode="json") if decision.handoff_signal else None,
                    "social_repair_signal": decision.repair_signal.model_dump(mode="json") if decision.repair_signal else None,
                    "social_repair_decision": decision.repair_decision.model_dump(mode="json") if decision.repair_decision else None,
                    "social_epistemic_signal": decision.epistemic_signal.model_dump(mode="json") if decision.epistemic_signal else None,
                    "social_epistemic_decision": decision.epistemic_decision.model_dump(mode="json") if decision.epistemic_decision else None,
                }
            )
        logger.info(
            "room_orion_invocation_sent room_id=%s message_id=%s session_id=%s chat_profile=social_room",
            message.room_id,
            message.message_id,
            session_id,
        )
        hub_result = await self.hub_client.chat(payload=hub_payload, session_id=session_id)
        reply_text = str(hub_result.get("text") or "").strip()
        correlation_id = str(hub_result.get("correlation_id") or uuid4())
        logger.info(
            "room_orion_reply_received room_id=%s message_id=%s correlation_id=%s reply_len=%s",
            message.room_id,
            message.message_id,
            correlation_id,
            len(reply_text),
        )

        if not reply_text:
            skip_reason = "empty_reply"
            await self._publish(
                self.settings.room_skipped_channel,
                "external.room.turn.skipped.v1",
                self._skip_event(message, skip_reason, correlation_id=correlation_id, decision=decision),
            )
            return {"status": "skipped", "reason": skip_reason, "message_id": message.message_id, "correlation_id": correlation_id}

        if self.settings.social_bridge_dry_run:
            skip_reason = "dry_run"
            await self._publish(
                self.settings.room_skipped_channel,
                "external.room.turn.skipped.v1",
                self._skip_event(message, skip_reason, correlation_id=correlation_id, decision=decision),
            )
            return {
                "status": "skipped",
                "reason": skip_reason,
                "message_id": message.message_id,
                "correlation_id": correlation_id,
                "reply_text": reply_text,
            }

        post_request = ExternalRoomPostRequestV1(
            platform=message.platform,
            room_id=message.room_id,
            thread_id=message.thread_id,
            correlation_id=correlation_id,
            reply_to_message_id=message.message_id,
            text=reply_text,
            metadata={
                "source": self.settings.service_name,
                "chat_profile": "social_room",
                "inbound_message_id": message.message_id,
            },
        )
        try:
            delivery_raw = await self.callsyne_client.post_message(post_request)
            posted_message_id = str(delivery_raw.get("message_id") or delivery_raw.get("id") or "").strip()
            delivery_event = self._delivery_event(
                message,
                correlation_id=correlation_id,
                reply_text=reply_text,
                posted_message_id=posted_message_id,
                delivery_raw=delivery_raw,
            )
            logger.info(
                "room_outbound_post_succeeded room_id=%s inbound_message_id=%s outbound_message_id=%s correlation_id=%s",
                message.room_id,
                message.message_id,
                posted_message_id,
                correlation_id,
            )
            await self._publish(
                self.settings.room_delivery_channel,
                "external.room.post.result.v1",
                delivery_event,
            )
            self._record_outbound(message)
            return {
                "status": "ok",
                "message_id": message.message_id,
                "correlation_id": correlation_id,
                "posted_message_id": posted_message_id,
                "reply_text": reply_text,
            }
        except Exception as exc:
            logger.warning(
                "room_outbound_post_failed room_id=%s inbound_message_id=%s correlation_id=%s error=%s",
                message.room_id,
                message.message_id,
                correlation_id,
                exc,
            )
            await self._publish(
                self.settings.room_delivery_channel,
                "external.room.post.result.v1",
                self._delivery_event(
                    message,
                    correlation_id=correlation_id,
                    reply_text=reply_text,
                    posted_message_id="",
                    delivery_raw={"error": str(exc)},
                    delivery_ok=False,
                    delivery_error=str(exc),
                ),
            )
            raise

    def _policy_decision(
        self,
        message: CallSyneRoomMessageV1,
        *,
        social_memory: Dict[str, Any],
    ) -> SocialTurnPolicyDecisionV1:
        mode = self._policy_mode()
        if not self.settings.social_bridge_enabled:
            return self.policy._decision(
                message=message,
                decision="skip",
                should_speak=False,
                reasons=["bridge disabled"],
                mode=mode,
                addressed=False,
                cooldown_active=False,
                consecutive_limit_hit=False,
                quiet_room=True,
                novelty_score=0.0,
                continuity_score=0.0,
                open_thread=None,
            )
        if self.settings.social_bridge_room_allowlist and message.room_id not in self.settings.social_bridge_room_allowlist:
            return self.policy._decision(
                message=message,
                decision="skip",
                should_speak=False,
                reasons=["room not allowlisted"],
                mode=mode,
                addressed=False,
                cooldown_active=False,
                consecutive_limit_hit=False,
                quiet_room=True,
                novelty_score=0.0,
                continuity_score=0.0,
                open_thread=None,
            )
        if not message.message_id:
            return self.policy._decision(
                message=message,
                decision="skip",
                should_speak=False,
                reasons=["missing transport message id"],
                mode=mode,
                addressed=False,
                cooldown_active=False,
                consecutive_limit_hit=False,
                quiet_room=True,
                novelty_score=0.0,
                continuity_score=0.0,
                open_thread=None,
            )

        state = self._room_state.setdefault(self._room_key(message), RoomState())
        addressed = self._is_addressed_to_orion(message)
        now = float(self._clock())
        cooldown_active = False
        if state.last_outbound_at is not None and self.settings.social_bridge_cooldown_sec > 0:
            cooldown_active = (now - state.last_outbound_at) < float(self.settings.social_bridge_cooldown_sec)
        consecutive_limit_hit = (
            self.settings.social_bridge_max_consecutive_orion_turns > 0
            and state.consecutive_orion_turns >= self.settings.social_bridge_max_consecutive_orion_turns
        )
        decision = self.policy.evaluate(
            message=message,
            context=PolicyContext(
                social_memory=social_memory,
                is_duplicate=self._is_duplicate(message),
                is_self_message=self._is_self_message(message),
                addressed=addressed,
                cooldown_active=cooldown_active,
                consecutive_limit_hit=consecutive_limit_hit,
            ),
            mode=mode,
        )

        return decision

    def _policy_mode(self) -> str:
        if self.settings.social_bridge_only_when_addressed:
            return "addressed_only"
        return self.settings.social_bridge_autonomy_mode

    def _policy_skip_reason(self, decision: SocialTurnPolicyDecisionV1) -> str:
        if decision.cooldown_active:
            return "cooldown_active"
        if decision.consecutive_limit_hit:
            return "max_consecutive_orion_turns"
        if any("self-loop" in reason for reason in decision.reasons):
            return "self_message"
        if any("duplicate" in reason for reason in decision.reasons):
            return "duplicate_message"
        if any("allowlisted" in reason for reason in decision.reasons):
            return "room_not_allowlisted"
        if any("addressed_only" in reason for reason in decision.reasons):
            return "not_addressed"
        if any("low novelty" in reason for reason in decision.reasons):
            return "low_novelty"
        if any("peer_targeted_elsewhere" in reason for reason in decision.reasons):
            return "peer_targeted_elsewhere"
        if any("aimed at another participant" in reason for reason in decision.reasons):
            return "peer_targeted_elsewhere"
        if any("bridge disabled" in reason for reason in decision.reasons):
            return "bridge_disabled"
        if any("missing transport message id" in reason for reason in decision.reasons):
            return "missing_message_id"
        return "policy_wait"

    def _is_self_message(self, message: CallSyneRoomMessageV1) -> bool:
        if message.sender_id in self.settings.social_bridge_self_participant_ids:
            return True
        if message.metadata.get("sender_is_self") is True:
            return True
        sender_name = str(message.sender_name or "").strip().lower()
        self_name = self.settings.social_bridge_self_name.strip().lower()
        return bool(sender_name and self_name and sender_name == self_name)

    def _is_addressed_to_orion(self, message: CallSyneRoomMessageV1) -> bool:
        if message.mentions_orion:
            return True
        if message.reply_to_sender_id and message.reply_to_sender_id in self.settings.social_bridge_self_participant_ids:
            return True
        return bool(message.metadata.get("reply_target_is_orion"))

    def _dedupe_key(self, message: CallSyneRoomMessageV1) -> str:
        return f"{message.platform}:{message.room_id}:{message.message_id}"

    def _prune_seen(self) -> None:
        now = float(self._clock())
        ttl = max(int(self.settings.social_bridge_dedupe_ttl_sec), 1)
        expired = [key for key, expiry in self._seen.items() if expiry <= now]
        for key in expired:
            self._seen.pop(key, None)
        self._seen = {key: expiry for key, expiry in self._seen.items() if expiry > (now - ttl * 2)}

    def _is_duplicate(self, message: CallSyneRoomMessageV1) -> bool:
        self._prune_seen()
        return self._dedupe_key(message) in self._seen

    def _mark_seen(self, key: str) -> None:
        self._prune_seen()
        self._seen[key] = float(self._clock()) + max(int(self.settings.social_bridge_dedupe_ttl_sec), 1)

    def _room_key(self, message: CallSyneRoomMessageV1) -> str:
        return f"{message.platform}:{message.room_id}:{message.thread_id or 'room'}"

    def _session_id(self, message: CallSyneRoomMessageV1) -> str:
        return f"{self.settings.social_bridge_session_namespace}:{message.platform}:{message.room_id}:{message.thread_id or 'room'}"

    def _hub_payload(
        self,
        message: CallSyneRoomMessageV1,
        *,
        decision: SocialTurnPolicyDecisionV1,
    ) -> Dict[str, Any]:
        continuity_anchor = f"{message.platform} room {message.room_id} thread {message.thread_id or 'room'}"
        return {
            "messages": [{"role": "user", "content": message.text}],
            "mode": "brain",
            "chat_profile": "social_room",
            "user_id": message.sender_id,
            "use_recall": self.settings.social_bridge_use_recall,
            "options": {
                "tool_execution_policy": "none",
                "action_execution_policy": "none",
            },
            "external_room": {
                "platform": message.platform,
                "room_id": message.room_id,
                "thread_id": message.thread_id,
                "transport_message_id": message.message_id,
                "reply_to_message_id": message.reply_to_message_id,
                "reply_to_sender_id": message.reply_to_sender_id,
                "target_participant_id": message.target_participant_id,
                "target_participant_name": message.target_participant_name,
                "mentioned_participant_ids": list(message.mentioned_participant_ids or []),
                "mentioned_participant_names": list(message.mentioned_participant_names or []),
            },
            "external_participant": {
                "participant_id": message.sender_id,
                "participant_name": message.sender_name,
                "participant_kind": message.sender_kind,
            },
            "social_turn_policy": decision.model_dump(mode="json"),
            "social_thread_routing": decision.thread_routing.model_dump(mode="json") if decision.thread_routing else None,
            "social_handoff_signal": decision.handoff_signal.model_dump(mode="json") if decision.handoff_signal else None,
            "social_repair_signal": decision.repair_signal.model_dump(mode="json") if decision.repair_signal else None,
            "social_repair_decision": decision.repair_decision.model_dump(mode="json") if decision.repair_decision else None,
            "social_epistemic_signal": decision.epistemic_signal.model_dump(mode="json") if decision.epistemic_signal else None,
            "social_epistemic_decision": decision.epistemic_decision.model_dump(mode="json") if decision.epistemic_decision else None,
            "continuity_anchor": continuity_anchor,
        }

    def _participant(self, message: CallSyneRoomMessageV1) -> ExternalRoomParticipantV1:
        return ExternalRoomParticipantV1(
            participant_ref=f"{message.platform}:{message.room_id}:{message.sender_id}",
            platform=message.platform,
            room_id=message.room_id,
            participant_id=message.sender_id,
            participant_name=message.sender_name,
            participant_kind=message.sender_kind,
            last_message_id=message.message_id,
            last_seen_at=message.created_at,
            metadata={
                "thread_id": message.thread_id,
                "source": self.settings.service_name,
            },
        )

    def _intake_event(self, message: CallSyneRoomMessageV1, *, correlation_id: str | None = None) -> ExternalRoomMessageV1:
        return ExternalRoomMessageV1(
            correlation_id=correlation_id,
            platform=message.platform,
            room_id=message.room_id,
            thread_id=message.thread_id,
            transport_message_id=message.message_id,
            reply_to_message_id=message.reply_to_message_id,
            sender_id=message.sender_id,
            sender_name=message.sender_name,
            sender_kind=message.sender_kind,
            text=message.text,
            source=self.settings.service_name,
            observed_at=_utcnow_iso(),
            transport_ts=message.created_at,
            raw_payload=message.raw_payload,
            metadata=message.metadata,
        )

    def _skip_event(
        self,
        message: CallSyneRoomMessageV1,
        reason: str,
        *,
        correlation_id: str | None = None,
        decision: SocialTurnPolicyDecisionV1 | None = None,
    ) -> ExternalRoomTurnSkippedV1:
        base = self._intake_event(message, correlation_id=correlation_id)
        payload = base.model_dump(mode="json")
        payload.pop("skip_reason", None)
        metadata = dict(payload.get("metadata") or {})
        if decision is not None:
            metadata["policy_decision"] = decision.model_dump(mode="json")
            metadata["policy_reasons"] = list(decision.reasons)
        payload["metadata"] = metadata
        return ExternalRoomTurnSkippedV1(**payload, skip_reason=reason)

    def _delivery_event(
        self,
        message: CallSyneRoomMessageV1,
        *,
        correlation_id: str,
        reply_text: str,
        posted_message_id: str,
        delivery_raw: Dict[str, Any],
        delivery_ok: bool = True,
        delivery_error: str | None = None,
    ) -> ExternalRoomPostResultV1:
        return ExternalRoomPostResultV1(
            correlation_id=correlation_id,
            platform=message.platform,
            room_id=message.room_id,
            thread_id=message.thread_id,
            transport_message_id=posted_message_id or f"posted-{uuid4()}",
            reply_to_message_id=message.message_id,
            sender_id=self.settings.social_bridge_self_participant_ids[0] if self.settings.social_bridge_self_participant_ids else self.settings.service_name,
            sender_name=self.settings.social_bridge_self_name,
            sender_kind="peer_ai",
            text=reply_text,
            source=self.settings.service_name,
            observed_at=_utcnow_iso(),
            transport_ts=_utcnow_iso(),
            raw_payload=delivery_raw,
            metadata={
                "inbound_message_id": message.message_id,
                "thread_id": message.thread_id,
                "chat_profile": "social_room",
            },
            delivery_ok=delivery_ok,
            delivery_error=delivery_error,
        )

    def _record_outbound(self, message: CallSyneRoomMessageV1) -> None:
        state = self._room_state.setdefault(self._room_key(message), RoomState())
        state.last_outbound_at = float(self._clock())
        state.consecutive_orion_turns += 1

    async def _publish(self, channel: str, kind: str, payload: Any) -> None:
        if self.bus is None or not getattr(self.bus, "enabled", False):
            return
        envelope = BaseEnvelope(
            kind=kind,
            source=ServiceRef(
                name=self.settings.service_name,
                version=self.settings.service_version,
                node=self.settings.node_name,
            ),
            payload=payload.model_dump(mode="json") if hasattr(payload, "model_dump") else payload,
        )
        await self.bus.publish(channel, envelope)

    async def _load_social_memory(self, message: CallSyneRoomMessageV1) -> Dict[str, Any]:
        try:
            return await self.social_memory_client.get_summary(
                platform=message.platform,
                room_id=message.room_id,
                participant_id=message.sender_id,
            )
        except Exception as exc:
            logger.debug(
                "social_memory_lookup_failed room_id=%s participant_id=%s error=%s",
                message.room_id,
                message.sender_id,
                exc,
            )
            return {}
