from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import uuid4

from app.settings import get_settings

from orion.autonomy.fcc_env import expand_env_path, load_fcc_env
from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.codec import OrionCodec
from orion.core.bus.resilience import publish_with_reconnect
from orion.embodiment import aitown_client
from orion.embodiment.arbiter import ArbiterState, decide
from orion.embodiment.perception import build_perception
from orion.embodiment.resolver import resolve_destination
from orion.embodiment.salience import SalienceState, evaluate_salience
from orion.embodiment.speech import build_speech_prompt, is_injectable, should_speak
from orion.journaler.schemas import JournalTriggerV1
from orion.schemas.embodiment import (
    EMBODIMENT_OUTCOME_KIND,
    EMBODIMENT_PERCEPTION_KIND,
    EmbodimentIntentV1,
    EmbodimentOutcomeV1,
    WorldPerceptionV1,
)

logger = logging.getLogger("orion.embodiment.worker")

# Fixed cross-service contract endpoint (owned by orion-actions). The channel is
# `orion:actions:trigger:journal.v1` and the consumer routes on the dotted kind
# below — this is not operator-tunable, so it lives as a constant, not an env key.
JOURNAL_TRIGGER_CHANNEL = "orion:actions:trigger:journal.v1"
JOURNAL_TRIGGER_KIND = "orion.actions.trigger.journal.v1"

# AI Town canonical conversation inputs. The upstream convex/aiTown/inputs.ts is
# NOT vendored in this checkout, so these follow the AI Town canonical schema.
# TODO(embodiment): confirm names/args against upstream when it is vendored.
START_CONVERSATION_INPUT = "startConversation"
START_TYPING_INPUT = "startTyping"
FINISH_SENDING_MESSAGE_INPUT = "finishSendingMessage"
WRITE_MESSAGE_MUTATION = "messages:writeMessage"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class EmbodimentWorker:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._stop = asyncio.Event()
        self._bus = OrionBusAsync(
            self._settings.bus_url, enabled=self._settings.bus_enabled, codec=OrionCodec()
        )
        self._arbiter = ArbiterState()
        self._hold_sec = self._settings.deliberate_hold_sec
        self._wander_radius = self._settings.wander_radius
        self._social_cooldown_sec = self._settings.social_cooldown_sec
        self._move_cooldown_sec = self._settings.move_cooldown_sec
        self._last_conversation_start: Optional[datetime] = None
        self._last_move_at: Optional[datetime] = None
        self._speaking_conversations: set[str] = set()
        self._salience = SalienceState()
        # Conversation-completion tracking for the journal gate (perception delta).
        self._active_conversation_id: Optional[str] = None
        self._active_conversation_partner: Optional[str] = None
        self._active_conversation_utterances: int = 0
        self._load_fcc_env()
        self._orion_player_id = str(os.environ.get("AITOWN_ORION_PLAYER_ID") or "").strip()
        self._world_id = str(os.environ.get("AITOWN_WORLD_ID") or "").strip()
        try:
            self._locations = json.loads(self._settings.locations_json or "{}")
        except json.JSONDecodeError:
            self._locations = {}

    def _load_fcc_env(self) -> None:
        for k, v in load_fcc_env(expand_env_path(self._settings.fcc_env_path)).items():
            os.environ.setdefault(k, v)
        # Explicit override wins over the ~/.fcc/.env value (bridge reachability).
        override = self._settings.aitown_convex_url.strip()
        if override:
            os.environ["AITOWN_CONVEX_URL"] = override

    def _service_ref(self) -> ServiceRef:
        return ServiceRef(
            name=self._settings.service_name,
            version=self._settings.service_version,
            node=self._settings.node_name,
        )

    # --- unit-testable core -------------------------------------------------
    def process_intent(self, intent: EmbodimentIntentV1, *, now: datetime) -> EmbodimentOutcomeV1:
        decision = decide(intent, self._arbiter, now=now, hold_sec=self._hold_sec)
        if not decision.accept:
            return EmbodimentOutcomeV1(
                intent_correlation_id=intent.correlation_id, source=intent.source,
                status="preempted", reason=decision.reason, player_id=self._orion_player_id or None,
            )

        player_id = (intent.player_id or self._orion_player_id or "").strip()
        if not player_id:
            return EmbodimentOutcomeV1(
                intent_correlation_id=intent.correlation_id, source=intent.source,
                status="denied", reason="missing AITOWN_ORION_PLAYER_ID (run bootstrap)",
            )

        try:
            players = aitown_client.list_players(world_id=self._world_id or None) or []
        except aitown_client.AitownClientError as exc:
            return EmbodimentOutcomeV1(
                intent_correlation_id=intent.correlation_id, source=intent.source,
                status="error", reason=f"list_players failed: {exc}", player_id=player_id,
            )

        result = resolve_destination(
            intent, orion_player_id=player_id, players=players,
            locations=self._locations, wander_radius=self._wander_radius,
        )
        if result.status == "resolved_noop":
            return EmbodimentOutcomeV1(
                intent_correlation_id=intent.correlation_id, source=intent.source,
                status="resolved_noop", reason=result.reason, player_id=player_id,
            )
        if result.status == "denied" or result.destination is None:
            return EmbodimentOutcomeV1(
                intent_correlation_id=intent.correlation_id, source=intent.source,
                status="denied", reason=result.reason, player_id=player_id,
            )

        if intent.kind == "start_conversation":
            # Enforce social cooldown before any actuation so repeated invites
            # (deliberate or involuntary) cannot spam conversation starts.
            last = self._last_conversation_start
            if last is not None and (now - last) < timedelta(seconds=float(self._social_cooldown_sec)):
                remaining = float(self._social_cooldown_sec) - (now - last).total_seconds()
                return EmbodimentOutcomeV1(
                    intent_correlation_id=intent.correlation_id, source=intent.source,
                    status="denied",
                    reason=f"social cooldown active {remaining:.1f}s remaining",
                    player_id=player_id, resolved_destination=result.destination,
                )
            if not result.ref_player_id:
                return EmbodimentOutcomeV1(
                    intent_correlation_id=intent.correlation_id, source=intent.source,
                    status="denied", reason="start_conversation target has no player id",
                    player_id=player_id, resolved_destination=result.destination,
                )
            try:
                aitown_client.move_to(
                    player_id=player_id, x=result.destination["x"], y=result.destination["y"],
                    world_id=self._world_id or None,
                )
                aitown_client.send_input(
                    name=START_CONVERSATION_INPUT,
                    args={"playerId": player_id, "invitee": result.ref_player_id},
                    world_id=self._world_id or None,
                )
            except aitown_client.AitownClientError as exc:
                return EmbodimentOutcomeV1(
                    intent_correlation_id=intent.correlation_id, source=intent.source,
                    status="error", reason=f"send_input failed: {exc}",
                    player_id=player_id, resolved_destination=result.destination,
                )
            self._last_conversation_start = now
            self._last_move_at = now
            return EmbodimentOutcomeV1(
                intent_correlation_id=intent.correlation_id, source=intent.source,
                status="actuated", reason=f"start_conversation ({result.reason})",
                player_id=player_id, resolved_destination=result.destination, send_input_ok=True,
            )

        # Debounce competing move actuations (multiple deliberate/involuntary
        # producers can each resolve a different destination) to avoid sprite thrash.
        if (
            self._move_cooldown_sec > 0
            and self._last_move_at is not None
            and (now - self._last_move_at) < timedelta(seconds=float(self._move_cooldown_sec))
        ):
            remaining = float(self._move_cooldown_sec) - (now - self._last_move_at).total_seconds()
            return EmbodimentOutcomeV1(
                intent_correlation_id=intent.correlation_id, source=intent.source,
                status="resolved_noop",
                reason=f"move cooldown active {remaining:.1f}s remaining",
                player_id=player_id, resolved_destination=result.destination,
            )

        try:
            aitown_client.move_to(
                player_id=player_id, x=result.destination["x"], y=result.destination["y"],
                world_id=self._world_id or None,
            )
        except aitown_client.AitownClientError as exc:
            return EmbodimentOutcomeV1(
                intent_correlation_id=intent.correlation_id, source=intent.source,
                status="error", reason=f"send_input failed: {exc}",
                player_id=player_id, resolved_destination=result.destination,
            )

        self._last_move_at = now
        return EmbodimentOutcomeV1(
            intent_correlation_id=intent.correlation_id, source=intent.source,
            status="actuated", reason=result.reason, player_id=player_id,
            resolved_destination=result.destination, send_input_ok=True,
        )

    # --- async plumbing -----------------------------------------------------
    async def start(self) -> None:
        if not self._settings.enabled:
            logger.info("embodiment_worker_disabled ORION_EMBODIMENT_ENABLED=false")
            return
        await self._bus.connect()
        asyncio.create_task(self._consume_loop(), name="embodiment-consume")
        if self._settings.perception_interval_sec > 0:
            asyncio.create_task(self._perception_loop(), name="embodiment-perception")

    async def stop(self) -> None:
        self._stop.set()
        await self._bus.close()

    async def _consume_loop(self) -> None:
        channel = self._settings.channel_intent
        async with self._bus.subscribe(channel) as pubsub:
            while not self._stop.is_set():
                try:
                    msg = await asyncio.wait_for(
                        pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0), timeout=1.2
                    )
                except asyncio.TimeoutError:
                    continue
                if not msg or msg.get("type") not in ("message", "pmessage"):
                    continue
                try:
                    await self._handle(msg)
                except Exception:
                    logger.exception("embodiment_handle_failed")

    async def _handle(self, raw_msg: dict) -> None:
        decoded = self._bus.codec.decode(raw_msg.get("data"))
        if not decoded.ok:
            logger.warning("embodiment_decode_failed: %s", decoded.error)
            return
        try:
            intent = EmbodimentIntentV1.model_validate(decoded.envelope.payload or {})
        except ValueError as exc:
            logger.error("embodiment_invalid_intent err=%s", exc)
            return
        outcome = await asyncio.to_thread(self.process_intent, intent, now=_utcnow())
        await self._publish_outcome(outcome)

    async def _publish_outcome(self, outcome: EmbodimentOutcomeV1) -> None:
        env = BaseEnvelope(
            kind=EMBODIMENT_OUTCOME_KIND, source=self._service_ref(),
            correlation_id=uuid4(), payload=outcome.model_dump(mode="json"),
        )
        try:
            await publish_with_reconnect(
                self._bus, self._settings.channel_outcome, env, log_label="embodiment_outcome"
            )
        except Exception:
            logger.exception("embodiment_outcome_publish_failed")

    # --- memory / journal ---------------------------------------------------
    async def _maybe_journal_episode(self, event: dict) -> Optional[str]:
        """Gated, deduped, fail-open journal emit for salient town episodes.

        Off unless ``memory_enabled``. The pure salience gate (bounded/deduped by
        ``self._salience``) decides whether the event is an episode and produces
        the who/what summary. A salient event publishes exactly one
        ``JournalTriggerV1`` (trigger_kind ``town_episode`` / source_kind
        ``embodiment``) to the actions journal channel. Never crashes the worker.
        """
        if not getattr(self._settings, "memory_enabled", False):
            return None
        try:
            evaluation = evaluate_salience(event, self._salience)
        except Exception:
            logger.exception("embodiment_salience_eval_failed")
            return None
        if not evaluation.salient:
            return None
        summary = (evaluation.summary or "").strip()
        if not summary:
            # No real who/what content -> refuse to emit an empty-shell episode.
            logger.warning("embodiment_journal_skipped_empty_summary event_type=%s", event.get("type"))
            return None
        try:
            trigger = JournalTriggerV1(
                trigger_kind="town_episode",
                source_kind="embodiment",
                summary=summary,
                source_ref=evaluation.source_ref,
            )
            env = BaseEnvelope(
                kind=JOURNAL_TRIGGER_KIND, source=self._service_ref(),
                correlation_id=uuid4(), payload=trigger.model_dump(mode="json"),
            )
            await publish_with_reconnect(
                self._bus, JOURNAL_TRIGGER_CHANNEL, env, log_label="embodiment_journal"
            )
        except Exception:
            logger.exception("embodiment_journal_publish_failed")
            return None
        return evaluation.source_ref or summary

    @staticmethod
    def _conversation_partner_name(perception: WorldPerceptionV1, convo: dict) -> Optional[str]:
        """Best-effort human name of Orion's conversation partner.

        Prefers an explicit partner name/id on the conversation frame; falls back
        to the nearest perceived player. Returns ``None`` if nothing is known so
        the summary carries "someone" rather than a fabricated name.
        """
        for key in ("with", "partner", "partner_name", "invitee", "other_player_name"):
            val = convo.get(key)
            if val:
                return str(val)
        nearby = perception.nearby_players or []
        if nearby:
            first = nearby[0]
            return str(first.get("name") or first.get("player_id") or "") or None
        return None

    async def _journal_from_perception(self, perception: WorldPerceptionV1) -> None:
        """Derive salient episodes from a perception tick and journal them.

        Two signals, both bounded/deduped downstream by the pure salience gate:
          - conversation completion: the conversation Orion was in is no longer
            active -> emit a ``conversation_completed`` event carrying the partner
            and the count of utterances Orion contributed (0 utterances is not
            salient, so silent fly-bys are dropped by the gate).
          - encounters: each perceived player is offered as an ``encounter``; the
            gate journals only the first sighting of each player and dedupes the
            rest via ``SalienceState.seen_players``.

        Fail-open: never raises into the perception loop.
        """
        if not getattr(self._settings, "memory_enabled", False):
            return
        try:
            convo = perception.active_conversation or {}
            convo_id = str(convo.get("conversation_id") or convo.get("id") or "").strip() or None
            prev_id = self._active_conversation_id
            if prev_id and prev_id != convo_id:
                # Only journal if we know who Orion talked to — a who-less episode
                # would be empty-shell content, so drop it rather than emit "someone".
                if self._active_conversation_partner:
                    await self._maybe_journal_episode({
                        "type": "conversation_completed",
                        "with": self._active_conversation_partner,
                        "utterances": self._active_conversation_utterances,
                        "conversation_id": prev_id,
                    })
                self._active_conversation_partner = None
                self._active_conversation_utterances = 0
            if convo_id and convo_id != prev_id:
                self._active_conversation_partner = self._conversation_partner_name(perception, convo)
                self._active_conversation_utterances = 0
            self._active_conversation_id = convo_id

            for np_ in perception.nearby_players or []:
                player_id = str(np_.get("player_id") or "").strip()
                if not player_id:
                    continue
                await self._maybe_journal_episode({
                    "type": "encounter",
                    "player_id": player_id,
                    "name": np_.get("name"),
                })
        except Exception:
            logger.exception("embodiment_journal_from_perception_failed")

    # --- perception ---------------------------------------------------------
    async def _emit_perception_once(self) -> Optional[WorldPerceptionV1]:
        player_id = (self._orion_player_id or "").strip()
        if not player_id:
            return None
        try:
            players = await asyncio.to_thread(
                aitown_client.list_players, world_id=self._world_id or None
            )
        except Exception:
            logger.exception("embodiment_perception_list_players_failed")
            return None
        try:
            perception = build_perception(players=players or [], orion_player_id=player_id)
        except Exception:
            logger.exception("embodiment_perception_build_failed")
            return None
        if perception is None:
            return None
        env = BaseEnvelope(
            kind=EMBODIMENT_PERCEPTION_KIND, source=self._service_ref(),
            correlation_id=uuid4(), payload=perception.model_dump(mode="json"),
        )
        try:
            await publish_with_reconnect(
                self._bus, self._settings.channel_perception, env,
                log_label="embodiment_perception",
            )
        except Exception:
            logger.exception("embodiment_perception_publish_failed")
            return None
        return perception

    async def _perception_loop(self) -> None:
        interval = self._settings.perception_interval_sec
        while not self._stop.is_set():
            try:
                perception = await self._emit_perception_once()
            except Exception:
                logger.exception("embodiment_perception_loop_failed")
                perception = None
            if perception is not None and getattr(self._settings, "memory_enabled", False):
                try:
                    await self._journal_from_perception(perception)
                except Exception:
                    logger.exception("embodiment_memory_loop_failed")
            if perception is not None and getattr(self._settings, "speech_enabled", False):
                try:
                    await self._speak_once(perception)
                except Exception:
                    logger.exception("embodiment_speech_loop_failed")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                continue

    # --- speech (cortex-generated town utterances) --------------------------
    async def _speak_once(self, perception: WorldPerceptionV1) -> Optional[str]:
        """Gated, fail-open speech pass. Own-agent only; one utterance per convo."""
        if not getattr(self._settings, "speech_enabled", False):
            return None
        own = (self._orion_player_id or "").strip()
        if not should_speak(perception, own):
            return None
        convo = perception.active_conversation or {}
        convo_id = str(convo.get("conversation_id") or convo.get("id") or "")
        if not convo_id or convo_id in self._speaking_conversations:
            return None

        prompt = build_speech_prompt(perception, own)
        # Hold the conversation as in-flight through both the utterance request AND
        # the injection so a later perception tick cannot double-inject.
        self._speaking_conversations.add(convo_id)
        try:
            try:
                reply = await self._request_utterance(prompt, correlation_id=str(uuid4()))
            except Exception:
                logger.exception("embodiment_speech_request_failed convo=%s", convo_id)
                reply = ""

            if not is_injectable(reply):
                logger.info("embodiment_speech_empty_reply_skipped convo=%s", convo_id)
                return None

            try:
                await asyncio.to_thread(self._inject_utterance, own, convo_id, reply)
            except Exception:
                logger.exception("embodiment_speech_inject_failed convo=%s", convo_id)
                return None
            # Record Orion's contribution so the journal gate sees a real exchange
            # when this conversation later completes.
            if convo_id == getattr(self, "_active_conversation_id", None):
                self._active_conversation_utterances = (
                    getattr(self, "_active_conversation_utterances", 0) + 1
                )
            return reply
        finally:
            self._speaking_conversations.discard(convo_id)

    async def _request_utterance(self, prompt: str, *, correlation_id: str) -> str:
        """Reuse the cortex exec rail to generate an utterance. Fail-open -> ''."""
        from orion.cognition.cortex_payload_extract import extract_cortex_payload_text
        from orion.cognition.plan_loader import build_plan_for_verb
        from orion.schemas.cortex.schemas import PlanExecutionArgs, PlanExecutionRequest

        try:
            plan = build_plan_for_verb(self._settings.speech_verb, mode=self._settings.speech_lane)
            req = PlanExecutionRequest(
                plan=plan,
                args=PlanExecutionArgs(
                    request_id=correlation_id,
                    trigger_source=self._settings.service_name,
                    extra={"lane": self._settings.speech_lane},
                ),
                context={"user_message": prompt, "metadata": {"correlation_id": correlation_id}},
            )
            reply_channel = f"{self._settings.cortex_result_prefix}:{uuid4()}"
            env = BaseEnvelope(
                kind=req.kind,
                source=self._service_ref(),
                correlation_id=correlation_id,
                reply_to=reply_channel,
                payload=req.model_dump(mode="json"),
            )
            msg = await self._bus.rpc_request(
                self._settings.cortex_request_channel,
                env,
                reply_channel=reply_channel,
                timeout_sec=float(self._settings.speech_timeout_sec),
            )
            decoded = self._bus.codec.decode(msg.get("data"))
            if not decoded.ok:
                return ""
            payload = decoded.envelope.payload
            result = payload.get("result") if isinstance(payload, dict) else None
            text = extract_cortex_payload_text(result if isinstance(result, dict) else (payload or {}))
            return str(text or "")
        except Exception:
            logger.exception("embodiment_speech_rpc_failed corr=%s", correlation_id)
            return ""

    def _inject_utterance(self, own_player_id: str, conversation_id: str, text: str) -> None:
        """Inject an utterance into the AI Town conversation (startTyping -> write -> finish).

        Input/mutation names follow the AI Town canonical schema (upstream not
        vendored here). See TODO on the input constants above.
        """
        message_uuid = str(uuid4())
        wid = self._world_id or None
        aitown_client.send_input(
            name=START_TYPING_INPUT,
            args={"playerId": own_player_id, "conversationId": conversation_id, "messageUuid": message_uuid},
            world_id=wid,
        )
        aitown_client.convex_mutation(
            WRITE_MESSAGE_MUTATION,
            {
                "worldId": self._world_id,
                "conversationId": conversation_id,
                "messageUuid": message_uuid,
                "author": own_player_id,
                "text": text,
            },
        )
        aitown_client.send_input(
            name=FINISH_SENDING_MESSAGE_INPUT,
            args={"playerId": own_player_id, "conversationId": conversation_id, "messageUuid": message_uuid},
            world_id=wid,
        )
