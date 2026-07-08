from __future__ import annotations

import asyncio
import json
import logging
import math
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
from orion.embodiment.intents import build_intent
from orion.embodiment.worldmap import walkable_tiles
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
# NOTE: the `finishSendingMessage` input is NOT sent directly — `messages:writeMessage`
# enqueues it server-side with a numeric timestamp. See `_inject_utterance`.
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
        self._last_idle_wander_at: Optional[datetime] = None
        self._last_social_attempt_at: Optional[datetime] = None
        # Throttle for the loop heartbeat log so a healthy (silent) loop is still
        # observable in `docker logs` without spamming on a tight perception interval.
        self._last_heartbeat_log_at: Optional[datetime] = None
        # Serializes the two concurrent process_intent callers (consume loop +
        # idle-wander loop) so the move-cooldown read-then-write cannot double-fire.
        self._actuate_lock = asyncio.Lock()
        self._speaking_conversations: set[str] = set()
        # Conversations Orion has already opened (spoken first in). Guards the
        # "no messages yet -> open" speech path so a persistently failing message
        # fetch (which looks like an empty transcript) can't re-trigger an opener
        # every tick and spam the conversation.
        self._opened_conversations: set[str] = set()
        # Conversations for which Orion has already issued the one-shot stop that
        # clears its lingering path so the engine (Conversation.tick) orients it
        # toward the partner. Bounded per-convo so we don't fight the engine.
        self._faced_conversations: set[str] = set()
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
        # Lazily-loaded, cached walkable tile set (static map). None = not-yet-tried
        # or unavailable -> resolver falls back to unconstrained wander.
        self._walkable: Optional[set[tuple[int, int]]] = None
        self._walkable_loaded = False

    def _walkable_tiles(self) -> Optional[set[tuple[int, int]]]:
        """Fail-open, cached walkable set from the AI Town map. Fetched once."""
        if self._walkable_loaded:
            return self._walkable
        self._walkable_loaded = True
        try:
            tiles = walkable_tiles(aitown_client.get_world_map(self._world_id or None))
            self._walkable = tiles or None
        except Exception:
            logger.debug("embodiment_worldmap_load_failed", exc_info=True)
            self._walkable = None
        return self._walkable

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
            walkable=self._walkable_tiles(),
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
        await self._actuate(intent, now=_utcnow())

    async def _actuate(self, intent: EmbodimentIntentV1, *, now: datetime) -> EmbodimentOutcomeV1:
        """Serialized actuate+publish. The lock keeps the cooldown check and the
        ``_last_move_at`` write atomic across the consume and idle-wander loops."""
        async with self._actuate_lock:
            outcome = await asyncio.to_thread(self.process_intent, intent, now=now)
        await self._publish_outcome(outcome)
        return outcome

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
        if getattr(self._settings, "world_heartbeat_enabled", False):
            # Wake/keep the engine running so actuation lands (inactive worlds drop inputs).
            try:
                await asyncio.to_thread(aitown_client.heartbeat_world, self._world_id or None)
            except Exception:
                logger.debug("embodiment_world_heartbeat_failed", exc_info=True)
        try:
            players = await asyncio.to_thread(
                aitown_client.list_players, world_id=self._world_id or None
            )
        except Exception:
            logger.exception("embodiment_perception_list_players_failed")
            return None
        conversations: list = []
        try:
            conversations = await asyncio.to_thread(
                aitown_client.list_conversations, world_id=self._world_id or None
            ) or []
        except Exception:
            logger.debug("embodiment_perception_list_conversations_failed", exc_info=True)
        # Only fetch the transcript for the conversation Orion is actually
        # participating in (avoids a query per unrelated town conversation).
        messages: list = []
        cid = self._orion_conversation_id(conversations, player_id, want_status="participating")
        if cid:
            try:
                messages = await asyncio.to_thread(
                    aitown_client.list_messages, cid, world_id=self._world_id or None
                ) or []
            except Exception:
                logger.debug("embodiment_perception_list_messages_failed", exc_info=True)
        try:
            perception = build_perception(
                players=players or [], orion_player_id=player_id,
                conversations=conversations, messages=messages,
            )
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

    @staticmethod
    def _orion_conversation_id(
        conversations: list, player_id: str, *, want_status: Optional[str] = None
    ) -> Optional[str]:
        for cv in conversations or []:
            for p in cv.get("participants") or []:
                if str(p.get("playerId")) == player_id:
                    status = (p.get("status") or {}).get("kind")
                    if want_status is None or status == want_status:
                        return str(cv.get("id"))
        return None

    # --- conversation engagement --------------------------------------------
    async def _engage_conversation(self, perception: WorldPerceptionV1) -> None:
        """Drive Orion into conversations (Orion has no town-AI to do it):
        accept invites, physically walk to the partner to reach `participating`,
        and opportunistically initiate with a nearby player when idle. Fail-open."""
        own = (self._orion_player_id or "").strip()
        if not own:
            return
        convo = perception.active_conversation
        if convo:
            status = convo.get("status")
            cid = str(convo.get("conversation_id") or "")
            other = convo.get("other") or {}
            if status == "invited" and cid:
                try:
                    await asyncio.to_thread(
                        aitown_client.accept_invite,
                        player_id=own, conversation_id=cid, world_id=self._world_id or None,
                    )
                except Exception:
                    logger.debug("embodiment_accept_invite_failed", exc_info=True)
                return
            if status == "walkingOver" and other.get("player_id"):
                # Move onto the partner; AI Town paths us adjacent, which trips the
                # walkingOver -> participating transition once we're within range.
                intent = build_intent(
                    kind="approach_player", source="involuntary",
                    reason="walk to conversation partner", correlation_id=str(uuid4()),
                    player_id=own, ref=str(other.get("player_id")),
                )
                await self._actuate(intent, now=_utcnow())
            elif status == "participating":
                # The engine's Conversation.tick only orients participants who are
                # NOT pathfinding. If Orion reached `participating` still carrying a
                # lingering path, it never faces the partner. Issue one stop (moveTo
                # own current tile) per conversation so the next tick orients Orion.
                # Speech itself is driven by the turn-taking gate in `_speak_once`.
                await self._face_partner_if_pathfinding(perception, own, cid)
            return
        await self._maybe_initiate_conversation(perception)

    async def _face_partner_if_pathfinding(
        self, perception: WorldPerceptionV1, own: str, cid: str
    ) -> None:
        """Clear Orion's lingering path exactly once per conversation so the engine
        orients it toward the partner.

        Fires only when ``perception.pathfinding`` is truthy — issuing a stop when
        Orion is already stopped would fight the engine's own post-transition move
        and spam inputs.         Guarded per-conversation via ``_faced_conversations``. The
        stop is a ``moveTo`` to Orion's own *current* position — a zero-length path
        the engine resolves to an immediate stop (no micro-move that would keep
        ``pathfinding`` truthy), after which the next ``Conversation.tick`` sets
        ``facing``. Fail-open: never raises into the perception loop.
        """
        if not cid or cid in self._faced_conversations:
            return
        if not getattr(perception, "pathfinding", False):
            return
        pos = perception.position or {}
        try:
            tx = float(pos["x"])
            ty = float(pos["y"])
        except (KeyError, TypeError, ValueError):
            return
        # Mark before actuating so a transient failure still can't re-fire every tick
        # (we prefer under-facing once over spamming stops at the shared engine).
        self._faced_conversations.add(cid)
        try:
            await asyncio.to_thread(
                aitown_client.move_to,
                player_id=own, x=tx, y=ty, world_id=self._world_id or None,
            )
            logger.info(
                "embodiment_face_partner_stop convo=%s pos=(%.2f,%.2f)", cid, tx, ty
            )
        except Exception:
            logger.exception("embodiment_face_partner_stop_failed convo=%s", cid)

    async def _maybe_initiate_conversation(self, perception: WorldPerceptionV1) -> None:
        dist = float(getattr(self._settings, "social_initiate_distance", 0.0) or 0.0)
        if dist <= 0:
            return
        now = _utcnow()
        last = self._last_social_attempt_at
        if last is not None and (now - last).total_seconds() < float(self._social_cooldown_sec):
            return
        candidates = [
            n for n in (perception.nearby_players or [])
            if n.get("player_id") and float(n.get("distance", 1e9)) <= dist
        ]
        if not candidates:
            return
        self._last_social_attempt_at = now
        target = candidates[0]
        intent = build_intent(
            kind="start_conversation", source="involuntary",
            reason="approach nearby player", correlation_id=str(uuid4()),
            player_id=self._orion_player_id or None, ref=str(target.get("player_id")),
        )
        await self._actuate(intent, now=now)

    async def _perception_loop(self) -> None:
        interval = self._settings.perception_interval_sec
        while not self._stop.is_set():
            try:
                perception = await self._emit_perception_once()
            except Exception:
                logger.exception("embodiment_perception_loop_failed")
                perception = None
            self._maybe_log_heartbeat(perception)
            if perception is not None and getattr(self._settings, "memory_enabled", False):
                try:
                    await self._journal_from_perception(perception)
                except Exception:
                    logger.exception("embodiment_memory_loop_failed")
            if perception is not None and getattr(self._settings, "social_enabled", False):
                try:
                    await self._engage_conversation(perception)
                except Exception:
                    logger.exception("embodiment_social_loop_failed")
            if perception is not None and getattr(self._settings, "speech_enabled", False):
                try:
                    await self._speak_once(perception)
                except Exception:
                    logger.exception("embodiment_speech_loop_failed")
            # Don't wander off while engaging a conversation — but only when social
            # engagement is enabled. With it off, a town-initiated invite would
            # otherwise freeze Orion (suppressed wander + no engagement).
            in_conversation = (
                getattr(self._settings, "social_enabled", False)
                and perception is not None
                and perception.active_conversation is not None
            )
            if not in_conversation:
                try:
                    await self._maybe_idle_wander(now=_utcnow())
                except Exception:
                    logger.exception("embodiment_idle_wander_failed")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                continue

    def _maybe_log_heartbeat(self, perception: Optional[WorldPerceptionV1]) -> None:
        """Emit a throttled INFO heartbeat summarizing the loop's current perception.

        A healthy loop otherwise logs nothing (all loop logs are exception-only), so
        `docker logs` looked dead even while working — which forced live DB probing to
        diagnose the void bug. This is the smallest observability seam: one line every
        ~30s with real perceived state, not fabricated cognition.
        """
        now = _utcnow()
        last = self._last_heartbeat_log_at
        if last is not None and (now - last).total_seconds() < 30.0:
            return
        self._last_heartbeat_log_at = now
        if perception is None:
            logger.info("embodiment_heartbeat perception=none")
            return
        convo = perception.active_conversation or {}
        other = convo.get("other") or {}
        logger.info(
            "embodiment_heartbeat player=%s nearby=%d active_convo=%s status=%s partner=%s facing_partner=%s",
            self._orion_player_id or "?",
            len(perception.nearby_players or []),
            convo.get("conversation_id") or "-",
            convo.get("status") or "-",
            other.get("player_id") or "-",
            convo.get("facing_partner"),
        )

    async def _maybe_idle_wander(self, *, now: datetime) -> None:
        """Self-driven involuntary wander when no move has actuated within
        ``idle_heartbeat_sec``. Off when the setting is 0. Keeps Orion alive when
        no external producer (C/D) is emitting, without fabricating cognition."""
        hb = float(getattr(self._settings, "idle_heartbeat_sec", 0.0) or 0.0)
        if hb <= 0 or not (self._orion_player_id or "").strip():
            return
        # Idle = no actuated move AND no wander attempt within the window. Gating on
        # the attempt (not just actuation) stops a non-actuated wander (noop/preempted)
        # from re-firing every perception tick and spamming the outcome channel.
        recent = max([t for t in (self._last_move_at, self._last_idle_wander_at) if t], default=None)
        if recent is not None and (now - recent).total_seconds() < hb:
            return
        self._last_idle_wander_at = now
        intent = build_intent(
            kind="wander", source="involuntary", reason="idle heartbeat wander",
            correlation_id=str(uuid4()), player_id=self._orion_player_id or None,
        )
        await self._actuate(intent, now=now)

    # --- speech (cortex-generated town utterances) --------------------------
    async def _speak_once(self, perception: WorldPerceptionV1) -> Optional[str]:
        """Gated, fail-open speech pass. Own-agent only; one utterance per convo."""
        if not getattr(self._settings, "speech_enabled", False):
            return None
        own = (self._orion_player_id or "").strip()
        if not should_speak(perception, own):
            return None
        convo = perception.active_conversation or {}
        # Only speak once actually participating (not while invited/walkingOver).
        if (convo.get("status") or "") != "participating":
            return None
        convo_id = str(convo.get("conversation_id") or convo.get("id") or "")
        if not convo_id or convo_id in self._speaking_conversations:
            return None
        # Turn-taking: speak only when it's Orion's turn — the last message is from
        # someone else, or the conversation is empty and Orion hasn't opened it yet.
        # Speaking when Orion already spoke last (or re-opening an already-opened
        # convo whose transcript we failed to read) causes self-echo flood loops.
        messages = convo.get("messages") or []
        if messages:
            if str(messages[-1].get("author_id") or "") == own:
                return None
        elif convo_id in self._opened_conversations:
            return None

        prompt = build_speech_prompt(perception, own)
        # Hold the conversation as in-flight through both the utterance request AND
        # the injection so a later perception tick cannot double-inject.
        self._speaking_conversations.add(convo_id)
        try:
            try:
                reply = await self._request_utterance(
                    prompt, correlation_id=str(uuid4()), convo_id=convo_id
                )
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
            if not messages:
                # Opened the conversation; don't re-open on a later empty transcript.
                self._opened_conversations.add(convo_id)
            # Record Orion's contribution so the journal gate sees a real exchange
            # when this conversation later completes.
            if convo_id == getattr(self, "_active_conversation_id", None):
                self._active_conversation_utterances = (
                    getattr(self, "_active_conversation_utterances", 0) + 1
                )
            return reply
        finally:
            self._speaking_conversations.discard(convo_id)

    async def _request_utterance(
        self, prompt: str, *, correlation_id: str, convo_id: Optional[str] = None
    ) -> str:
        """Generate a town utterance. Dispatcher: prefer the unified turn (full
        cognition pass over the hub saga); fall back to the quick cortex rail on
        timeout/error/empty. Fail-open -> ''.

        Set ``speech_unified_enabled`` false to force the legacy quick-only path.
        """
        if getattr(self._settings, "speech_unified_enabled", False):
            session_id = f"{self._settings.unified_session_prefix}:{convo_id or 'orion'}"
            try:
                unified = await self._request_utterance_unified(
                    prompt, correlation_id=correlation_id, session_id=session_id
                )
                if unified.strip():
                    return unified
                # A non-final/empty frame already logged its discriminating reason.
            except Exception as exc:
                logger.info(
                    "embodiment_speech_unified_fallback reason=%s corr=%s",
                    type(exc).__name__, correlation_id,
                )
        return await self._request_utterance_quick(prompt, correlation_id=correlation_id)

    async def _request_utterance_unified(
        self, prompt: str, *, correlation_id: str, session_id: str
    ) -> str:
        """Route the utterance through the hub-only unified turn saga
        (``POST /api/chat`` with ``mode=orion``). Returns the final text on success;
        returns "" to signal fallback on any non-final/empty frame, logging a
        discriminating ``reason`` (``turn_error`` vs ``turn_deferred`` vs
        ``non_final:<type>`` vs ``empty``) so an operator can tell a real cognition
        failure from a benign quiet turn. Network/JSON errors propagate to the
        dispatcher, which logs one fallback line."""
        import urllib.request

        body = json.dumps(
            {
                "mode": "orion",
                "session_id": session_id,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode("utf-8")

        def _post() -> dict:
            req = urllib.request.Request(
                self._settings.hub_chat_url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(
                req, timeout=float(self._settings.unified_timeout_sec)
            ) as resp:
                raw = resp.read().decode("utf-8")
            return json.loads(raw)

        frame = await asyncio.to_thread(_post)
        frame_type = frame.get("type") if isinstance(frame, dict) else None
        if frame_type != "final":
            reason = frame_type if frame_type in ("turn_error", "turn_deferred") else f"non_final:{frame_type}"
            logger.info(
                "embodiment_speech_unified_fallback reason=%s corr=%s", reason, correlation_id
            )
            return ""
        text = str(frame.get("llm_response") or "").strip()
        if not text:
            logger.info(
                "embodiment_speech_unified_fallback reason=empty corr=%s", correlation_id
            )
        return text

    async def _request_utterance_quick(self, prompt: str, *, correlation_id: str) -> str:
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
        """Inject an utterance into the AI Town conversation (startTyping -> writeMessage).

        Input/mutation names follow the AI Town canonical schema (upstream not
        vendored here). See TODO on the input constants above.

        CRITICAL (the "void" fix): the engine only advances conversation state
        (``conversation.lastMessage`` + ``numMessages``) — which a partner agent's
        tick reads to decide it's their turn — on a valid ``finishSendingMessage``
        input carrying a numeric ``timestamp`` (ms), NOT ``messageUuid``.

        ``messages:writeMessage`` already enqueues that ``finishSendingMessage``
        server-side with ``timestamp: Date.now()`` (see upstream ``convex/messages.ts``),
        so we do NOT send a second one from here. Doing so previously (a) double-counted
        ``numMessages`` for Orion's turns and, when sent with the wrong args
        (``messageUuid`` and no ``timestamp``), (b) poisoned the shared engine: the
        malformed input built ``lastMessage={author}`` with no timestamp, which fails
        the ``serializedConversation`` validator in ``saveWorld`` and crashes every
        ``runStep`` — freezing the whole town until the stale input backlog was purged.
        """
        message_uuid = str(uuid4())
        wid = self._world_id or None
        aitown_client.send_input(
            name=START_TYPING_INPUT,
            args={"playerId": own_player_id, "conversationId": conversation_id, "messageUuid": message_uuid},
            world_id=wid,
        )
        # writeMessage writes the row AND enqueues the canonical finishSendingMessage
        # (with a numeric timestamp) itself — this single call advances the turn.
        aitown_client.convex_mutation(
            WRITE_MESSAGE_MUTATION,
            {
                "worldId": self._world_id,
                "conversationId": conversation_id,
                "messageUuid": message_uuid,
                # Deployed messages:writeMessage validator requires `playerId` (not `author`).
                "playerId": own_player_id,
                "text": text,
            },
        )
