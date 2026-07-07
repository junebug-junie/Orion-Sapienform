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
from orion.schemas.embodiment import (
    EMBODIMENT_OUTCOME_KIND,
    EMBODIMENT_PERCEPTION_KIND,
    EmbodimentIntentV1,
    EmbodimentOutcomeV1,
    WorldPerceptionV1,
)

logger = logging.getLogger("orion.embodiment.worker")

# AI Town canonical conversation inputs. The upstream convex/aiTown/inputs.ts is
# NOT vendored in this checkout, so these follow the AI Town canonical schema.
# TODO(embodiment): confirm names/args against upstream when it is vendored.
START_CONVERSATION_INPUT = "startConversation"


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
        self._last_conversation_start: Optional[datetime] = None
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
            return EmbodimentOutcomeV1(
                intent_correlation_id=intent.correlation_id, source=intent.source,
                status="actuated", reason=f"start_conversation ({result.reason})",
                player_id=player_id, resolved_destination=result.destination, send_input_ok=True,
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
                await self._emit_perception_once()
            except Exception:
                logger.exception("embodiment_perception_loop_failed")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                continue
