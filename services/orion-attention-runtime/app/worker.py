from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from uuid import uuid4

from orion.attention.field_attention.builder import build_attention_frame
from orion.attention.field_attention.goal_provenance import (
    DominanceStreak,
    top_node_substrate_target,
    update_dominance_streak,
)
from orion.attention.field_attention.policy import load_attention_policy
from orion.attention.field_attention.selectors import PREDICTION_ERROR_NATIVE_TARGETS
from orion.schemas.field_attention_frame import FieldAttentionFrameV1
from orion.schemas.field_goal import FieldGoalProvenanceV1

from app.health_monitor import HealthMonitor
from app.settings import get_settings
from app.store import AttentionRuntimeStore

logger = logging.getLogger("orion.attention.runtime")


class AttentionRuntimeWorker:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._store = AttentionRuntimeStore(self._settings.postgres_uri)
        self._policy = load_attention_policy(Path(self._settings.attention_policy_path))
        self._health_monitor = HealthMonitor(self._store, self._settings)
        self._stop = asyncio.Event()
        # Field-native goal-provenance producer (SSP sec6 Objective 3) -- see
        # docs/superpowers/specs/2026-07-30-goal-provenance-and-decision-lattice-
        # observability-design.md. In-memory only: resets to a cold streak on
        # restart, a safe failure mode (worst case is a brief warm-up delay
        # before the next real emission, not a wrong one).
        self._node_streak = DominanceStreak()
        self._bus = None
        self._poll_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        s = self._settings
        if s.enable_goal_provenance_producer and s.orion_bus_enabled:
            from orion.core.bus.async_service import OrionBusAsync

            self._bus = OrionBusAsync(url=s.orion_bus_url)
            await self._bus.connect()
        self._poll_task = asyncio.create_task(self._poll_loop(), name="attention-runtime-poll")
        asyncio.create_task(self._prune_loop(), name="attention-runtime-prune")
        asyncio.create_task(self._health_loop(), name="attention-runtime-health")

    async def stop(self) -> None:
        self._stop.set()
        # Await the poll loop (the only loop that can be mid-publish) before
        # closing the bus -- otherwise a goal-provenance publish in flight when
        # stop() runs can have its connection torn down mid-call, which
        # publish_with_reconnect would silently paper over by reconnecting
        # right after an intentional close. The loop's own stop_event wait has
        # a <=1.2s timeout, so this bounds cleanly.
        if self._poll_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._poll_task
        if self._bus is not None:
            await self._bus.close()

    async def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                goal = await asyncio.to_thread(self._tick)
                if goal is not None:
                    await self._publish_goal(goal)
            except Exception:
                logger.exception("attention_runtime_tick_failed")
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=float(self._settings.attention_poll_interval_sec),
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    def _prune_tick(self) -> None:
        retention = float(self._settings.attention_frame_retention_hours)
        if retention <= 0:
            return
        deleted = self._store.prune_attention_frames(retention_hours=retention)
        if deleted:
            logger.info(
                "attention_frames_pruned deleted=%d retention_hours=%.1f", deleted, retention
            )

    async def _prune_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.to_thread(self._prune_tick)
            except Exception:
                logger.exception("attention_frame_prune_failed")
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=float(self._settings.attention_frame_prune_interval_sec),
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    async def _health_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.to_thread(self._health_monitor.run_tick)
            except Exception:
                logger.exception("attention_runtime_health_check_failed")
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=float(self._settings.health_check_interval_sec),
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    def _tick(self) -> FieldGoalProvenanceV1 | None:
        if not self._settings.enable_attention_runtime:
            return None

        field = self._store.load_latest_field()
        if field is None:
            return None

        if self._store.load_attention_frame_for_field_tick(field.tick_id) is not None:
            return None

        previous = self._store.load_latest_attention_frame()
        # Candidate A (precision-weighted salience, 2026-07-30): real
        # historical error series, per qualified target, fetched fresh every
        # tick -- see select_node_targets' own docstring for why this can't
        # be pre-computed once (the pure builder has no DB access).
        histories = {
            node_id: self._store.load_prediction_error_history(
                reducer_key=reducer_key, limit=self._settings.prediction_error_history_limit
            )
            for node_id, reducer_key in PREDICTION_ERROR_NATIVE_TARGETS.items()
        }
        frame = build_attention_frame(
            field=field,
            policy=self._policy,
            prediction_error_histories=histories,
            previous_frame=previous,
        )
        self._store.save_attention_frame(frame)
        logger.info(
            "attention_frame_saved frame_id=%s tick_id=%s salience=%.3f",
            frame.frame_id,
            field.tick_id,
            frame.overall_salience,
        )
        return self._maybe_build_goal(frame)

    def _maybe_build_goal(self, frame: FieldAttentionFrameV1) -> FieldGoalProvenanceV1 | None:
        if not self._settings.enable_goal_provenance_producer or self._bus is None:
            return None
        winner = top_node_substrate_target(frame)
        winner_id = winner.target_id if winner is not None else None
        self._node_streak, should_emit = update_dominance_streak(
            self._node_streak, winner_id, min_streak=self._settings.goal_provenance_min_streak
        )
        if not should_emit or winner is None:
            return None
        return FieldGoalProvenanceV1(
            subject="attention",
            model_layer="field_attention",
            entity_id=winner.target_id,
            kind="memory.field_goals.proposed.v1",
            field_target_id=winner.target_id,
            target_kind=winner.target_kind,
            salience_score=winner.salience_score,
            source_field_tick_id=frame.source_field_tick_id,
            source_attention_frame_id=frame.frame_id,
            priority=winner.salience_score,
            provenance={"intake_channel": "internal.attention_runtime"},
        )

    async def _publish_goal(self, goal: FieldGoalProvenanceV1) -> None:
        if self._bus is None:
            return
        try:
            from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
            from orion.core.bus.resilience import publish_with_reconnect

            env = BaseEnvelope(
                kind=goal.kind,
                source=ServiceRef(
                    name=self._settings.service_name,
                    version=self._settings.service_version,
                    node=self._settings.node_name,
                ),
                correlation_id=uuid4(),
                payload=goal.model_dump(mode="json"),
            )
            await publish_with_reconnect(
                self._bus,
                self._settings.channel_goal_proposal,
                env,
                log_label="attention_runtime_goal_provenance",
            )
            logger.info(
                "field_goal_provenance_published artifact_id=%s field_target_id=%s "
                "salience=%.3f streak=%d",
                goal.artifact_id,
                goal.field_target_id,
                goal.salience_score,
                self._node_streak.count,
            )
        except Exception:
            logger.exception("field_goal_provenance_publish_failed")
