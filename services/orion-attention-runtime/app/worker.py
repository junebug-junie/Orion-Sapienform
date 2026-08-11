from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from uuid import uuid4

from orion.attention.field_attention.builder import build_attention_frame
from orion.attention.field_attention.candidate_precision_weighted import (
    NODE_TARGET_PREDICTION_ERROR_EWMA_ALPHA,
    NODE_TARGET_PREDICTION_ERROR_MIN_VARIANCE,
)
from orion.attention.field_attention.goal_provenance import (
    DominanceStreak,
    top_node_substrate_target,
    update_dominance_streak,
)
from orion.attention.field_attention.policy import load_attention_policy
from orion.attention.field_attention.selectors import PREDICTION_ERROR_NATIVE_TARGETS
from orion.schemas.field_attention_frame import FieldAttentionFrameV1
from orion.schemas.field_goal import DominanceStreakTickV1, FieldGoalProvenanceV1

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
        # observability-design.md. Persisted (2026-07-31 fix): lazy-loaded from
        # `substrate_goal_provenance_streak` on the first real tick (see
        # `_maybe_build_goal`) rather than always starting cold -- a restart no
        # longer truncates a genuinely-long streak back to zero. See
        # `AttentionRuntimeStore.load_node_dominance_streak`'s docstring for why
        # this stopped being an acceptable in-memory-only gap.
        self._node_streak: DominanceStreak | None = None
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
                goal, streak_tick = await asyncio.to_thread(self._tick)
                if goal is not None:
                    await self._publish_goal(goal)
                if streak_tick is not None:
                    await self._publish_streak_tick(streak_tick)
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

    def _tick(self) -> tuple[FieldGoalProvenanceV1 | None, DominanceStreakTickV1 | None]:
        if not self._settings.enable_attention_runtime:
            return None, None

        field = self._store.load_latest_field()
        if field is None:
            return None, None

        if self._store.load_attention_frame_for_field_tick(field.tick_id) is not None:
            return None, None

        previous = self._store.load_latest_attention_frame()
        # Candidate A (precision-weighted salience): real, persisted,
        # incrementally-updated EWMA baseline per qualified target, advanced by
        # whatever real new substrate_reduction_receipts rows landed since the
        # last tick (2026-07-30 fix -- see candidate_precision_weighted.py's
        # module docstring and orion/sentience_striving_program/README.md §12
        # for the live incident this replaces: the old per-tick raw-window
        # recompute let a target with as few as 2 real samples surviving the
        # ~30-minute retention window win a fully-confident-looking
        # salience_score=1.0). `observation_count` on the returned baseline is
        # a real cumulative count, immune to that retention pruner.
        baselines = {
            node_id: self._store.advance_node_prediction_error_baseline(
                target_id=node_id,
                reducer_key=reducer_key,
                alpha=NODE_TARGET_PREDICTION_ERROR_EWMA_ALPHA,
                min_variance=NODE_TARGET_PREDICTION_ERROR_MIN_VARIANCE,
                fetch_limit=self._settings.prediction_error_history_limit,
            )
            for node_id, reducer_key in PREDICTION_ERROR_NATIVE_TARGETS.items()
        }
        frame = build_attention_frame(
            field=field,
            policy=self._policy,
            prediction_error_baselines=baselines,
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

    def _maybe_build_goal(
        self, frame: FieldAttentionFrameV1
    ) -> tuple[FieldGoalProvenanceV1 | None, DominanceStreakTickV1 | None]:
        if not self._settings.enable_goal_provenance_producer or self._bus is None:
            return None, None
        if self._node_streak is None:
            self._node_streak = self._store.load_node_dominance_streak()
        winner = top_node_substrate_target(frame)
        winner_id = winner.target_id if winner is not None else None
        self._node_streak, should_emit = update_dominance_streak(
            self._node_streak, winner_id, min_streak=self._settings.goal_provenance_min_streak
        )
        self._store.save_node_dominance_streak(self._node_streak)

        streak_tick: DominanceStreakTickV1 | None = None
        if self._settings.enable_goal_provenance_streak_tick_telemetry:
            streak_tick = DominanceStreakTickV1(
                target_id=self._node_streak.target_id,
                streak_count=self._node_streak.count,
                min_streak_at_tick=self._settings.goal_provenance_min_streak,
                qualified=should_emit,
                source_field_tick_id=frame.source_field_tick_id,
                source_attention_frame_id=frame.frame_id,
            )

        if not should_emit or winner is None:
            return None, streak_tick
        goal = FieldGoalProvenanceV1(
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
        return goal, streak_tick

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

    async def _publish_streak_tick(self, streak_tick: DominanceStreakTickV1) -> None:
        if self._bus is None:
            return
        try:
            from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
            from orion.core.bus.resilience import publish_with_reconnect

            env = BaseEnvelope(
                kind="debug.attention.streak_tick.v1",
                source=ServiceRef(
                    name=self._settings.service_name,
                    version=self._settings.service_version,
                    node=self._settings.node_name,
                ),
                correlation_id=uuid4(),
                payload=streak_tick.model_dump(mode="json"),
            )
            await publish_with_reconnect(
                self._bus,
                self._settings.channel_goal_provenance_streak_tick,
                env,
                log_label="attention_runtime_streak_tick",
            )
        except Exception:
            # Debug telemetry: never let a publish failure here look like a real incident --
            # field_goal_provenance_publish_failed above stays the loud one.
            logger.debug("goal_provenance_streak_tick_publish_failed", exc_info=True)
