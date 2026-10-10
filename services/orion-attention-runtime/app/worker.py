from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from orion.attention.field_attention.builder import build_attention_frame
from orion.attention.field_attention.candidate_precision_weighted import (
    NODE_TARGET_PREDICTION_ERROR_EWMA_ALPHA,
    NODE_TARGET_PREDICTION_ERROR_MIN_VARIANCE,
)
from orion.attention.field_attention.goal_provenance import (
    qualified_node_targets,
    DominanceStreak,
    top_node_substrate_target,
    update_dominance_streak,
)
from orion.attention.field_attention.policy import load_attention_policy
from orion.attention.pe_history_cache import NodePeHistoryCache
from orion.attention.field_attention.selectors import PREDICTION_ERROR_NATIVE_TARGETS
from orion.schemas.field_attention_frame import FieldAttentionFrameV1
from orion.schemas.field_state import FieldStateV1
from orion.schemas.field_goal import FieldGoalProvenanceV1

from app.health_monitor import HealthMonitor
from app.settings import get_settings
from app.store import AttentionRuntimeStore

logger = logging.getLogger("orion.attention.runtime")

_VISION_ORGAN_NODE = "node:substrate.vision_organ"


def _camera_absent_reason(field: FieldStateV1) -> str | None:
    """The field carries the vision organ's `vision_frame_staleness` (1.0 = no
    camera delivering; key dropped = unmeasured); the rule itself is the one
    shared with the broadcast (`perception_absent_reason`). Embedding-only
    staleness is not visible here: perception then writes 0.0, reads quiet,
    and cannot win either."""
    from orion.attention.world_first import perception_absent_reason

    vec = field.node_vectors.get(_VISION_ORGAN_NODE) or {}
    return perception_absent_reason(
        vision_frame_staleness=vec.get("vision_frame_staleness"),
        vision_measured="vision_frame_staleness" in vec,
    )


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
        # The field tick the last saved frame was built from (#2534 decision 2).
        self._last_field: FieldStateV1 | None = None
        # World-first attention (spec 2026-10-07 section A): this process has
        # no magnitudes of its own, so it keeps a 7-day window of the
        # substrate's stored prediction-error readings.
        self._pe_history = NodePeHistoryCache()
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

    def _previous_field_for(self, previous: FieldAttentionFrameV1 | None) -> FieldStateV1 | None:
        """The field tick `previous` was built from: the one this worker saw last
        tick (normal case, no query), else one primary-key read (after a
        restart). Best effort: None keeps the plain proxy diff for this tick."""
        if previous is None:
            return None
        cached = self._last_field
        if cached is not None and cached.tick_id == previous.source_field_tick_id:
            return cached
        try:
            return self._store.load_field_for_tick(previous.source_field_tick_id)
        except Exception:
            logger.warning("attention_previous_field_load_failed", exc_info=True)
            return None

    def _tick(self) -> FieldGoalProvenanceV1 | None:
        if not self._settings.enable_attention_runtime:
            return None

        field = self._store.load_latest_field()
        if field is None:
            return None

        if self._store.load_attention_frame_for_field_tick(field.tick_id) is not None:
            return None

        previous = self._store.load_latest_attention_frame()
        previous_field = self._previous_field_for(previous)
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
        now = datetime.now(timezone.utc)
        candidates = (
            self._world_first_candidates(field, now)
            if self._settings.attention_world_first_enabled
            else None
        )
        frame = build_attention_frame(
            field=field,
            policy=self._policy,
            prediction_error_baselines=baselines,
            previous_frame=previous,
            previous_field=previous_field,
            now=now,
            world_first_candidates=candidates,
        )
        goal = self._maybe_build_goal(frame)
        self._last_field = field
        winner = frame.dominant_targets[0] if frame.dominant_targets else None
        logger.info(
            "attention_frame_saved frame_id=%s tick_id=%s salience=%.3f world_first=%s "
            "winner=%s",
            frame.frame_id,
            field.tick_id,
            frame.overall_salience,
            candidates is not None,
            winner.target_id if winner is not None else "none",
        )
        return goal

    def _world_first_candidates(self, field: FieldStateV1, now: datetime) -> list:
        """Every source's bid this tick, each scored against its own 7 days.

        Never raises: a source that cannot be read becomes an ABSENT
        candidate (silence is not calm), never a calm one and never a
        fallback to the old ranking.
        """
        from orion.attention.world_first import (
            CHAT_RATE_WINDOW,
            PERCEPTION_NODE_ID,
            SUBSTRATE_NODE_PREFIX,
            chat_candidate,
            node_candidate,
        )
        from orion.substrate.prediction_error_magnitude import WINDOW_7D

        magnitudes: dict = {}
        history_error: str | None = None
        try:
            self._pe_history.refresh(self._store.fetch_node_prediction_error_history, now=now)
            magnitudes = self._pe_history.magnitudes(now=now)
        except Exception as exc:  # noqa: BLE001
            history_error = f"prediction-error history unreadable ({type(exc).__name__})"
            logger.warning("attention_world_first_pe_history_failed err=%s", exc)

        node_ids = set(magnitudes) | {
            k for k in field.node_vectors if k.startswith(SUBSTRATE_NODE_PREFIX)
            and "prediction_error" in field.node_vectors[k]
        }
        candidates = []
        for node_id in sorted(node_ids):
            mag, observed_at, mid = magnitudes.get(node_id, (None, None, None))
            absent = history_error
            if node_id == PERCEPTION_NODE_ID and absent is None:
                absent = _camera_absent_reason(field)
            candidates.append(
                node_candidate(
                    node_id=node_id,
                    label=f"{node_id.removeprefix(SUBSTRATE_NODE_PREFIX)} prediction error",
                    magnitude=mag,
                    observed_at=observed_at,
                    now=now,
                    absent_reason=absent,
                    rank_percentile=mid,
                    event_decay=bool(getattr(self._settings, "attention_event_decay_enabled", True)),
                )
            )
        try:
            turns = self._store.fetch_chat_turn_times(now - WINDOW_7D - CHAT_RATE_WINDOW)
        except Exception as exc:  # noqa: BLE001
            logger.warning("attention_world_first_chat_read_failed err=%s", exc)
            turns = None
        candidates.append(chat_candidate(turns, now=now))
        return candidates

    def _maybe_build_goal(
        self, frame: FieldAttentionFrameV1
    ) -> FieldGoalProvenanceV1 | None:
        if not self._settings.enable_goal_provenance_producer or self._bus is None:
            self._store.save_attention_frame(frame)
            return None
        if self._node_streak is None:
            self._node_streak = self._store.load_node_dominance_streak()
        # With fewer than two qualified candidates no competition set can change
        # the answer, so the cross-service read is skipped (review finding).
        competing = (
            self._load_competition() if len(qualified_node_targets(frame)) >= 2 else None
        )
        winner = top_node_substrate_target(
            frame, competing=competing, current=self._node_streak.target_id
        )
        winner_id = winner.target_id if winner is not None else None
        next_streak, should_emit = update_dominance_streak(
            self._node_streak, winner_id, min_streak=self._settings.goal_provenance_min_streak
        )
        # Successful recording commits with its frame. Recorder failures are
        # isolated in the store so they cannot change goal selection/emission.
        if not self._store.save_attention_frame(
            frame, streak=next_streak,
            target_kind=winner.target_kind if winner else None,
            min_streak=self._settings.goal_provenance_min_streak,
        ):
            return None
        self._node_streak = next_streak

        if not should_emit or winner is None:
            return None
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
        # The bridge's receipt lives in this service's own log, deliberately
        # NOT on the schema: FieldGoalProvenanceV1 is extra="forbid" on three
        # consumers (substrate-runtime, world-pulse, spark-concept-induction),
        # and a producer-first deploy of two new fields dropped 186 goals live
        # on 2026-09-06 before the consumers could be rebuilt. The downstream
        # truth is the self-model's `voluntary_override_absent_reason`.
        logger.info(
            "field_goal_provenance_competition_read artifact_id=%s field_target_id=%s "
            "competition_read=%s competing=%s",
            goal.artifact_id,
            goal.field_target_id,
            (
                "unavailable" if competing is None
                else "in_competition" if winner.target_id in competing
                else "not_in_competition"
            ),
            ",".join(sorted(competing)) if competing else "",
        )
        return goal

    def _load_competition(self) -> set[str] | None:
        """The substrate competition's current open-loop node ids, or None.

        None on the kill switch, on a missing/stale projection, and on any
        read error -- the selector treats None as "unknown" and falls back to
        its pre-bridge top-1, so this read can never make the producer emit
        fewer goals than before. Logged, not raised: a goal tick must not die
        on a cross-service read.
        """
        if not self._settings.enable_goal_reads_competition:
            return None
        try:
            return self._store.load_competing_loop_refs(
                max_age_sec=self._settings.goal_competition_max_age_sec
            )
        except Exception as exc:  # noqa: BLE001 -- fail-open by contract
            logger.warning("goal_provenance_competition_read_failed err=%s", exc)
            return None

    async def _publish_envelope(
        self,
        *,
        kind: str,
        channel: str,
        payload: dict,
        log_label: str,
        failure_event: str,
    ) -> bool:
        """Publish goal provenance; a bus failure never escapes the tick loop."""
        if self._bus is None:
            return False
        try:
            from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
            from orion.core.bus.resilience import publish_with_reconnect

            env = BaseEnvelope(
                kind=kind,
                source=ServiceRef(
                    name=self._settings.service_name,
                    version=self._settings.service_version,
                    node=self._settings.node_name,
                ),
                correlation_id=uuid4(),
                payload=payload,
            )
            await publish_with_reconnect(self._bus, channel, env, log_label=log_label)
            return True
        except Exception:
            logger.exception(failure_event)
            return False

    async def _publish_goal(self, goal: FieldGoalProvenanceV1) -> None:
        published = await self._publish_envelope(
            kind=goal.kind,
            channel=self._settings.channel_goal_proposal,
            payload=goal.model_dump(mode="json"),
            log_label="attention_runtime_goal_provenance",
            failure_event="field_goal_provenance_publish_failed",
        )
        if published:
            logger.info(
                "field_goal_provenance_published artifact_id=%s field_target_id=%s "
                "salience=%.3f streak=%d",
                goal.artifact_id,
                goal.field_target_id,
                goal.salience_score,
                self._node_streak.count,
            )
