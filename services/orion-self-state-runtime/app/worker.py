from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

from orion.autonomy.deviation_gate import DeviationGate
from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.resilience import publish_with_reconnect
from orion.identity.snapshot import build_identity_snapshot
from orion.schemas.embodiment import WorldPerceptionV1
from orion.schemas.self_state import SelfStateV1
from orion.self_state.builder import build_self_state
from orion.self_state.deviation import observe_dimension_deviation
from orion.self_state.policy import load_self_state_policy
from orion.self_state.prediction import (
    build_next_cycle_prediction,
    compute_overall_surprise,
    compute_prediction_errors,
)

from app.settings import get_settings
from app.store import SelfStateRuntimeStore

logger = logging.getLogger("orion.self_state.runtime")

_bus: Optional[OrionBusAsync] = None
_IDENTITY_SNAPSHOT_EVERY_N = 10
# Age gate for the embodied-perception grounding input: stale perception must
# not keep asserting "I am near X" long after the town view went quiet.
_PERCEPTION_MAX_AGE_SEC = 600.0


def set_publisher_bus(bus: OrionBusAsync) -> None:
    global _bus
    _bus = bus


def _svc_ref(settings) -> ServiceRef:
    return ServiceRef(name=settings.service_name, version=settings.service_version)


class SelfStateRuntimeWorker:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._store = SelfStateRuntimeStore(self._settings.postgres_uri)
        self._policy = load_self_state_policy(Path(self._settings.self_state_policy_path))
        self._stop = asyncio.Event()
        self._tick_count = 0
        # Latest embodied town perception cached off the bus (best-effort,
        # age-gated observability input). Default-off; None until observed.
        self._latest_perception: Optional[WorldPerceptionV1] = None
        self._latest_perception_at: Optional[datetime] = None
        # Per-dimension deviation baseline (2026-07-12, Phase 2 measurement
        # pass -- log-only, no schema field yet). In-memory, per-process:
        # a restart resets every dimension's learned baseline to cold-start,
        # same accepted limitation as DriveEngine's pressure store elsewhere.
        self._deviation_gate = DeviationGate()

    async def start(self) -> None:
        asyncio.create_task(self._poll_loop(), name="self-state-runtime-poll")
        asyncio.create_task(self._prune_loop(), name="self-state-runtime-prune")
        # Gated + fail-open: only subscribe to town perception when enabled and
        # the publisher bus is live.
        if (
            self._settings.embodiment_perception_selfstate_enabled
            and _bus is not None
            and _bus.enabled
        ):
            asyncio.create_task(
                self._perception_listener_loop(), name="self-state-runtime-perception"
            )

    async def stop(self) -> None:
        self._stop.set()

    async def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                state = await asyncio.to_thread(self._tick)
                if state is not None and _bus is not None and _bus.enabled:
                    await self._publish_self_state(state)
            except Exception:
                logger.exception("self_state_runtime_tick_failed")
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=float(self._settings.self_state_poll_interval_sec),
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    def _prune_tick(self) -> None:
        retention = float(self._settings.self_state_retention_hours)
        if retention <= 0:
            return
        deleted = self._store.prune_history(retention_hours=retention)
        if deleted:
            logger.info(
                "self_state_history_pruned deleted=%d retention_hours=%.1f",
                deleted,
                retention,
            )

    async def _prune_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.to_thread(self._prune_tick)
            except Exception:
                logger.exception("self_state_prune_failed")
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=float(self._settings.self_state_prune_interval_sec),
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    def cache_perception(
        self, perception: WorldPerceptionV1, *, now: Optional[datetime] = None
    ) -> bool:
        """Cache the latest town perception as a best-effort observability input.

        Gated by ``embodiment_perception_selfstate_enabled``; when disabled the
        perception is ignored (returns ``False``). Fail-open by construction.
        """
        if not self._settings.embodiment_perception_selfstate_enabled:
            return False
        self._latest_perception = perception
        self._latest_perception_at = now or datetime.now(timezone.utc)
        return True

    def perception_input(self, *, now: Optional[datetime] = None) -> Optional[dict]:
        """Return the age-gated embodied-perception grounding signal, or ``None``.

        Mirrors the ``hub_presence``/``attention_broadcast`` observability inputs:
        returns ``None`` when disabled, unset, or stale so ``build_self_state``
        degrades to defaults.
        """
        if not self._settings.embodiment_perception_selfstate_enabled:
            return None
        perception = self._latest_perception
        observed_at = self._latest_perception_at
        if perception is None or observed_at is None:
            return None
        ref = now or datetime.now(timezone.utc)
        if observed_at.tzinfo is None:
            observed_at = observed_at.replace(tzinfo=timezone.utc)
        if (ref - observed_at).total_seconds() > _PERCEPTION_MAX_AGE_SEC:
            return None
        return {
            "player_id": perception.player_id,
            "position": dict(perception.position or {}),
            "nearby_count": len(perception.nearby_players or []),
            "as_of": observed_at.isoformat(),
        }

    async def _perception_listener_loop(self) -> None:
        channel = self._settings.embodiment_channel_perception
        logger.info("self_state_embodiment_perception subscribing channel=%s", channel)
        try:
            async with _bus.subscribe(channel) as pubsub:
                while not self._stop.is_set():
                    try:
                        msg = await asyncio.wait_for(
                            pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0),
                            timeout=1.2,
                        )
                    except asyncio.TimeoutError:
                        continue
                    except asyncio.CancelledError:
                        break
                    if not msg or msg.get("type") not in ("message", "pmessage"):
                        continue
                    try:
                        self._handle_perception_message(msg)
                    except Exception:
                        logger.exception("self_state_embodiment_perception_handle_failed")
        except asyncio.CancelledError:
            raise
        finally:
            logger.info("self_state_embodiment_perception stopped channel=%s", channel)

    def _handle_perception_message(self, raw_msg: dict) -> None:
        decoded = _bus.codec.decode(raw_msg.get("data"))
        if not decoded.ok:
            logger.warning("self_state_embodiment_perception_decode_failed: %s", decoded.error)
            return
        try:
            perception = WorldPerceptionV1.model_validate(decoded.envelope.payload or {})
        except ValueError as exc:
            logger.error("self_state_embodiment_perception_invalid err=%s", exc)
            return
        self.cache_perception(perception)

    async def _publish_self_state(self, state: SelfStateV1) -> None:
        envelope = BaseEnvelope(
            kind="substrate.self_state.v1",
            source=_svc_ref(self._settings),
            correlation_id=uuid4(),
            payload=state.model_dump(mode="json"),
        )
        try:
            await publish_with_reconnect(
                _bus,
                self._settings.channel_substrate_self_state,
                envelope,
                log_label="self_state_publish",
            )
        except Exception:
            logger.exception(
                "self_state_publish_failed self_state_id=%s", state.self_state_id
            )

    def _tick(self) -> Optional[SelfStateV1]:
        if not self._settings.enable_self_state_runtime:
            return None

        attention = self._store.load_latest_attention_frame()
        if attention is None:
            return None

        if self._store.load_self_state_for_attention_frame(attention.frame_id) is not None:
            return None

        field = self._store.load_field_for_tick(attention.source_field_tick_id)
        if field is None:
            logger.warning(
                "self_state_skip_missing_field tick_id=%s frame_id=%s",
                attention.source_field_tick_id,
                attention.frame_id,
            )
            return None

        previous = self._store.load_latest_self_state()
        if previous is not None:
            if previous.self_state_policy_id != self._policy.policy_id:
                previous = None
            else:
                prev_ts = previous.generated_at
                if prev_ts.tzinfo is None:
                    prev_ts = prev_ts.replace(tzinfo=timezone.utc)
                if (
                    datetime.now(timezone.utc) - prev_ts
                ).total_seconds() > self._settings.self_state_max_previous_age_sec:
                    previous = None

        prev_prediction = self._store.load_latest_self_state_prediction()

        # Best-effort observability inputs: the loaders never raise and return
        # None when absent/stale, so build_self_state degrades to defaults.
        attention_broadcast = self._store.load_latest_attention_broadcast()
        hub_presence = self._store.load_hub_presence()

        # Fold the embodied "I am near X" grounding signal into hub_presence so
        # it threads through build_self_state (and is persisted onto
        # SelfStateV1.hub_presence as an inspectable projection field) without
        # touching the shared builder. It does not yet drive any dimension score
        # or condition — this is the first observable seam, not felt-state input.
        # Gated + age-gated inside perception_input; None when absent/disabled.
        perception_input = self.perception_input()
        if perception_input is not None:
            hub_presence = dict(hub_presence or {})
            hub_presence["embodiment"] = perception_input

        state = build_self_state(
            field=field,
            attention=attention,
            policy=self._policy,
            previous_self_state=previous,
            enable_transport_influence=self._settings.enable_transport_self_state_influence,
            attention_broadcast=attention_broadcast,
            hub_presence=hub_presence,
        )

        if prev_prediction is not None:
            state.prediction_error_scores = compute_prediction_errors(state, prev_prediction)
            state.overall_surprise = compute_overall_surprise(state.prediction_error_scores)

        self._log_deviation_probe(state)

        self._store.save_self_state(state)

        next_prediction = build_next_cycle_prediction(state)
        self._store.save_self_state_prediction(next_prediction)
        logger.info(
            "self_state_saved self_state_id=%s frame_id=%s condition=%s intensity=%.3f",
            state.self_state_id,
            attention.frame_id,
            state.overall_condition,
            state.overall_intensity,
        )

        self._tick_count += 1
        if self._tick_count % _IDENTITY_SNAPSHOT_EVERY_N == 0:
            self._maybe_emit_identity_snapshot(state)

        return state

    def _log_deviation_probe(self, state: SelfStateV1) -> None:
        """Measurement-only (Phase 2, 2026-07-12): log each dimension's
        deviation-from-its-own-baseline impulse alongside the confidence
        value already on the wire, so the two can be compared on real live
        data before deciding whether channel_dimension_confidence() should
        be replaced by this variance-based mechanism instead
        (orion/self_state/scoring.py's spread-based formula was flagged as
        cruder than DeviationGate's approach in the Phase 1 PR report).
        Never raises: a failure here must not block persisting self_state.
        Toggleable via SELF_STATE_DEVIATION_PROBE_ENABLED without a redeploy
        in case this proves noisy at production tick volume.
        """
        if not self._settings.self_state_deviation_probe_enabled:
            return
        try:
            impulses = observe_dimension_deviation(self._deviation_gate, state, self._policy)
            confidences = {
                dim_id: round(state.dimensions[dim_id].confidence, 4) for dim_id in impulses
            }
            logger.info(
                "self_state_deviation_probe self_state_id=%s deviation=%s confidence=%s",
                state.self_state_id,
                {k: round(v, 4) for k, v in impulses.items()},
                confidences,
            )
        except Exception:
            logger.exception("self_state_deviation_probe_failed")

    def _maybe_emit_identity_snapshot(self, state) -> None:  # type: ignore[type-arg]
        try:
            ranked = sorted(state.dimensions.items(), key=lambda kv: kv[1].score, reverse=True)
            dominant_drive = ranked[0][0] if ranked else "unknown"
            active_drives = [dim_id for dim_id, _ in ranked[:3]]
            key_unknowns = [dim_id for dim_id, dim in ranked if dim.score < 0.2][:5]
            snapshot = build_identity_snapshot(
                self_state=state,
                dominant_drive=dominant_drive,
                active_drives=active_drives,
                key_unknowns=key_unknowns,
            )
            self._store.save_identity_snapshot(snapshot)
            logger.info(
                "identity_snapshot_saved snapshot_id=%s condition=%s drive=%s",
                snapshot.snapshot_id,
                snapshot.self_state_condition,
                snapshot.dominant_drive,
            )
        except Exception:
            logger.exception("identity_snapshot_emit_failed")
