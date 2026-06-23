from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.resilience import publish_with_reconnect
from orion.identity.snapshot import build_identity_snapshot
from orion.schemas.self_state import SelfStateV1
from orion.self_state.builder import build_self_state
from orion.self_state.policy import load_self_state_policy
from orion.self_state.prediction import build_next_cycle_prediction, compute_prediction_errors

from app.settings import get_settings
from app.store import SelfStateRuntimeStore

logger = logging.getLogger("orion.self_state.runtime")

_bus: Optional[OrionBusAsync] = None
_IDENTITY_SNAPSHOT_EVERY_N = 10


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

    async def start(self) -> None:
        asyncio.create_task(self._poll_loop(), name="self-state-runtime-poll")

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

        state = build_self_state(
            field=field,
            attention=attention,
            policy=self._policy,
            previous_self_state=previous,
            enable_transport_influence=self._settings.enable_transport_self_state_influence,
        )

        if prev_prediction is not None:
            state.prediction_error_scores = compute_prediction_errors(state, prev_prediction)

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
