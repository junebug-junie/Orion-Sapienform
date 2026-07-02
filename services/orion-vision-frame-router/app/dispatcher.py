from __future__ import annotations

import asyncio
import os
import time
from typing import TYPE_CHECKING

from loguru import logger
from orion.core.bus.bus_schemas import BaseEnvelope
from orion.schemas.vision import VisionFramePointerPayload, VisionTaskResultPayload

from .envelopes import make_host_task_envelope
from .metrics import RouterMetrics
from .policy import FrameDispatchPolicy
from .settings import Settings
from .state import RouterState

if TYPE_CHECKING:
    from orion.core.bus.async_service import OrionBusAsync


class FrameDispatcher:
    def __init__(
        self,
        *,
        settings: Settings,
        policy: FrameDispatchPolicy,
        state: RouterState,
        metrics: RouterMetrics,
        bus: OrionBusAsync | None,
    ) -> None:
        self.settings = settings
        self.policy = policy
        self.state = state
        self.metrics = metrics
        self.bus = bus
        self._state_lock = asyncio.Lock()

    async def handle_frame_envelope(self, env: BaseEnvelope) -> None:
        try:
            await self._handle_frame_envelope_inner(env)
        except Exception as exc:
            self.metrics.last_error = f"frame_handler_error: {exc}"

    async def _handle_frame_envelope_inner(self, env: BaseEnvelope) -> None:
        try:
            frame = VisionFramePointerPayload.model_validate(env.payload)
        except Exception as exc:
            self.metrics.last_error = f"invalid_frame_payload: {exc}"
            return

        self.metrics.record_seen()
        camera_id = frame.camera_id or "unknown"
        image_path = (frame.image_path or "").strip()
        image_path_exists: bool | None = None
        if image_path and self.policy.require_image_path_exists(camera_id):
            image_path_exists = await asyncio.to_thread(os.path.isfile, image_path)

        async with self._state_lock:
            decision = self.policy.decide(
                env,
                self.state,
                now=time.time(),
                image_path_exists=image_path_exists,
            )
            if not decision.should_dispatch:
                self.metrics.record_skip(decision.reason)
                return

            task = self.policy.build_task_request(frame, env, decision)
            corr = str(env.correlation_id)
            reply_to = f"{self.settings.CHANNEL_REPLY_PREFIX}:{corr}"
            task_env = make_host_task_envelope(
                frame_env=env,
                frame=frame,
                task=task,
                service_name=self.settings.SERVICE_NAME,
                service_version=self.settings.SERVICE_VERSION,
                reply_to=reply_to,
            )

            if not self.settings.DRY_RUN and self.bus:
                await self.bus.publish(self.settings.CHANNEL_HOST_INTAKE, task_env)

            want_caption = bool((task.request or {}).get("want_caption"))
            logger.info(
                "[ROUTER] dispatch tier={} task_type={} want_caption={} camera_id={} stream_id={}",
                decision.dispatch_tier,
                task.task_type,
                want_caption,
                camera_id,
                frame.stream_id,
            )

            self.state.mark_dispatched(
                correlation_id=corr,
                camera_id=frame.camera_id or "unknown",
                image_path=frame.image_path or "",
                task_type=task.task_type,
                reply_to=reply_to,
                now=time.time(),
                frame_ts=frame.frame_ts,
            )
            self.metrics.record_dispatch()

    async def handle_reply_envelope(self, env: BaseEnvelope) -> None:
        try:
            await self._handle_reply_envelope_inner(env)
        except Exception as exc:
            self.metrics.last_error = f"reply_handler_error: {exc}"

    async def _handle_reply_envelope_inner(self, env: BaseEnvelope) -> None:
        try:
            result = VisionTaskResultPayload.model_validate(env.payload)
        except Exception as exc:
            self.metrics.last_error = f"invalid_reply_payload: {exc}"
            return

        corr = str(env.correlation_id)
        async with self._state_lock:
            cleared = self.state.clear_pending(corr, now=time.time())
            if not cleared:
                return
            self.metrics.host_replies_total += 1
            if not result.ok:
                self.metrics.host_errors_total += 1

    async def sweep_timeouts(self, *, now: float) -> int:
        async with self._state_lock:
            expired = self.state.expired_correlation_ids(
                now=now, timeout_s=self.settings.TASK_TIMEOUT_SECONDS
            )
            cleared = 0
            for cid in expired:
                if self.state.clear_pending(cid, now=now):
                    self.metrics.host_timeouts_total += 1
                    cleared += 1
            return cleared
