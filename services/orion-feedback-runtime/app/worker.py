from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.feedback.builder import build_feedback_frame
from orion.feedback.policy import load_feedback_policy
from orion.schemas.feedback_frame import FeedbackFrameV1

from app.settings import get_settings
from app.store import FeedbackRuntimeStore

logger = logging.getLogger("orion.feedback.runtime")


class FeedbackRuntimeWorker:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._store = FeedbackRuntimeStore(self._settings.postgres_uri)
        self._policy = load_feedback_policy(Path(self._settings.feedback_policy_path))
        self._stop = asyncio.Event()
        self._bus = OrionBusAsync(
            self._settings.bus_url,
            enabled=self._settings.bus_enabled,
        )

    def _service_ref(self) -> ServiceRef:
        return ServiceRef(
            name=self._settings.service_name,
            version=self._settings.service_version,
            node=self._settings.node_name,
        )

    async def start(self) -> None:
        await self._bus.connect()
        asyncio.create_task(self._poll_loop(), name="feedback-runtime-poll")

    async def stop(self) -> None:
        self._stop.set()
        await self._bus.close()

    async def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame = await asyncio.to_thread(self._tick)
                if frame is not None:
                    await self._publish_feedback_frame(frame)
            except Exception:
                logger.exception("feedback_runtime_tick_failed")
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=float(self._settings.feedback_poll_interval_sec),
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    async def _publish_feedback_frame(self, frame: FeedbackFrameV1) -> None:
        env = BaseEnvelope(
            kind="feedback.frame.v1",
            source=self._service_ref(),
            payload=frame.model_dump(mode="json"),
        )
        await self._bus.publish(self._settings.feedback_bus_channel, env)
        logger.info(
            "feedback_frame_published frame_id=%s channel=%s",
            frame.frame_id,
            self._settings.feedback_bus_channel,
        )

    def _tick(self) -> Optional[FeedbackFrameV1]:
        if not self._settings.enable_feedback_runtime:
            return None

        dispatch = self._store.load_latest_dispatch_frame_without_feedback()
        if dispatch is None:
            return None
        if self._store.load_feedback_frame_for_dispatch(dispatch.frame_id) is not None:
            return None

        policy_frame = self._store.load_policy_frame(dispatch.source_policy_frame_id)
        proposal_frame = self._store.load_proposal_frame(dispatch.source_proposal_frame_id)
        self_state_before = self._store.load_self_state(dispatch.source_self_state_id)
        self_state_after = self._store.load_latest_self_state_after(
            dispatch.generated_at,
            window_sec=self._policy.windows.field_after_window_sec,
        )
        cortex_results = self._store.load_cortex_result_evidence(dispatch)

        frame = build_feedback_frame(
            dispatch_frame=dispatch,
            policy_frame=policy_frame,
            proposal_frame=proposal_frame,
            self_state_before=self_state_before,
            self_state_after=self_state_after,
            cortex_results=cortex_results or None,
            policy=self._policy,
        )
        self._store.save_feedback_frame(frame)
        logger.info(
            "feedback_frame_saved frame_id=%s dispatch_frame_id=%s outcome_status=%s observations=%d",
            frame.frame_id,
            dispatch.frame_id,
            frame.outcome_status,
            len(frame.observations),
        )
        return frame
