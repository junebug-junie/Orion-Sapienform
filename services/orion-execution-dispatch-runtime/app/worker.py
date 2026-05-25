from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from orion.execution_dispatch.builder import build_execution_dispatch_frame
from orion.execution_dispatch.policy import load_execution_dispatch_policy

from app.settings import get_settings
from app.store import ExecutionDispatchRuntimeStore

logger = logging.getLogger("orion.execution_dispatch.runtime")


class ExecutionDispatchRuntimeWorker:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._store = ExecutionDispatchRuntimeStore(self._settings.postgres_uri)
        self._policy = load_execution_dispatch_policy(
            Path(self._settings.execution_dispatch_policy_path)
        )
        self._stop = asyncio.Event()

    async def start(self) -> None:
        asyncio.create_task(self._poll_loop(), name="execution-dispatch-runtime-poll")

    async def stop(self) -> None:
        self._stop.set()

    async def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.to_thread(self._tick)
            except Exception:
                logger.exception("execution_dispatch_runtime_tick_failed")
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=float(self._settings.execution_dispatch_poll_interval_sec),
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    def _tick(self) -> None:
        if not self._settings.enable_execution_dispatch_runtime:
            return

        policy_frame = self._store.load_latest_policy_frame()
        if policy_frame is None:
            return

        if self._store.load_dispatch_frame_for_policy_frame(policy_frame.frame_id) is not None:
            return

        proposal = self._store.load_proposal_frame(policy_frame.source_proposal_frame_id)
        if proposal is None:
            logger.warning(
                "execution_dispatch_skip_missing_proposal proposal_frame_id=%s",
                policy_frame.source_proposal_frame_id,
            )
            return

        self_state = self._store.load_self_state(policy_frame.source_self_state_id)
        if self_state is None:
            logger.warning(
                "execution_dispatch_skip_missing_self_state self_state_id=%s",
                policy_frame.source_self_state_id,
            )
            return

        frame = build_execution_dispatch_frame(
            policy_frame=policy_frame,
            proposal_frame=proposal,
            self_state=self_state,
            policy=self._policy,
            override_dispatch_mode=self._settings.execution_dispatch_mode,
        )
        self._store.save_dispatch_frame(frame)
        logger.info(
            "execution_dispatch_frame_saved frame_id=%s policy_frame_id=%s candidates=%d blocked=%d",
            frame.frame_id,
            policy_frame.frame_id,
            len(frame.candidates),
            frame.blocked_count,
        )
