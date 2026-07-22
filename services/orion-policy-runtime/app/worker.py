from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from orion.policy.builder import build_policy_decision_frame
from orion.policy.policy import load_substrate_policy

from app.settings import get_settings
from app.store import PolicyRuntimeStore

logger = logging.getLogger("orion.policy.runtime")


class PolicyRuntimeWorker:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._store = PolicyRuntimeStore(self._settings.postgres_uri)
        self._policy = load_substrate_policy(Path(self._settings.substrate_policy_path))
        self._stop = asyncio.Event()

    async def start(self) -> None:
        asyncio.create_task(self._poll_loop(), name="policy-runtime-poll")

    async def stop(self) -> None:
        self._stop.set()

    async def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.to_thread(self._tick)
            except Exception:
                logger.exception("policy_runtime_tick_failed")
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=float(self._settings.policy_poll_interval_sec),
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    def _tick(self) -> None:
        if not self._settings.enable_policy_runtime:
            return

        proposal = self._store.load_next_proposal_without_policy_frame()
        if proposal is None:
            return

        # 2026-07-22 (SelfStateV1 burn): build_policy_decision_frame now
        # evaluates directly off proposal_frame (which already carries
        # source_field_tick_id) -- no separate self-state load, and so no
        # "unevaluable" fallback for a load failure that can no longer
        # happen here. build_unevaluable_policy_decision_frame is kept in
        # orion/policy/builder.py for now (still exported), but this worker
        # no longer has a caller for it.
        frame = build_policy_decision_frame(
            proposal_frame=proposal,
            policy=self._policy,
        )
        self._store.save_policy_decision_frame(frame)
        logger.info(
            "policy_decision_frame_saved frame_id=%s proposal_frame_id=%s decisions=%d",
            frame.frame_id,
            proposal.frame_id,
            len(frame.decisions),
        )
