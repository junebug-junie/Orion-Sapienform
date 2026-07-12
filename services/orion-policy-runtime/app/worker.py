from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from orion.policy.builder import build_policy_decision_frame, build_unevaluable_policy_decision_frame
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

        self_state = self._store.load_self_state(proposal.source_self_state_id)
        if self_state is None:
            # A naive `return` here would retry this exact same proposal
            # forever -- it's always "the oldest unresolved" until a policy
            # frame exists for it, permanently blocking every proposal
            # queued behind it. Record an honest "could not evaluate" frame
            # instead so the FIFO queue advances (2026-07-12 live incident:
            # a schema change made an old self-state row permanently
            # unloadable, stalling the whole queue for it).
            logger.warning(
                "policy_self_state_unavailable self_state_id=%s proposal_frame_id=%s",
                proposal.source_self_state_id,
                proposal.frame_id,
            )
            frame = build_unevaluable_policy_decision_frame(
                proposal_frame=proposal,
                policy_id=self._policy.policy_id,
                reason=f"self_state {proposal.source_self_state_id} unavailable or schema-incompatible",
            )
            self._store.save_policy_decision_frame(frame)
            logger.info(
                "policy_decision_frame_saved_unevaluable frame_id=%s proposal_frame_id=%s",
                frame.frame_id,
                proposal.frame_id,
            )
            return

        frame = build_policy_decision_frame(
            proposal_frame=proposal,
            self_state=self_state,
            policy=self._policy,
        )
        self._store.save_policy_decision_frame(frame)
        logger.info(
            "policy_decision_frame_saved frame_id=%s proposal_frame_id=%s decisions=%d",
            frame.frame_id,
            proposal.frame_id,
            len(frame.decisions),
        )
