from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from orion.proposals.builder import build_proposal_frame
from orion.proposals.policy import load_proposal_policy

from app.settings import get_settings
from app.store import ProposalRuntimeStore

logger = logging.getLogger("orion.proposal.runtime")


class ProposalRuntimeWorker:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._store = ProposalRuntimeStore(self._settings.postgres_uri)
        self._policy = load_proposal_policy(Path(self._settings.proposal_policy_path))
        self._stop = asyncio.Event()

    async def start(self) -> None:
        asyncio.create_task(self._poll_loop(), name="proposal-runtime-poll")

    async def stop(self) -> None:
        self._stop.set()

    async def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.to_thread(self._tick)
            except Exception:
                logger.exception("proposal_runtime_tick_failed")
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=float(self._settings.proposal_poll_interval_sec),
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    def _tick(self) -> None:
        if not self._settings.enable_proposal_runtime:
            return

        self_state = self._store.load_latest_self_state()
        if self_state is None:
            return

        if self._store.load_proposal_frame_for_self_state(self_state.self_state_id) is not None:
            return

        attention = self._store.load_attention_frame(self_state.source_attention_frame_id)
        field = self._store.load_field_for_tick(self_state.source_field_tick_id)
        if field is None:
            logger.warning(
                "proposal_skip_missing_field tick_id=%s self_state_id=%s",
                self_state.source_field_tick_id,
                self_state.self_state_id,
            )

        previous = self._store.load_latest_proposal_frame()
        frame = build_proposal_frame(
            self_state=self_state,
            attention=attention,
            field=field,
            policy=self._policy,
            previous_frame=previous,
        )
        self._store.save_proposal_frame(frame)
        logger.info(
            "proposal_frame_saved frame_id=%s self_state_id=%s candidates=%d",
            frame.frame_id,
            self_state.self_state_id,
            len(frame.candidates),
        )
