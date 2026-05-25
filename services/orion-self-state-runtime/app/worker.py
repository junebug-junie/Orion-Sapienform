from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from orion.self_state.builder import build_self_state
from orion.self_state.policy import load_self_state_policy

from app.settings import get_settings
from app.store import SelfStateRuntimeStore

logger = logging.getLogger("orion.self_state.runtime")


class SelfStateRuntimeWorker:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._store = SelfStateRuntimeStore(self._settings.postgres_uri)
        self._policy = load_self_state_policy(Path(self._settings.self_state_policy_path))
        self._stop = asyncio.Event()

    async def start(self) -> None:
        asyncio.create_task(self._poll_loop(), name="self-state-runtime-poll")

    async def stop(self) -> None:
        self._stop.set()

    async def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.to_thread(self._tick)
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

    def _tick(self) -> None:
        if not self._settings.enable_self_state_runtime:
            return

        attention = self._store.load_latest_attention_frame()
        if attention is None:
            return

        if self._store.load_self_state_for_attention_frame(attention.frame_id) is not None:
            return

        field = self._store.load_field_for_tick(attention.source_field_tick_id)
        if field is None:
            logger.warning(
                "self_state_skip_missing_field tick_id=%s frame_id=%s",
                attention.source_field_tick_id,
                attention.frame_id,
            )
            return

        previous = self._store.load_latest_self_state()
        state = build_self_state(
            field=field,
            attention=attention,
            policy=self._policy,
            previous_self_state=previous,
        )
        self._store.save_self_state(state)
        logger.info(
            "self_state_saved self_state_id=%s frame_id=%s condition=%s intensity=%.3f",
            state.self_state_id,
            attention.frame_id,
            state.overall_condition,
            state.overall_intensity,
        )
