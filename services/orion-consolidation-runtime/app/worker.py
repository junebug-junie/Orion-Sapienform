from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from orion.consolidation.builder import build_consolidation_frame
from orion.consolidation.policy import load_consolidation_policy
from orion.consolidation.windows import compute_consolidation_window, stable_consolidation_frame_id

from app.settings import get_settings
from app.store import ConsolidationRuntimeStore

logger = logging.getLogger("orion.consolidation.runtime")


class ConsolidationRuntimeWorker:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._store = ConsolidationRuntimeStore(self._settings.postgres_uri)
        self._policy = load_consolidation_policy(
            Path(self._settings.consolidation_policy_path)
        )
        self._stop = asyncio.Event()

    async def start(self) -> None:
        asyncio.create_task(self._poll_loop(), name="consolidation-runtime-poll")

    async def stop(self) -> None:
        self._stop.set()

    async def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.to_thread(self._tick)
            except Exception:
                logger.exception("consolidation_runtime_tick_failed")
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=float(self._settings.consolidation_poll_interval_sec),
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    def _tick(self) -> None:
        if not self._settings.enable_consolidation_runtime:
            return

        window_start, window_end = compute_consolidation_window(
            lookback_minutes=self._policy.window.lookback_minutes,
        )
        frame_id = stable_consolidation_frame_id(
            window_start=window_start,
            window_end=window_end,
            policy_id=self._policy.policy_id,
        )
        if self._store.load_consolidation_frame_for_window(frame_id) is not None:
            return

        data = self._store.load_window_data(
            window_start,
            window_end,
            self._policy.window.max_frames_per_source,
        )
        frame = build_consolidation_frame(window=data, policy=self._policy)
        self._store.save_consolidation_frame(frame)
        self._store.upsert_expectations(frame.expectations)
        self._store.save_tensor_slices(
            frame.tensor_slices,
            window_start=frame.window_start,
            window_end=frame.window_end,
        )
        self._store.upsert_schema_candidates(frame.schema_candidates)
        logger.info(
            "consolidation_frame_saved frame_id=%s window_start=%s window_end=%s motifs=%d expectations=%d tensor_slices=%d schema_candidates=%d",
            frame.frame_id,
            frame.window_start.isoformat(),
            frame.window_end.isoformat(),
            len(frame.motif_observations),
            len(frame.expectations),
            len(frame.tensor_slices),
            len(frame.schema_candidates),
        )
