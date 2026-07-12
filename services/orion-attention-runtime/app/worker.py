from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from orion.attention.field_attention.builder import build_attention_frame
from orion.attention.field_attention.policy import load_attention_policy

from app.health_monitor import HealthMonitor
from app.settings import get_settings
from app.store import AttentionRuntimeStore

logger = logging.getLogger("orion.attention.runtime")


class AttentionRuntimeWorker:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._store = AttentionRuntimeStore(self._settings.postgres_uri)
        self._policy = load_attention_policy(Path(self._settings.attention_policy_path))
        self._health_monitor = HealthMonitor(self._store, self._settings)
        self._stop = asyncio.Event()

    async def start(self) -> None:
        asyncio.create_task(self._poll_loop(), name="attention-runtime-poll")
        asyncio.create_task(self._prune_loop(), name="attention-runtime-prune")
        asyncio.create_task(self._health_loop(), name="attention-runtime-health")

    async def stop(self) -> None:
        self._stop.set()

    async def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.to_thread(self._tick)
            except Exception:
                logger.exception("attention_runtime_tick_failed")
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=float(self._settings.attention_poll_interval_sec),
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    def _prune_tick(self) -> None:
        retention = float(self._settings.attention_frame_retention_hours)
        if retention <= 0:
            return
        deleted = self._store.prune_attention_frames(retention_hours=retention)
        if deleted:
            logger.info(
                "attention_frames_pruned deleted=%d retention_hours=%.1f", deleted, retention
            )

    async def _prune_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.to_thread(self._prune_tick)
            except Exception:
                logger.exception("attention_frame_prune_failed")
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=float(self._settings.attention_frame_prune_interval_sec),
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    async def _health_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.to_thread(self._health_monitor.run_tick)
            except Exception:
                logger.exception("attention_runtime_health_check_failed")
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=float(self._settings.health_check_interval_sec),
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    def _tick(self) -> None:
        if not self._settings.enable_attention_runtime:
            return

        field = self._store.load_latest_field()
        if field is None:
            return

        if self._store.load_attention_frame_for_field_tick(field.tick_id) is not None:
            return

        previous = self._store.load_latest_attention_frame()
        frame = build_attention_frame(
            field=field,
            policy=self._policy,
            previous_frame=previous,
        )
        if not self._settings.enable_transport_attention_visibility:
            frame = frame.model_copy(
                update={
                    "dominant_targets": [
                        t
                        for t in frame.dominant_targets
                        if t.target_id != "capability:transport"
                    ],
                    "capability_targets": [
                        t
                        for t in frame.capability_targets
                        if t.target_id != "capability:transport"
                    ],
                    "node_targets": frame.node_targets,
                    "system_targets": frame.system_targets,
                    "suppressed_targets": frame.suppressed_targets
                    + [t for t in frame.dominant_targets if t.target_id == "capability:transport"],
                }
            )
        self._store.save_attention_frame(frame)
        logger.info(
            "attention_frame_saved frame_id=%s tick_id=%s salience=%.3f",
            frame.frame_id,
            field.tick_id,
            frame.overall_salience,
        )
