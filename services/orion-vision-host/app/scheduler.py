from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, Optional, Tuple

from loguru import logger

from .gpu import GpuInspector


@dataclass
class ScheduledPick:
    device: str
    gpu_index: Optional[int]


class VisionScheduler:
    """
    Multi-GPU scheduler with:
    - global inflight cap
    - per-GPU inflight cap
    - queueing when busy
    - VRAM-aware GPU selection via NVML
    """

    def __init__(
        self,
        *,
        devices: list[str],
        pick_metric: str,
        max_inflight: int,
        max_inflight_per_gpu: int,
        queue_when_busy: bool,
        max_queue: int,
        reserve_mb: int,
        soft_floor_mb: int,
        hard_floor_mb: int,
    ):
        self.devices = devices
        self.pick_metric = pick_metric

        self.max_inflight = max_inflight
        self.max_inflight_per_gpu = max_inflight_per_gpu
        self.queue_when_busy = queue_when_busy
        self.max_queue = max_queue

        self.reserve_mb = reserve_mb
        self.soft_floor_mb = soft_floor_mb
        self.hard_floor_mb = hard_floor_mb

        self._global_sem = asyncio.Semaphore(max_inflight)
        self._gpu_sems: Dict[int, asyncio.Semaphore] = {}

        self._queue: asyncio.Queue[Tuple[Callable[[], Awaitable], asyncio.Future]] = asyncio.Queue(maxsize=max_queue)
        self._queue_task: Optional[asyncio.Task] = None

        self.gpu = GpuInspector()

        # Parse "cuda:X" indices from devices list
        self._gpu_indices = []
        for d in devices:
            if d.startswith("cuda:"):
                try:
                    self._gpu_indices.append(int(d.split(":")[1]))
                except Exception:
                    continue

        for idx in self._gpu_indices:
            self._gpu_sems[idx] = asyncio.Semaphore(max_inflight_per_gpu)

    async def start(self) -> None:
        if self._queue_task is None:
            self._queue_task = asyncio.create_task(self._queue_worker(), name="visionhost-queue-worker")
            logger.info(
                f"[SCHED] started max_inflight={self.max_inflight} "
                f"per_gpu={self.max_inflight_per_gpu} queue={self.max_queue}"
            )

    async def stop(self) -> None:
        if self._queue_task:
            self._queue_task.cancel()
            try:
                await self._queue_task
            except Exception:
                pass
            self._queue_task = None
        logger.info("[SCHED] stopped")

    async def _queue_worker(self) -> None:
        while True:
            fn, fut = await self._queue.get()
            if fut.cancelled():
                continue
            try:
                result = await fn()
                if not fut.cancelled():
                    fut.set_result(result)
            except Exception as e:
                if not fut.cancelled():
                    fut.set_exception(e)

    def _pick_gpu_index(self) -> Optional[int]:
        if not self._gpu_indices:
            return None

        pick = self.gpu.pick_best_gpu(
            candidates=self._gpu_indices,
            reserve_mb=self.reserve_mb,
            hard_floor_mb=self.hard_floor_mb,
            metric=self.pick_metric,
        )
        if pick is None:
            return None
        idx, info = pick
        logger.debug(f"[SCHED] picked gpu={idx} free={info.free_mb}MB used={info.used_mb}MB")
        return idx

    def _device_str(self, gpu_index: Optional[int]) -> str:
        if gpu_index is None:
            return "cpu"
        return f"cuda:{gpu_index}"

    async def submit(
        self,
        handler: Callable[[ScheduledPick], Awaitable],
    ):
        """
        Submit a unit of work to the scheduler; returns handler result.
        """
        async def run_once():
            async with self._global_sem:
                gpu_index = self._pick_gpu_index()
                if gpu_index is None:
                    # No GPU available above hard floor; caller can degrade/refuse.
                    pick = ScheduledPick(device="cpu", gpu_index=None)
                    return await handler(pick)

                gpu_sem = self._gpu_sems.get(gpu_index)
                if gpu_sem is None:
                    pick = ScheduledPick(device=self._device_str(gpu_index), gpu_index=gpu_index)
                    return await handler(pick)

                async with gpu_sem:
                    pick = ScheduledPick(device=self._device_str(gpu_index), gpu_index=gpu_index)
                    return await handler(pick)

        # If queueing is desired, we try to put into queue when global sem is saturated.
        # We can’t directly introspect semaphore waiters safely, so we queue only if immediate acquire fails.
        try:
            acquired = self._global_sem.locked() and self.queue_when_busy
        except Exception:
            acquired = False

        if self.queue_when_busy and acquired:
            fut: asyncio.Future = asyncio.get_running_loop().create_future()
            try:
                self._queue.put_nowait((run_once, fut))
            except asyncio.QueueFull:
                raise RuntimeError("VisionHost queue is full")
            return await fut

        return await run_once()
