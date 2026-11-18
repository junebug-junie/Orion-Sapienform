from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from loguru import logger

from .gpu import GpuInspector, torch_to_nvml_index


class VisionQueueFullError(RuntimeError):
    """Raised when the bounded async queue cannot accept another pending task."""

    pass


@dataclass
class ScheduledPick:
    device: str
    gpu_index: Optional[int]
    refusal_reason: Optional[str] = None


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

        # "cuda:N" is a torch index; the admission gate reads NVML, which
        # indexes every card the container can see. Map each torch index to
        # the NVML card it really is, so the gate checks the card the work
        # runs on. Identity only when CUDA_VISIBLE_DEVICES is not a plain
        # index list (UUID/MIG entries) -- the old behaviour. A numeric list
        # that does not cover cuda:N means that device does not exist: drop
        # it rather than gate it on some other card's memory.
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        cvd_numeric = cvd is None or all(
            e.strip().isdigit() for e in cvd.split(",") if e.strip()
        ) and bool(cvd.strip())
        self._nvml_by_torch: Dict[int, int] = {}
        for idx in list(self._gpu_indices):
            nvml_idx = torch_to_nvml_index(idx)
            if nvml_idx is None and not cvd_numeric:
                logger.warning(
                    f"[SCHED] cannot map torch cuda:{idx} to an NVML index "
                    f"(CUDA_VISIBLE_DEVICES not a plain index list); assuming identity"
                )
                nvml_idx = idx
            if nvml_idx is None or nvml_idx in self._nvml_by_torch.values():
                logger.error(
                    f"[SCHED] dropping cuda:{idx}: not a distinct visible device "
                    f"(CUDA_VISIBLE_DEVICES={cvd!r}, nvml={nvml_idx})"
                )
                self._gpu_indices.remove(idx)
                self._gpu_sems.pop(idx, None)
                continue
            self._nvml_by_torch[idx] = nvml_idx
        self._torch_by_nvml: Dict[int, int] = {n: t for t, n in self._nvml_by_torch.items()}
        logger.info(f"[SCHED] gpu map torch->nvml {self._nvml_by_torch}")

        # gpu_hard_floor refusal accounting (surfaced on /health). A full
        # refusal streak used to be silent: ~600 refusals in 25 min on
        # 2026-10-15 with nothing on /health saying so.
        self.hard_floor_refusals_total = 0
        self.hard_floor_refusal_streak = 0
        self.last_refusal_reason: Optional[str] = None
        self.last_refusal_at: Optional[float] = None
        self.last_admit_at: Optional[float] = None

    def refusal_stats(self) -> Dict[str, Any]:
        return {
            "gpu_hard_floor_refusals_total": self.hard_floor_refusals_total,
            "gpu_hard_floor_refusal_streak": self.hard_floor_refusal_streak,
            "last_refusal_reason": self.last_refusal_reason,
            "last_refusal_at": self.last_refusal_at,
            "last_admit_at": self.last_admit_at,
        }

    def _gpu_label(self, nvml_idx: Any) -> str:
        torch_idx = self._torch_by_nvml.get(nvml_idx, nvml_idx)
        if torch_idx == nvml_idx:
            return f"gpu={nvml_idx}"
        return f"gpu=cuda:{torch_idx} (nvml={nvml_idx})"

    def _describe_refusal(self) -> str:
        detail: List[Dict[str, Any]] = list(getattr(self.gpu, "last_pick_detail", None) or [])
        if not detail:
            return f"no candidate GPU visible (configured torch->nvml={self._nvml_by_torch})"
        parts = []
        for row in detail:
            if not row.get("visible", True):
                parts.append(f"{self._gpu_label(row.get('gpu'))} not visible to NVML")
                continue
            parts.append(
                f"{self._gpu_label(row['gpu'])} nvml_free={row['nvml_free_mb']}MB + own_reusable={row['own_reusable_mb']}MB "
                f"- reserve={row['reserve_mb']}MB = effective={row['effective_free_mb']}MB "
                f"< hard_floor={row['hard_floor_mb']}MB"
            )
        return "; ".join(parts)

    def _record_refusal(self) -> str:
        reason = self._describe_refusal()
        self.hard_floor_refusals_total += 1
        self.hard_floor_refusal_streak += 1
        self.last_refusal_reason = reason
        self.last_refusal_at = time.time()
        streak = self.hard_floor_refusal_streak
        if streak == 1 or streak % 100 == 0:
            logger.warning(f"[SCHED] gpu_hard_floor refusal streak={streak} total={self.hard_floor_refusals_total} {reason}")
        return reason

    def _record_admit(self) -> None:
        if self.hard_floor_refusal_streak:
            logger.info(f"[SCHED] gpu_hard_floor refusal streak ended after {self.hard_floor_refusal_streak} refusals")
        self.hard_floor_refusal_streak = 0
        self.last_admit_at = time.time()

    def queue_depth(self) -> int:
        """Pending items waiting on the worker (best-effort backlog indicator)."""
        return self._queue.qsize()

    def can_pick_gpu(self) -> bool:
        """True if VRAM policy allows scheduling onto at least one configured CUDA device."""
        if not self._gpu_indices:
            return False
        return self._pick_gpu_index() is not None

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
            except asyncio.CancelledError:
                pass
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
            candidates=[self._nvml_by_torch.get(i, i) for i in self._gpu_indices],
            reserve_mb=self.reserve_mb,
            hard_floor_mb=self.hard_floor_mb,
            metric=self.pick_metric,
        )
        if pick is None:
            return None
        nvml_idx, info = pick
        idx = self._torch_by_nvml.get(nvml_idx, nvml_idx)
        logger.debug(
            f"[SCHED] picked cuda:{idx} (nvml={nvml_idx} {info.name}) free={info.free_mb}MB used={info.used_mb}MB"
        )
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
                    reason = self._record_refusal() if self._gpu_indices else None
                    pick = ScheduledPick(device="cpu", gpu_index=None, refusal_reason=reason)
                    return await handler(pick)
                self._record_admit()

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
                raise VisionQueueFullError("VisionHost queue is full")
            return await fut

        return await run_once()
