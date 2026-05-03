"""Scheduler behavior without GPU inference."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from app.scheduler import VisionQueueFullError, VisionScheduler


def test_queue_full_raises_when_bounded_queue_rejects_put(monkeypatch):
    """Exercise VisionQueueFullError without relying on the queue worker + sem interaction."""

    fake_info = MagicMock(free_mb=12000, used_mb=4000, total_mb=16000)

    def fake_pick(*args, **kwargs):
        return 0, fake_info

    async def body():
        sched = VisionScheduler(
            devices=["cuda:0"],
            pick_metric="free_vram_mb",
            max_inflight=2,
            max_inflight_per_gpu=1,
            queue_when_busy=True,
            max_queue=10,
            reserve_mb=0,
            soft_floor_mb=0,
            hard_floor_mb=0,
        )
        monkeypatch.setattr(sched.gpu, "pick_best_gpu", fake_pick)
        monkeypatch.setattr(sched._global_sem, "locked", lambda: True)

        def raise_full(_item):
            raise asyncio.QueueFull()

        monkeypatch.setattr(sched._queue, "put_nowait", raise_full)

        await sched.start()

        async def handler(pick):
            return "unused"

        with pytest.raises(VisionQueueFullError):
            await sched.submit(handler)

        await sched.stop()

    asyncio.run(body())


def test_can_pick_gpu_false_without_cuda_devices():
    sched = VisionScheduler(
        devices=[],
        pick_metric="free_vram_mb",
        max_inflight=2,
        max_inflight_per_gpu=1,
        queue_when_busy=False,
        max_queue=10,
        reserve_mb=0,
        soft_floor_mb=0,
        hard_floor_mb=0,
    )
    assert sched.can_pick_gpu() is False
