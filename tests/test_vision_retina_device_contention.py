"""Regression guard for the real bug found live on carbon (2026-08-22):
the continuous capture_loop (cv2.VideoCapture on /dev/video0) and the
on-demand /capture/clip endpoint's ffmpeg process both wanted the same
physical webcam device at once -- a device only one process can hold
exclusively -- so every real clip capture failed with "Device or resource
busy". No amount of mocking could have caught this without a real camera;
this test instead locks down the coordination contract (pause_device()
actually releases/reopens the source, and genuinely excludes a concurrent
capture_loop tick) so a future refactor can't silently reintroduce the race.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "services" / "orion-vision-retina"))
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.main import RetinaService
from app.settings import Settings


class _TrackingSource:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def start(self) -> None:
        self.calls.append("start")

    async def stop(self) -> None:
        self.calls.append("stop")

    async def read(self):
        self.calls.append("read")
        return None


@pytest.fixture
def svc():
    settings = Settings()
    bus = MagicMock()
    bus.publish = AsyncMock()
    service = RetinaService(settings=settings, bus=bus)
    service.source = _TrackingSource()
    return service


@pytest.mark.asyncio
async def test_pause_device_stops_then_restarts_the_source(svc):
    async with svc.pause_device():
        assert svc.source.calls == ["stop"]
    assert svc.source.calls == ["stop", "start"]


@pytest.mark.asyncio
async def test_pause_device_still_restarts_source_if_body_raises(svc):
    with pytest.raises(ValueError):
        async with svc.pause_device():
            raise ValueError("clip capture failed mid-way")
    assert svc.source.calls == ["stop", "start"], (
        "source must be reopened even when the clip capture itself fails -- "
        "otherwise a failed on-demand capture would permanently break the "
        "continuous presence stream too"
    )


@pytest.mark.asyncio
async def test_pause_device_excludes_a_concurrent_capture_loop_tick(svc):
    """The actual race this whole fix exists to prevent: capture_loop must
    not be able to call source.read() while pause_device() has the device
    released for an on-demand clip capture."""
    order: list[str] = []

    async def simulated_capture_loop_tick():
        async with svc._device_lock:
            order.append("loop_tick_acquired")
            await svc.source.read()
            order.append("loop_tick_released")

    async def simulated_clip_capture():
        async with svc.pause_device():
            order.append("clip_capture_holds_device")
            await asyncio.sleep(0.05)  # stand-in for the real ffmpeg capture
            order.append("clip_capture_releasing_device")

    # Start the clip capture first so it grabs the lock; the loop tick,
    # started immediately after, must queue behind it rather than
    # interleave a read() while the device is released.
    clip_task = asyncio.create_task(simulated_clip_capture())
    await asyncio.sleep(0.01)  # let clip_task acquire the lock first
    loop_task = asyncio.create_task(simulated_capture_loop_tick())

    await asyncio.gather(clip_task, loop_task)

    assert order.index("clip_capture_releasing_device") < order.index("loop_tick_acquired"), (
        f"capture_loop's read ran while the device was released for clip "
        f"capture -- the exact race this fix exists to prevent. order={order}"
    )
