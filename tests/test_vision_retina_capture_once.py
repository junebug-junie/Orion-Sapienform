from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "services" / "orion-vision-retina"))
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from app.main import RetinaService
from app.settings import Settings
from app.sources import FrameReadResult


class _FakeSource:
    def __init__(self, frame: np.ndarray | None) -> None:
        self._frame = frame

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def read(self) -> FrameReadResult | None:
        if self._frame is None:
            return None
        h, w = self._frame.shape[:2]
        return FrameReadResult(frame=self._frame, ts=1710000001.0, width=w, height=h)


@pytest.mark.asyncio
async def test_capture_once_publishes_pointer(tmp_path: Path) -> None:
    settings = Settings(
        FRAME_STORAGE_DIR=str(tmp_path),
        RETINA_CAMERA_ID="test-cam-01",
        RETINA_STREAM_ID="test-stream-01",
        CHANNEL_RETINA_PUB="orion:vision:frames",
    )
    bus = MagicMock()
    bus.publish = AsyncMock()
    svc = RetinaService(settings=settings, bus=bus)
    svc.source = _FakeSource(np.zeros((12, 16, 3), dtype=np.uint8))

    ok = await svc.capture_once()
    assert ok is True
    bus.publish.assert_awaited_once()
    channel, env = bus.publish.await_args.args
    assert channel == "orion:vision:frames"
    assert env.kind == "vision.frame.pointer"
    assert env.schema_id == "orion.envelope"
    assert env.payload["camera_id"] == "test-cam-01"
    assert env.payload["stream_id"] == "test-stream-01"
    assert env.payload["width"] == 16
    assert env.payload["height"] == 12
    assert Path(env.payload["image_path"]).is_file()


@pytest.mark.asyncio
async def test_capture_once_skips_publish_on_read_failure(tmp_path: Path) -> None:
    settings = Settings(FRAME_STORAGE_DIR=str(tmp_path))
    bus = MagicMock()
    bus.publish = AsyncMock()
    svc = RetinaService(settings=settings, bus=bus)
    svc.source = _FakeSource(None)

    ok = await svc.capture_once()
    assert ok is False
    bus.publish.assert_not_awaited()
    assert svc.metrics.last_error == "source read returned no frame"
    assert svc._source_ok() is False
    assert svc.metrics.fps_observed == 0.0


@pytest.mark.asyncio
async def test_capture_once_updates_fps_observed(tmp_path: Path) -> None:
    settings = Settings(FRAME_STORAGE_DIR=str(tmp_path))
    bus = MagicMock()
    bus.publish = AsyncMock()
    svc = RetinaService(settings=settings, bus=bus)
    svc.source = _FakeSource(np.zeros((4, 4, 3), dtype=np.uint8))
    svc._last_publish_ts = 0.0

    await svc.capture_once()
    assert svc.metrics.fps_observed > 0.0
