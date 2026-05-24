from __future__ import annotations

import asyncio
import os
import random
import time
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class FrameReadResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    frame: Any
    ts: float
    width: int
    height: int
    source_meta: dict[str, Any] = Field(default_factory=dict)


class FrameSource(Protocol):
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def read(self) -> FrameReadResult | None: ...


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


class FolderFrameSource:
    def __init__(self, directory: str) -> None:
        if not directory:
            raise ValueError("folder source requires RETINA_SOURCE directory")
        self.directory = directory
        self._files: list[str] = []
        self._index = 0

    def _refresh_files(self) -> None:
        if not os.path.isdir(self.directory):
            self._files = []
            return
        self._files = sorted(
            os.path.join(self.directory, name)
            for name in os.listdir(self.directory)
            if Path(name).suffix.lower() in _IMAGE_EXTS
        )

    async def start(self) -> None:
        self._refresh_files()
        self._index = 0

    async def stop(self) -> None:
        return None

    async def read(self) -> FrameReadResult | None:
        if not self._files:
            self._refresh_files()
        if not self._files:
            return None
        path = self._files[self._index % len(self._files)]
        self._index += 1
        frame = await asyncio.to_thread(cv2.imread, path)
        if frame is None:
            return None
        h, w = frame.shape[:2]
        return FrameReadResult(
            frame=frame,
            ts=time.time(),
            width=w,
            height=h,
            source_meta={"path": path, "type": "folder"},
        )


class MockFrameSource(FolderFrameSource):
    """Test/dev: random sample from folder like legacy mock."""

    async def read(self) -> FrameReadResult | None:
        if not self._files:
            return None
        path = random.choice(self._files)
        frame = await asyncio.to_thread(cv2.imread, path)
        if frame is None:
            return None
        h, w = frame.shape[:2]
        return FrameReadResult(
            frame=frame,
            ts=time.time(),
            width=w,
            height=h,
            source_meta={"path": path, "type": "mock"},
        )


class _VideoCaptureSource:
    def __init__(
        self,
        source: str | int,
        *,
        width: int | None,
        height: int | None,
        reconnect_seconds: float,
        source_type: str,
    ) -> None:
        self._source = source
        self._width = width
        self._height = height
        self._reconnect_seconds = reconnect_seconds
        self._source_type = source_type
        self._cap: cv2.VideoCapture | None = None
        self._failures = 0

    async def start(self) -> None:
        await self._open()

    async def stop(self) -> None:
        if self._cap is not None:
            await asyncio.to_thread(self._cap.release)
            self._cap = None

    async def _open(self) -> None:
        await self.stop()
        cap = await asyncio.to_thread(cv2.VideoCapture, self._source)
        if self._width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._width))
        if self._height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._height))
        self._cap = cap

    async def read(self) -> FrameReadResult | None:
        if self._cap is None or not self._cap.isOpened():
            await self._open()
        if self._cap is None or not self._cap.isOpened():
            self._failures += 1
            return None
        ok, frame = await asyncio.to_thread(self._cap.read)
        if not ok or frame is None:
            self._failures += 1
            if self._failures >= 3:
                await asyncio.sleep(self._reconnect_seconds)
                await self._open()
                self._failures = 0
            return None
        self._failures = 0
        h, w = frame.shape[:2]
        return FrameReadResult(
            frame=frame,
            ts=time.time(),
            width=w,
            height=h,
            source_meta={"type": self._source_type},
        )


class RtspFrameSource(_VideoCaptureSource):
    def __init__(
        self,
        url: str,
        *,
        width: int | None = None,
        height: int | None = None,
        reconnect_seconds: float = 5.0,
    ) -> None:
        if not url:
            raise ValueError("rtsp source requires RETINA_SOURCE url")
        super().__init__(
            url,
            width=width,
            height=height,
            reconnect_seconds=reconnect_seconds,
            source_type="rtsp",
        )


class WebcamFrameSource(_VideoCaptureSource):
    def __init__(
        self,
        device: str,
        *,
        width: int | None = None,
        height: int | None = None,
        reconnect_seconds: float = 5.0,
    ) -> None:
        index: str | int = int(device) if device.isdigit() else device
        super().__init__(
            index,
            width=width,
            height=height,
            reconnect_seconds=reconnect_seconds,
            source_type="webcam",
        )


def create_frame_source(
    source_type: str,
    source: str,
    *,
    width: int | None = None,
    height: int | None = None,
    reconnect_seconds: float = 5.0,
) -> FrameSource:
    st = source_type.lower().strip()
    common = dict(width=width, height=height, reconnect_seconds=reconnect_seconds)
    if st == "folder":
        return FolderFrameSource(source)
    if st == "mock":
        return MockFrameSource(source)
    if st == "rtsp":
        return RtspFrameSource(source, **common)
    if st == "webcam":
        return WebcamFrameSource(source or "0", **common)
    raise ValueError(f"unsupported RETINA_SOURCE_TYPE: {source_type!r}")
