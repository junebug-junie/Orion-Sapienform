from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from loguru import logger

from orion.core.bus.async_service import OrionBusAsync
from orion.schemas.vision import VisionFramePointerPayload

from .envelopes import make_frame_pointer_envelope
from .frame_store import cleanup_old_frames, save_frame
from .health import RetinaMetrics, make_system_health_envelope
from .settings import Settings, get_settings
from .sources import create_frame_source


class RetinaService:
    def __init__(
        self,
        settings: Settings | None = None,
        bus: OrionBusAsync | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.bus = bus or OrionBusAsync(
            url=self.settings.ORION_BUS_URL,
            enforce_catalog=self.settings.ORION_BUS_ENFORCE_CATALOG,
        )
        self.source = create_frame_source(
            self.settings.RETINA_SOURCE_TYPE,
            self.settings.RETINA_SOURCE,
            width=self.settings.RETINA_WIDTH,
            height=self.settings.RETINA_HEIGHT,
            reconnect_seconds=self.settings.SOURCE_RECONNECT_SECONDS,
        )
        self.metrics = RetinaMetrics()
        self._capture_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None
        self._shutdown = asyncio.Event()
        self._last_cleanup = 0.0

    async def start(self) -> None:
        logger.remove()
        logger.add(lambda m: print(m, end=""), level=self.settings.LOG_LEVEL)

        await self.bus.connect()
        await self.source.start()
        self._shutdown.clear()
        self._capture_task = asyncio.create_task(self.capture_loop())
        self._health_task = asyncio.create_task(self._health_loop())
        logger.info(f"[RETINA] Started → {self.settings.CHANNEL_RETINA_PUB}")

    async def stop(self) -> None:
        self._shutdown.set()
        for task in (self._capture_task, self._health_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        await self.source.stop()
        await self.bus.close()

    async def capture_loop(self) -> None:
        interval = 1.0 / max(self.settings.RETINA_FPS, 0.01)
        while not self._shutdown.is_set():
            t0 = time.time()
            try:
                await self.capture_once()
            except Exception as exc:
                self.metrics.last_error = str(exc)
                logger.error(f"[RETINA] capture_once failed: {exc}")
            if time.time() - self._last_cleanup > 10:
                await asyncio.to_thread(
                    cleanup_old_frames,
                    self.settings.FRAME_STORAGE_DIR,
                    self.settings.FRAME_RETENTION_SECONDS,
                )
                self._last_cleanup = time.time()
            elapsed = time.time() - t0
            await asyncio.sleep(max(0.0, interval - elapsed))

    async def capture_once(self) -> bool:
        result = await self.source.read()
        if result is None:
            self.metrics.frames_failed += 1
            self.metrics.last_error = "source read returned no frame"
            return False
        saved = await asyncio.to_thread(
            save_frame,
            result.frame,
            directory=self.settings.FRAME_STORAGE_DIR,
            camera_id=self.settings.RETINA_CAMERA_ID,
            stream_id=self.settings.RETINA_STREAM_ID,
            ts=result.ts,
            quality=self.settings.JPEG_QUALITY,
        )
        payload = VisionFramePointerPayload(
            image_path=saved.image_path,
            camera_id=self.settings.RETINA_CAMERA_ID,
            stream_id=self.settings.RETINA_STREAM_ID,
            frame_ts=result.ts,
            width=saved.width,
            height=saved.height,
            format=saved.format,
        )
        env = make_frame_pointer_envelope(
            payload,
            service_name=self.settings.SERVICE_NAME,
            service_version=self.settings.SERVICE_VERSION,
        )
        await self.bus.publish(self.settings.CHANNEL_RETINA_PUB, env)
        self.metrics.frames_published += 1
        self.metrics.last_frame_ts = result.ts
        self.metrics.last_error = None
        logger.info(f"[RETINA] Published frame pointer: {saved.image_path}")
        return True

    def _source_ok(self) -> bool:
        if self.metrics.last_error is not None:
            return False
        if self.metrics.frames_published > 0:
            return True
        return self.metrics.frames_failed == 0

    async def _health_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                env = make_system_health_envelope(
                    service_name=self.settings.SERVICE_NAME,
                    service_version=self.settings.SERVICE_VERSION,
                    camera_id=self.settings.RETINA_CAMERA_ID,
                    stream_id=self.settings.RETINA_STREAM_ID,
                    source_type=self.settings.RETINA_SOURCE_TYPE,
                    source_ok=self._source_ok(),
                    metrics=self.metrics,
                    fps_target=self.settings.RETINA_FPS,
                    storage_dir=self.settings.FRAME_STORAGE_DIR,
                )
                await self.bus.publish(self.settings.CHANNEL_SYSTEM_HEALTH, env)
            except Exception as exc:
                logger.warning(f"[RETINA] health publish failed: {exc}")
            await asyncio.sleep(self.settings.HEALTH_INTERVAL_SECONDS)


service = RetinaService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await service.start()
    yield
    await service.stop()


app = FastAPI(title="Orion Vision Retina", version="0.2.0", lifespan=lifespan)
