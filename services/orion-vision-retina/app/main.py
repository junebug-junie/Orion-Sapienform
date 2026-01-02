import asyncio
import os
import uuid
import time
from typing import Optional
from pathlib import Path

from fastapi import FastAPI
from loguru import logger

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope
from orion.schemas.vision import VisionFramePointerPayload

from .settings import Settings

settings = Settings()

class RetinaService:
    def __init__(self):
        self.bus = OrionBusAsync(url=settings.ORION_BUS_URL)
        self._task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

    async def start(self):
        logger.remove()
        logger.add(lambda m: print(m, end=""), level=settings.LOG_LEVEL)

        await self.bus.connect()
        self._task = asyncio.create_task(self._loop())
        logger.info(f"[RETINA] Started. Publishing to {settings.CHANNEL_RETINA_PUB}")

    async def stop(self):
        self._shutdown_event.set()
        if self._task:
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self.bus.close()

    async def _loop(self):
        interval = 1.0 / settings.RETINA_FPS
        while not self._shutdown_event.is_set():
            start_ts = time.time()

            try:
                await self._capture_and_publish()
            except Exception as e:
                logger.error(f"[RETINA] Capture failed: {e}")

            elapsed = time.time() - start_ts
            sleep_time = max(0.0, interval - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    async def _capture_and_publish(self):
        # MOCK IMPLEMENTATION
        # In real world, this would grab from cv2.VideoCapture or watch a folder

        if settings.RETINA_SOURCE_TYPE == "mock":
             # We assume some test images exist or we just send pointers that might fail if host checks
             # For robustness, we check if we can find any image in the source path to point to.

             image_path = None
             if os.path.exists(settings.RETINA_SOURCE_PATH):
                 files = [f for f in os.listdir(settings.RETINA_SOURCE_PATH) if f.lower().endswith(('.jpg', '.png'))]
                 if files:
                     # Pick one randomly or rotate? Let's pick random for mock
                     import random
                     image_path = str(Path(settings.RETINA_SOURCE_PATH) / random.choice(files))

             if not image_path:
                 # logger.warning("[RETINA] No images found in source path, skipping frame.")
                 return

             payload = VisionFramePointerPayload(
                 image_path=image_path,
                 frame_ts=time.time(),
                 camera_id="mock-cam-01"
             )

             envelope = BaseEnvelope(
                 schema_id="vision.frame.pointer",
                 schema_version="1.0.0",
                 kind="vision.frame.pointer",
                 source=f"{settings.SERVICE_NAME}:{settings.SERVICE_VERSION}",
                 correlation_id=str(uuid.uuid4()),
                 payload=payload
             )

             await self.bus.publish(settings.CHANNEL_RETINA_PUB, envelope)
             logger.info(f"[RETINA] Published frame pointer: {image_path}")

service = RetinaService()
app = FastAPI(title="Orion Vision Retina", version="0.1.0", lifespan=None)

@app.on_event("startup")
async def startup():
    await service.start()

@app.on_event("shutdown")
async def shutdown():
    await service.stop()
