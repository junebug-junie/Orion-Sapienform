import asyncio
import logging
import os
import time
import uuid
import cv2
from datetime import datetime, timezone

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.vision import VisionFramePointerPayload

from .context import settings, bus, camera

logger = logging.getLogger("orion-vision-edge.capture")

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _save_frame(frame, path: str, quality: int = 90):
    cv2.imwrite(path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

def _cleanup_old_frames(directory: str, max_age: int):
    now = time.time()
    for f in os.listdir(directory):
        fp = os.path.join(directory, f)
        if os.path.isfile(fp):
            if now - os.path.getmtime(fp) > max_age:
                try:
                    os.remove(fp)
                except Exception:
                    pass

async def capture_loop():
    logger.info(f"[CAPTURE] Starting capture loop. Storage={settings.FRAME_STORAGE_DIR}")

    _ensure_dir(settings.FRAME_STORAGE_DIR)

    # Simple background cleanup task
    last_cleanup = 0

    while True:
        # 1. Throttle capture
        await asyncio.sleep(1.0 / settings.FPS)

        # Cleanup every 10s
        if time.time() - last_cleanup > 10:
            # Run in thread to avoid blocking loop
            asyncio.get_event_loop().run_in_executor(
                None, _cleanup_old_frames, settings.FRAME_STORAGE_DIR, getattr(settings, 'FRAME_RETENTION_SECONDS', 60)
            )
            last_cleanup = time.time()

        # 2. Get frame
        frame = camera.get_frame()
        if frame is None:
            continue

        # 3. Save to disk (Shared Storage)
        ts = time.time()
        filename = f"frame_{int(ts*1000)}_{uuid.uuid4().hex[:6]}.jpg"
        filepath = os.path.join(settings.FRAME_STORAGE_DIR, filename)

        try:
            await asyncio.get_event_loop().run_in_executor(
                None, _save_frame, frame, filepath, settings.JPEG_QUALITY
            )
        except Exception as e:
            logger.error(f"[CAPTURE] Failed to save frame: {e}")
            continue

        # 4. Publish Pointer
        payload = VisionFramePointerPayload(
            image_path=filepath,
            camera_id=settings.SOURCE, # Using Source URL/ID as camera_id
            stream_id=settings.STREAM_ID,
            frame_ts=ts,
            width=frame.shape[1],
            height=frame.shape[0],
            format="jpg"
        )

        # Construct envelope
        env = BaseEnvelope(
            kind=settings.CHANNEL_VISION_FRAMES, # or "vision.frame.pointer"
            source=ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION),
            payload=payload.model_dump(mode="json"),
        )

        if bus.enabled:
            try:
                await bus.publish(settings.CHANNEL_VISION_FRAMES, env)
            except Exception as e:
                logger.error(f"[CAPTURE] Failed to publish pointer: {e}")
