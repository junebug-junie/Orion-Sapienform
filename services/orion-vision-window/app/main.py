import asyncio
import time
import uuid
from typing import List, Dict, Any, Optional
from collections import deque

from fastapi import FastAPI
from loguru import logger

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope
from orion.schemas.vision import VisionArtifactPayload, VisionWindowPayload

from .settings import Settings

settings = Settings()

class WindowService:
    def __init__(self):
        self.bus = OrionBusAsync(url=settings.ORION_BUS_URL)
        self._consumer_task: Optional[asyncio.Task] = None
        self._emitter_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        self._buffer: List[Dict[str, Any]] = [] # Buffer of artifacts
        self._buffer_lock = asyncio.Lock()

        # We start a window
        self._window_start = time.time()

    async def start(self):
        logger.remove()
        logger.add(lambda m: print(m, end=""), level=settings.LOG_LEVEL)

        await self.bus.connect()
        self._consumer_task = asyncio.create_task(self._consume())
        self._emitter_task = asyncio.create_task(self._emit_loop())
        logger.info(f"[WINDOW] Started. Listening on {settings.CHANNEL_WINDOW_INTAKE}")

    async def stop(self):
        self._shutdown_event.set()
        if self._consumer_task:
            try:
                self._consumer_task.cancel()
                await self._consumer_task
            except asyncio.CancelledError:
                pass
        if self._emitter_task:
             try:
                self._emitter_task.cancel()
                await self._emitter_task
             except asyncio.CancelledError:
                pass
        await self.bus.close()

    async def _consume(self):
        async with self.bus.subscribe(settings.CHANNEL_WINDOW_INTAKE) as pubsub:
            async for msg in self.bus.iter_messages(pubsub):
                if self._shutdown_event.is_set():
                    break

                data = msg.get("data")
                decoded = self.bus.codec.decode(data)
                if not decoded.ok:
                    continue

                env = decoded.envelope
                # We expect VisionArtifactPayload
                try:
                    if isinstance(env.payload, dict):
                        payload = VisionArtifactPayload(**env.payload)
                    else:
                        payload = env.payload
                except Exception as e:
                    logger.warning(f"[WINDOW] Invalid payload: {e}")
                    continue

                async with self._buffer_lock:
                    self._buffer.append({
                        "artifact": payload,
                        "ts": time.time(), # Capture arrival time or use timing from payload? Arrival is simpler for windowing
                        "env": env
                    })

    async def _emit_loop(self):
        while not self._shutdown_event.is_set():
            now = time.time()
            if now - self._window_start >= settings.WINDOW_SIZE_SEC:
                await self._flush_window()
                self._window_start = time.time()

            await asyncio.sleep(1.0) # Check every second

    async def _flush_window(self):
        async with self._buffer_lock:
            if not self._buffer:
                return # Empty window, maybe don't emit? Or emit empty heartbeat? Let's skip empty.

            items = list(self._buffer)
            self._buffer.clear()

        # Aggregation logic
        artifact_ids = [item["artifact"].artifact_id for item in items]

        # Summarize
        counts = {}
        for item in items:
            art = item["artifact"]
            if art.outputs.objects:
                for obj in art.outputs.objects:
                    counts[obj.label] = counts.get(obj.label, 0) + 1

        top_labels = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]

        summary = {
            "object_counts": counts,
            "top_labels": top_labels,
            "item_count": len(items),
            "captions": [item["artifact"].outputs.caption.text for item in items if item["artifact"].outputs.caption]
        }

        window_payload = VisionWindowPayload(
            window_id=str(uuid.uuid4()),
            start_ts=self._window_start,
            end_ts=time.time(),
            summary=summary,
            artifact_ids=artifact_ids
        )

        envelope = BaseEnvelope(
            schema_id="vision.window",
            schema_version="1.0.0",
            kind="vision.window",
            source=f"{settings.SERVICE_NAME}:{settings.SERVICE_VERSION}",
            correlation_id=str(uuid.uuid4()),
            # Causality? We are aggregating many. Maybe pick the last one or create new chain?
            # Let's start new chain but reference in payload if needed.
            # Or pass trace context if we had one.
            payload=window_payload
        )

        await self.bus.publish(settings.CHANNEL_WINDOW_PUB, envelope)
        logger.info(f"[WINDOW] Emitted window {window_payload.window_id} with {len(items)} items")


service = WindowService()
app = FastAPI(title="Orion Vision Window", version="0.1.0", lifespan=None)

@app.on_event("startup")
async def startup():
    await service.start()

@app.on_event("shutdown")
async def shutdown():
    await service.stop()
