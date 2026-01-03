import asyncio
import time
import uuid
from typing import List, Dict, Any, Optional
from collections import deque
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope
from orion.schemas.vision import (
    VisionArtifactPayload,
    VisionWindowPayload,
    VisionWindowRequestPayload,
    VisionWindowResultPayload
)

from .settings import Settings

settings = Settings()

class WindowService:
    def __init__(self):
        self.bus = OrionBusAsync(url=settings.ORION_BUS_URL)
        self._consumer_task: Optional[asyncio.Task] = None
        self._rpc_task: Optional[asyncio.Task] = None
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
        self._rpc_task = asyncio.create_task(self._consume_rpc())
        self._emitter_task = asyncio.create_task(self._emit_loop())
        logger.info(f"[WINDOW] Started. Listening on {settings.CHANNEL_WINDOW_INTAKE} and {settings.CHANNEL_WINDOW_REQUEST}")

    async def stop(self):
        self._shutdown_event.set()
        for t in [self._consumer_task, self._rpc_task, self._emitter_task]:
            if t:
                try:
                    t.cancel()
                    await t
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
                        "ts": time.time(),
                        "env": env
                    })

    async def _consume_rpc(self):
        """
        Consumes one-shot RPC requests from Cortex Exec.
        Input: VisionWindowRequestPayload (wrapping an artifact)
        Output: VisionWindowResultPayload (wrapping a window of size 1)
        """
        async with self.bus.subscribe(settings.CHANNEL_WINDOW_REQUEST) as pubsub:
             async for msg in self.bus.iter_messages(pubsub):
                if self._shutdown_event.is_set():
                    break

                data = msg.get("data")
                decoded = self.bus.codec.decode(data)
                if not decoded.ok:
                    continue

                env = decoded.envelope
                asyncio.create_task(self._handle_rpc(env))

    async def _handle_rpc(self, env: BaseEnvelope):
        try:
            if isinstance(env.payload, dict):
                req = VisionWindowRequestPayload(**env.payload)
            else:
                req = env.payload
        except Exception as e:
            logger.error(f"[WINDOW] RPC invalid payload: {e}")
            return

        # Logic: Turn single artifact into a window
        art = req.artifact

        summary = {
            "object_counts": {},
            "top_labels": [],
            "item_count": 1,
            "captions": []
        }
        if art.outputs.objects:
             counts = {}
             for obj in art.outputs.objects:
                 counts[obj.label] = counts.get(obj.label, 0) + 1
             summary["object_counts"] = counts
             summary["top_labels"] = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]

        if art.outputs.caption:
            summary["captions"] = [art.outputs.caption.text]

        window_payload = VisionWindowPayload(
            window_id=str(uuid.uuid4()),
            start_ts=time.time(),
            end_ts=time.time(),
            summary=summary,
            artifact_ids=[art.artifact_id]
        )

        # 1. Reply to caller
        res_payload = VisionWindowResultPayload(window=window_payload)

        reply_env = BaseEnvelope(
            schema_id="vision.window.result",
            schema_version="1.0.0",
            kind="vision.window.result",
            source=f"{settings.SERVICE_NAME}:{settings.SERVICE_VERSION}",
            correlation_id=env.correlation_id,
            causality_chain=env.causality_chain + [env.correlation_id] if env.correlation_id else [],
            payload=res_payload,
            reply_to=None
        )

        if env.reply_to:
            await self.bus.publish(env.reply_to, reply_env)

        # 2. Broadcast for consistency (streaming coherence)
        broadcast_env = BaseEnvelope(
            schema_id="vision.window",
            schema_version="1.0.0",
            kind="vision.window",
            source=f"{settings.SERVICE_NAME}:{settings.SERVICE_VERSION}",
            correlation_id=env.correlation_id,
            causality_chain=env.causality_chain + [env.correlation_id] if env.correlation_id else [],
            payload=window_payload
        )
        await self.bus.publish(settings.CHANNEL_WINDOW_PUB, broadcast_env)


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
                return

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
            payload=window_payload
        )

        await self.bus.publish(settings.CHANNEL_WINDOW_PUB, envelope)
        logger.info(f"[WINDOW] Emitted window {window_payload.window_id} with {len(items)} items")


service = WindowService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await service.start()
    yield
    await service.stop()

app = FastAPI(title="Orion Vision Window", version="0.1.0", lifespan=lifespan)
