import asyncio
from typing import Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.vision import (
    VisionEventBundleItem,
    VisionEventPayload,
    VisionScribeAckPayload,
    VisionScribeRequestPayload,
    VisionScribeResultPayload
)

from .settings import Settings

settings = Settings()


class ScribeService:
    def __init__(self):
        self.bus = OrionBusAsync(url=settings.ORION_BUS_URL)
        self._consumer_task: Optional[asyncio.Task] = None
        self._rpc_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

    async def start(self):
        logger.remove()
        logger.add(lambda m: print(m, end=""), level=settings.LOG_LEVEL)

        await self.bus.connect()
        self._consumer_task = asyncio.create_task(self._consume())
        self._rpc_task = asyncio.create_task(self._consume_rpc())
        logger.info(f"[SCRIBE] Started. Listening on {settings.CHANNEL_SCRIBE_INTAKE} and {settings.CHANNEL_SCRIBE_REQUEST}")

    async def stop(self):
        self._shutdown_event.set()
        for t in [self._consumer_task, self._rpc_task]:
             if t:
                try:
                    t.cancel()
                    await t
                except asyncio.CancelledError:
                    pass
        await self.bus.close()

    async def _consume(self):
        async with self.bus.subscribe(settings.CHANNEL_SCRIBE_INTAKE) as pubsub:
            async for msg in self.bus.iter_messages(pubsub):
                if self._shutdown_event.is_set():
                    break

                data = msg.get("data")
                decoded = self.bus.codec.decode(data)
                if not decoded.ok:
                    continue

                env = decoded.envelope
                asyncio.create_task(self._process_events(env, is_rpc=False))

    async def _consume_rpc(self):
        async with self.bus.subscribe(settings.CHANNEL_SCRIBE_REQUEST) as pubsub:
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
                req = VisionScribeRequestPayload(**env.payload)
            else:
                req = env.payload
        except Exception as e:
            logger.error(f"[SCRIBE] RPC invalid payload: {e}")
            return

        ack = await self._write_to_sinks(req.events, env)

        # 1. Reply
        res_payload = VisionScribeResultPayload(ack=ack)

        reply_env = env.derive_child(
            kind="vision.scribe.result",
            source=ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION),
            payload=res_payload,
            reply_to=None
        )

        if env.reply_to:
            await self.bus.publish(env.reply_to, reply_env)

        # 2. Broadcast Ack
        ack_env = env.derive_child(
            kind="vision.scribe.ack",
            source=ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION),
            payload=ack
        )
        await self.bus.publish(settings.CHANNEL_SCRIBE_PUB, ack_env)

    async def _process_events(self, env: BaseEnvelope, is_rpc: bool = False):
        try:
            if isinstance(env.payload, dict):
                payload = VisionEventPayload(**env.payload)
            else:
                payload = env.payload
        except Exception as e:
            logger.error(f"[SCRIBE] Invalid payload: {e}")
            return

        ack = await self._write_to_sinks(payload, env)

        # Ack Broadcast
        ack_env = env.derive_child(
            kind="vision.scribe.ack",
            source=ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION),
            payload=ack
        )
        await self.bus.publish(settings.CHANNEL_SCRIBE_PUB, ack_env)
        logger.info(
            "[SCRIBE] Processed {} events ack_ok={}",
            len(payload.events),
            ack.ok,
        )

    async def _write_to_sinks(self, payload: VisionEventPayload, source_env: BaseEnvelope) -> VisionScribeAckPayload:
        errors = []
        try:
            for evt in payload.events:
                sql_ok = False
                # SQL Write (sole sink -- the Fuseki RDF write was removed
                # 2026-07-23: live-verified pure redundancy. Postgres
                # `vision_events` already receives the same event at the
                # same timestamp with a richer schema (confidence/salience/
                # evidence_refs/tags as real structured columns, not flat
                # RDF literals), and nothing in the codebase ever read the
                # Fuseki `orion:vision` graph back.)
                try:
                    await self._send_write(settings.CHANNEL_SQL_WRITE, "vision.event.v1", evt, source_env)
                    sql_ok = True
                except Exception as e:
                    logger.error(
                        "[SCRIBE] SQL Write failed event_id={} channel={}: {}",
                        evt.event_id,
                        settings.CHANNEL_SQL_WRITE,
                        e,
                    )
                    errors.append(f"SQL:{e}")

                logger.info(
                    "[SCRIBE] vision_persist event_id={} sql={} ack_ok={}",
                    evt.event_id,
                    sql_ok,
                    sql_ok,
                )

            if errors:
                 return VisionScribeAckPayload(ok=False, error="; ".join(errors), message="Partial success")
            return VisionScribeAckPayload(ok=True, message=f"Processed {len(payload.events)} events")
        except Exception as e:
            logger.error(f"[SCRIBE] Write flow failed: {e}")
            return VisionScribeAckPayload(ok=False, error=str(e))

    async def _send_write(self, channel: str, kind: str, payload: Any, source_env: BaseEnvelope):
        env = source_env.derive_child(
            kind=kind,
            source=ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION),
            payload=payload,
        )
        await self.bus.publish(channel, env)

service = ScribeService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await service.start()
    yield
    await service.stop()

app = FastAPI(title="Orion Vision Scribe", version="0.1.0", lifespan=lifespan)
