import asyncio
import uuid
import datetime
from typing import Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope
from orion.schemas.vision import (
    VisionEventPayload,
    VisionScribeAckPayload,
    VisionScribeRequestPayload,
    VisionScribeResultPayload
)

# Assuming shared schemas for writers exist as per instructions
try:
    from orion.schemas.sql.schemas import SqlWriteRequest
except ImportError:
    from pydantic import BaseModel, ConfigDict
    class SqlWriteRequest(BaseModel):
        model_config = ConfigDict(extra="allow")
        table: str
        data: dict
    logger.warning("Could not import SqlWriteRequest, using fallback")

try:
    from orion.schemas.rdf import RdfWriteRequest
except ImportError:
    from pydantic import BaseModel, ConfigDict
    class RdfWriteRequest(BaseModel):
        model_config = ConfigDict(extra="allow")
        graph: str
        triples: list
    logger.warning("Could not import RdfWriteRequest, using fallback")

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

        ack = await self._write_to_sinks(req.events)

        # 1. Reply
        res_payload = VisionScribeResultPayload(ack=ack)

        reply_env = BaseEnvelope(
            schema_id="vision.scribe.result",
            schema_version="1.0.0",
            kind="vision.scribe.result",
            source=f"{settings.SERVICE_NAME}:{settings.SERVICE_VERSION}",
            correlation_id=env.correlation_id,
            causality_chain=env.causality_chain + [env.correlation_id] if env.correlation_id else [],
            payload=res_payload,
            reply_to=None
        )

        if env.reply_to:
            await self.bus.publish(env.reply_to, reply_env)

        # 2. Broadcast Ack
        ack_env = BaseEnvelope(
            schema_id="vision.scribe.ack",
            schema_version="1.0.0",
            kind="vision.scribe.ack",
            source=f"{settings.SERVICE_NAME}:{settings.SERVICE_VERSION}",
            correlation_id=env.correlation_id or str(uuid.uuid4()),
            causality_chain=env.causality_chain + [env.correlation_id] if env.correlation_id else [],
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

        ack = await self._write_to_sinks(payload)

        # Ack Broadcast
        ack_env = BaseEnvelope(
            schema_id="vision.scribe.ack",
            schema_version="1.0.0",
            kind="vision.scribe.ack",
            source=f"{settings.SERVICE_NAME}:{settings.SERVICE_VERSION}",
            correlation_id=env.correlation_id or str(uuid.uuid4()),
            causality_chain=env.causality_chain + [env.correlation_id] if env.correlation_id else [],
            payload=ack
        )
        await self.bus.publish(settings.CHANNEL_SCRIBE_PUB, ack_env)
        logger.info(f"[SCRIBE] Processed and acked {len(payload.events)} events")

    async def _write_to_sinks(self, payload: VisionEventPayload) -> VisionScribeAckPayload:
        errors = []
        try:
            for evt in payload.events:
                # 1. SQL Write
                try:
                    sql_req = SqlWriteRequest(
                        table="vision_events",
                        data={
                            "event_id": evt.event_id,
                            "event_type": evt.event_type,
                            "narrative": evt.narrative,
                            "confidence": evt.confidence,
                            "created_at": datetime.datetime.utcnow().isoformat()
                        }
                    )
                    await self._send_write(settings.CHANNEL_SQL_WRITE, "sql.write.request", sql_req)
                except Exception as e:
                    logger.error(f"[SCRIBE] SQL Write failed for {evt.event_id}: {e}")
                    errors.append(f"SQL:{e}")

                # 2. RDF Write
                try:
                    triples = [
                        (f"orion:event:{evt.event_id}", "rdf:type", "orion:VisionEvent"),
                        (f"orion:event:{evt.event_id}", "orion:hasNarrative", evt.narrative),
                        (f"orion:event:{evt.event_id}", "orion:hasType", evt.event_type),
                    ]
                    for ent in evt.entities:
                        triples.append((f"orion:event:{evt.event_id}", "orion:mentionsEntity", ent))

                    rdf_req = RdfWriteRequest(graph="vision", triples=triples)
                    await self._send_write(settings.CHANNEL_RDF_ENQUEUE, "rdf.write.request", rdf_req)
                except Exception as e:
                    logger.error(f"[SCRIBE] RDF Write failed for {evt.event_id}: {e}")
                    errors.append(f"RDF:{e}")

            if errors:
                 return VisionScribeAckPayload(ok=False, error="; ".join(errors), message="Partial success")
            return VisionScribeAckPayload(ok=True, message=f"Processed {len(payload.events)} events")
        except Exception as e:
            logger.error(f"[SCRIBE] Write flow failed: {e}")
            return VisionScribeAckPayload(ok=False, error=str(e))

    async def _send_write(self, channel: str, kind: str, payload: Any):
        env = BaseEnvelope(
            schema_id=kind,
            schema_version="1.0.0",
            kind=kind,
            source=f"{settings.SERVICE_NAME}:{settings.SERVICE_VERSION}",
            correlation_id=str(uuid.uuid4()),
            payload=payload
        )
        await self.bus.publish(channel, env)

service = ScribeService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await service.start()
    yield
    await service.stop()

app = FastAPI(title="Orion Vision Scribe", version="0.1.0", lifespan=lifespan)
