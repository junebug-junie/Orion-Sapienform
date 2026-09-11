"""Internal bus RPC adapter. Postgres acceptance precedes event and receipt."""
from __future__ import annotations

import asyncio
import logging
from contextlib import suppress

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.schemas.reading import ReadingToolRequestV1, ReadingToolResultV1
from orion.world_pulse_read.events import TOOL_CHANNEL, TOOL_RESULT_PREFIX
from orion.world_pulse_read.queue import enqueue_reading, reading_status

logger = logging.getLogger(__name__)


class ReadingListener:
    def __init__(self, pool_provider, source_ref):
        self.pool_provider = pool_provider
        self.source_ref = source_ref
        self.task = None
        self.bus = None

    async def handle(self, envelope):
        # Same trusted internal bus boundary as harness RPC; never accept a
        # model-supplied reply subject or expose SQL/credentials through MCP.
        reply = f"{TOOL_RESULT_PREFIX}{envelope.correlation_id}"
        if envelope.reply_to != reply or envelope.kind != "reading.tool.request.v1":
            return
        try:
            command = ReadingToolRequestV1.model_validate(envelope.payload)
            pool = self.pool_provider()
            if pool is None:
                raise RuntimeError("reading_queue_unavailable")
            async with pool.acquire() as conn:
                if command.operation == "recommend_reading":
                    result = await enqueue_reading(conn, command.request, bus=self.bus, source=self.source_ref)
                else:
                    result = await reading_status(conn, command.request_id)
            response = ReadingToolResultV1(ok=True, result=result)
        except ValueError as exc:
            response = ReadingToolResultV1(ok=False, error=str(exc))
        except Exception:
            logger.warning("reading_tool_failed correlation_id=%s", envelope.correlation_id, exc_info=True)
            # Do not leak DSNs, SQL or arbitrary exception text into model context.
            response = ReadingToolResultV1(ok=False, error="reading_queue_unavailable; acceptance unknown, retry the same request")
        await self.bus.publish(reply, BaseEnvelope(
            kind="reading.tool.result.v1", correlation_id=envelope.correlation_id,
            source=self.source_ref, payload=response.model_dump(mode="json"),
        ))

    async def start(self, bus):
        self.bus = bus
        self.task = asyncio.create_task(self._run(), name="hub-reading-tools")

    async def stop(self):
        if self.task:
            self.task.cancel()
            with suppress(asyncio.CancelledError):
                await self.task
            self.task = None

    async def _run(self):
        while True:
            try:
                async with self.bus.subscribe(TOOL_CHANNEL) as pubsub:
                    async for raw in self.bus.iter_messages(pubsub):
                        try:
                            decoded = self.bus.codec.decode(raw.get("data"))
                            if decoded.ok:
                                await self.handle(decoded.envelope)
                        except Exception:
                            logger.warning("reading_rpc_message_failed", exc_info=True)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("reading_listener_reconnecting", exc_info=True)
                await asyncio.sleep(1)
