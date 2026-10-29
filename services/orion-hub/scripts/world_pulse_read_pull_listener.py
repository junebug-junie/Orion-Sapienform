"""Bus RPC adapter for the reading pull (orion/schemas/world_pulse_read_pull.py).

The motor allocator (orion-execution-dispatch-runtime) asks; the reading pipeline's ``pull`` runs
every gate and answers. This class only validates the envelope and publishes the reply: same trusted
internal bus boundary and reply-subject rule as ``scripts/reading_listener.py``.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import Any

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.schemas.world_pulse_read_pull import (
    WORLD_PULSE_READ_PULL_REPLY_PREFIX,
    WORLD_PULSE_READ_PULL_REQUEST_CHANNEL,
    WORLD_PULSE_READ_PULL_REQUEST_KIND,
    WORLD_PULSE_READ_PULL_RESULT_KIND,
    WorldPulseReadPullRequestV1,
)

logger = logging.getLogger("orion-hub.world_pulse_read_pull")


class WorldPulseReadPullListener:
    def __init__(self, pipeline: Any, source_ref: Any) -> None:
        self.pipeline = pipeline
        self.source_ref = source_ref
        self.bus: Any = None
        self.task: asyncio.Task | None = None

    async def handle(self, envelope: BaseEnvelope) -> None:
        reply = str(envelope.reply_to or "")
        # Never answer on a subject outside the pull reply prefix.
        if envelope.kind != WORLD_PULSE_READ_PULL_REQUEST_KIND or not reply.startswith(WORLD_PULSE_READ_PULL_REPLY_PREFIX):
            return
        try:
            req = WorldPulseReadPullRequestV1.model_validate(envelope.payload)
        except ValueError:
            logger.warning("reading_pull_request_invalid correlation_id=%s", envelope.correlation_id, exc_info=True)
            return
        if req.op == "act" and req.arm is None:
            logger.warning("reading_pull_act_without_arm dispatch_id=%s", req.dispatch_id)
            return
        result = await self.pipeline.pull(req)
        out = BaseEnvelope(kind=WORLD_PULSE_READ_PULL_RESULT_KIND, source=self.source_ref,
                           correlation_id=envelope.correlation_id, payload=result.model_dump(mode="json"))
        await self.bus.publish(reply, out)

    async def _run(self) -> None:
        while True:
            try:
                async with self.bus.subscribe(WORLD_PULSE_READ_PULL_REQUEST_CHANNEL) as pubsub:
                    async for raw in self.bus.iter_messages(pubsub):
                        try:
                            decoded = self.bus.codec.decode(raw.get("data"))
                            if decoded.ok:
                                await self.handle(decoded.envelope)
                        except Exception:  # noqa: BLE001
                            logger.warning("reading_pull_message_failed", exc_info=True)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.warning("reading_pull_listener_reconnecting", exc_info=True)
                await asyncio.sleep(1)

    async def start(self, bus: Any) -> None:
        self.bus = bus
        self.task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self.task is not None:
            self.task.cancel()
            with suppress(asyncio.CancelledError):
                await self.task
            self.task = None


__all__ = ["WorldPulseReadPullListener"]
