"""Relay context-exec agent_step bus events to per-correlation WebSocket queues."""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Dict, Optional, Set

from orion.core.bus.async_service import OrionBusAsync

logger = logging.getLogger("orion-hub.agent_step_relay")

AGENT_STEP_KIND = "context.exec.agent_step.v1"


class AgentStepRelay:
    def __init__(self, *, channel: str) -> None:
        self.channel = channel
        self._bus: Optional[OrionBusAsync] = None
        self._task: Optional[asyncio.Task] = None
        self._queues: Dict[str, Set[asyncio.Queue]] = defaultdict(set)

    async def start(self, bus: OrionBusAsync) -> None:
        if self._task and not self._task.done():
            return
        self._bus = bus
        self._task = asyncio.create_task(self._run(), name="hub-agent-step-relay")

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    def register_queue(self, correlation_id: str, queue: asyncio.Queue) -> None:
        self._queues[str(correlation_id)].add(queue)

    def unregister_queue(self, correlation_id: str, queue: asyncio.Queue) -> None:
        cid = str(correlation_id)
        self._queues.get(cid, set()).discard(queue)
        if cid in self._queues and not self._queues[cid]:
            self._queues.pop(cid, None)

    async def _run(self) -> None:
        if not self._bus:
            return
        logger.info("Subscribing to context-exec agent steps: %s", self.channel)
        try:
            async with self._bus.subscribe(self.channel) as pubsub:
                async for msg in self._bus.iter_messages(pubsub):
                    decoded = self._bus.codec.decode(msg.get("data"))
                    if not decoded.ok:
                        continue
                    env = decoded.envelope
                    await self._dispatch_payload(kind=str(env.kind), payload=env.payload or {})
        except asyncio.CancelledError:
            logger.info("Agent step relay cancelled.")
        except Exception as exc:
            logger.error("Agent step relay loop failed: %s", exc, exc_info=True)

    async def _dispatch_payload(self, *, kind: str, payload: dict[str, Any]) -> None:
        if kind != AGENT_STEP_KIND:
            return
        cid = str(payload.get("correlation_id") or "")
        queues = self._queues.get(cid)
        if not queues:
            return
        item = {"kind": "agent_step", "correlation_id": cid, "step": payload}
        for q in list(queues):
            try:
                q.put_nowait(item)
            except asyncio.QueueFull:
                continue
