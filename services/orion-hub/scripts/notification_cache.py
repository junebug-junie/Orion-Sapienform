from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Set

from orion.core.bus.async_service import OrionBusAsync
from orion.schemas.notify import HubNotificationEvent

logger = logging.getLogger("orion-hub.notifications")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class NotificationCache:
    def __init__(self, *, max_items: int, channel: str) -> None:
        self.max_items = max_items
        self.channel = channel
        self._items: Deque[Dict[str, Any]] = deque(maxlen=max_items)
        self._bus: Optional[OrionBusAsync] = None
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._subscribers: Set[asyncio.Queue] = set()

    async def start(self, bus: OrionBusAsync) -> None:
        if self._task and not self._task.done():
            return
        self._bus = bus
        self._task = asyncio.create_task(self._run(), name="hub-notification-cache")

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    def register_queue(self, queue: asyncio.Queue) -> None:
        self._subscribers.add(queue)

    def unregister_queue(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    async def _run(self) -> None:
        if not self._bus:
            return
        logger.info("Subscribing to in-app notifications: %s", self.channel)
        try:
            async with self._bus.subscribe(self.channel) as pubsub:
                async for msg in self._bus.iter_messages(pubsub):
                    await self._handle_message(msg)
        except asyncio.CancelledError:
            logger.info("Notification cache task cancelled.")
        except Exception as exc:
            logger.error("Notification cache loop failed: %s", exc, exc_info=True)

    async def _handle_message(self, msg: Dict[str, Any]) -> None:
        if not self._bus:
            return
        decoded = self._bus.codec.decode(msg.get("data"))
        if not decoded.ok:
            return
        payload = decoded.envelope.payload
        try:
            event = HubNotificationEvent.model_validate(payload)
        except Exception as exc:
            logger.warning("Failed to parse HubNotificationEvent: %s", exc)
            return

        item = event.model_dump(mode="json")
        item.setdefault("received_at", _utcnow().isoformat())

        async with self._lock:
            self._items.appendleft(item)

        for queue in list(self._subscribers):
            try:
                queue.put_nowait({"kind": "notification", "notification": item})
            except asyncio.QueueFull:
                continue

    async def get_latest(self, limit: int) -> List[Dict[str, Any]]:
        async with self._lock:
            return list(self._items)[:limit]
