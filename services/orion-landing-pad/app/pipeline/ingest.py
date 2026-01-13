from __future__ import annotations

import asyncio
import fnmatch
import time
from dataclasses import dataclass
from typing import List, Optional

from loguru import logger

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope

from ..observability.stats import PadStatsTracker


def _channel_matches(channel: str, patterns: List[str]) -> bool:
    for pat in patterns:
        if fnmatch.fnmatch(channel, pat):
            return True
    return False


@dataclass
class QueueItem:
    envelope: BaseEnvelope
    channel: str
    priority: str
    received_ts: float


class BoundedPadQueue:
    def __init__(self, *, maxsize: int, drop_policy: str):
        self.maxsize = maxsize
        self.drop_policy = drop_policy
        self._items: list[QueueItem] = []
        self._cond = asyncio.Condition()

    async def put(self, item: QueueItem) -> bool:
        async with self._cond:
            if len(self._items) >= self.maxsize:
                if item.priority == "high":
                    removed = self._drop_one(priority="low") or self._drop_one(priority="normal")
                    if not removed:
                        return False
                else:
                    removed = self._drop_one(priority="low")
                    if not removed:
                        return False

            self._items.append(item)
            self._cond.notify()
            return True

    async def get(self) -> QueueItem:
        async with self._cond:
            while not self._items:
                await self._cond.wait()
            return self._items.pop(0)

    def _drop_one(self, *, priority: str) -> bool:
        for idx, existing in enumerate(self._items):
            if existing.priority == priority:
                self._items.pop(idx)
                return True
        return False

    def depth(self) -> int:
        return len(self._items)


class IngestLoop:
    def __init__(
        self,
        *,
        bus: OrionBusAsync,
        queue: BoundedPadQueue,
        allowlist: List[str],
        denylist: List[str],
        app_name: str,
        stats: PadStatsTracker,
    ):
        self.bus = bus
        self.queue = queue
        self.allowlist = allowlist
        self.denylist = denylist
        self.app_name = app_name
        self.stats = stats
        self._stop = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="pad-ingest")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except Exception:
                pass

    async def _run(self) -> None:
        async with self.bus.subscribe(*self.allowlist, patterns=True) as pubsub:
            async for msg in self.bus.iter_messages(pubsub):
                if self._stop.is_set():
                    break
                channel_raw = msg.get("channel") or ""
                channel = channel_raw.decode("utf-8") if isinstance(channel_raw, (bytes, bytearray)) else str(channel_raw)
                if _channel_matches(channel, self.denylist):
                    self.stats.increment_dropped(reason="denylist")
                    continue
                data = msg.get("data")
                if data is None:
                    continue
                decoded = self.bus.codec.decode(data)
                if not decoded.ok or decoded.envelope is None:
                    self.stats.increment_dropped(reason="decode_failed")
                    continue

                env = decoded.envelope
                if env.source and env.source.name == self.app_name:
                    self.stats.increment_dropped(reason="self_loop")
                    continue
                if env.kind.startswith("orion.pad."):
                    self.stats.increment_dropped(reason="loop_guard_kind")
                    continue

                priority = self._priority_from_env(env)
                item = QueueItem(envelope=env, channel=channel, priority=priority, received_ts=time.time())
                accepted = await self.queue.put(item)
                if accepted:
                    self.stats.increment_ingested()
                else:
                    self.stats.increment_dropped(reason="queue_full")
                self.stats.set_queue_depth(self.queue.depth())

    @staticmethod
    def _priority_from_env(env: BaseEnvelope) -> str:
        k = env.kind.lower()
        if any(word in k for word in ("anomaly", "decision", "intent")):
            return "high"
        if "metric" in k:
            return "normal"
        return "low"
