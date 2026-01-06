from __future__ import annotations

import time
from collections import deque
from typing import Deque, List, Optional
from uuid import uuid4

from loguru import logger

from orion.schemas.pad import PadEventV1, StateBuckets, StateFrameV1, StateSummary

from ..observability.stats import PadStatsTracker
from ..tensor.hash_projection import HashProjectionTensorizer
from ..tensor.interface import Tensorizer
from ..store.redis_store import PadStore


class FrameAggregator:
    def __init__(
        self,
        *,
        store: PadStore,
        tensor_dim: int,
        window_ms: int,
        max_events: int,
        stats: PadStatsTracker,
        tensorizer: Optional[Tensorizer] = None,
    ):
        self.store = store
        self.window_ms = window_ms
        self.max_events = max_events
        self.stats = stats
        self.tensorizer = tensorizer or HashProjectionTensorizer(dim=tensor_dim)
        self._events: Deque[PadEventV1] = deque()

    def add_event(self, event: PadEventV1) -> None:
        self._events.append(event)
        cutoff = int(time.time() * 1000) - self.window_ms
        while self._events and self._events[0].ts_ms < cutoff:
            self._events.popleft()

    async def build_frame(self) -> Optional[StateFrameV1]:
        start = time.time()
        now_ms = int(time.time() * 1000)
        window_start = now_ms - self.window_ms
        window_events = [e for e in list(self._events) if e.ts_ms >= window_start]
        if not window_events:
            return None

        sorted_events = sorted(window_events, key=lambda e: e.salience, reverse=True)
        capped = sorted_events[: self.max_events]
        salient_ids = [e.event_id for e in capped]

        summary = StateSummary(
            top_signals=[f"{e.type}:{e.subject or e.source_channel}" for e in capped[:5]],
            active_tasks=[],
            risk_flags=[e.type for e in capped if e.type in ("anomaly", "decision", "intent")],
        )
        buckets = StateBuckets()
        tensor = self.tensorizer.encode(capped)

        frame = StateFrameV1(
            frame_id=str(uuid4()),
            ts_ms=now_ms,
            window_ms=self.window_ms,
            summary=summary,
            state=buckets,
            salient_event_ids=salient_ids,
            tensor=tensor,
        )
        await self.store.store_frame(frame)
        self.stats.increment_frames_built()
        self.stats.record_frame_meta(ts_ms=now_ms, count=len(window_events))
        duration_ms = (time.time() - start) * 1000.0
        self.stats.record_frame_build_ms(duration_ms=duration_ms)
        return frame
