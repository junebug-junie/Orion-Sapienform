from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from uuid import uuid4

from loguru import logger
from redis.asyncio import Redis

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.pad import (
    KIND_PAD_EVENT_V1,
    KIND_PAD_FRAME_V1,
    KIND_PAD_RPC_RESPONSE_V1,
    KIND_PAD_SIGNAL_V1,
    KIND_PAD_STATS_V1,
    PadRpcResponseV1,
)

from .observability.stats import PadStatsTracker
from .pipeline.aggregate import FrameAggregator
from .pipeline.ingest import BoundedPadQueue, IngestLoop, QueueItem
from .pipeline.normalize import NormalizationPipeline
from .rpc.server import PadRpcServer
from .settings import Settings
from .store.redis_store import PadStore


class LandingPadService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.bus = OrionBusAsync(settings.orion_bus_url, enabled=settings.orion_bus_enabled)
        self.queue = BoundedPadQueue(maxsize=settings.pad_max_queue_size, drop_policy=settings.pad_queue_drop_policy)
        self.stats = PadStatsTracker(tick_seconds=settings.pad_stats_tick_sec)

        redis_url = settings.merged_redis_url()
        self.redis: Redis = Redis.from_url(redis_url, decode_responses=False)
        self.store = PadStore(
            redis=self.redis,
            events_stream_key=settings.pad_events_stream_key,
            frames_stream_key=settings.pad_frames_stream_key,
            stream_maxlen=settings.pad_stream_maxlen,
            event_ttl=settings.pad_event_ttl_sec,
            frame_ttl=settings.pad_frame_ttl_sec,
        )

        self.normalizer = NormalizationPipeline(
            app_name=settings.app_name,
            min_salience=settings.pad_min_salience,
            pulse_salience=settings.pad_pulse_salience,
            stats=self.stats,
        )
        self.aggregator = FrameAggregator(
            store=self.store,
            tensor_dim=settings.pad_tensor_dim,
            window_ms=settings.pad_frame_window_ms,
            max_events=settings.pad_max_events_per_tick,
            stats=self.stats,
        )
        self.ingest_loop = IngestLoop(
            bus=self.bus,
            queue=self.queue,
            allowlist=settings.pad_input_allowlist_patterns,
            denylist=settings.pad_input_denylist_patterns,
            app_name=settings.app_name,
            stats=self.stats,
        )
        self.rpc = PadRpcServer(
            bus=self.bus,
            store=self.store,
            settings=settings,
            stats=self.stats,
        )

        self._stop = asyncio.Event()
        self._tasks: List[asyncio.Task] = []

    def _source(self) -> ServiceRef:
        return ServiceRef(name=self.settings.app_name, version=self.settings.service_version, node=self.settings.node_name)

    async def start(self) -> None:
        logger.info("Starting landing pad service")
        await self.bus.connect()
        self.stats.set_queue_depth(0)
        await self.ingest_loop.start()
        await self.rpc.start()
        self._tasks.append(asyncio.create_task(self._queue_worker(), name="pad-queue-worker"))
        self._tasks.append(asyncio.create_task(self._frame_ticker(), name="pad-frame-ticker"))
        self._tasks.append(asyncio.create_task(self._stats_ticker(), name="pad-stats-ticker"))
        self._tasks.append(asyncio.create_task(self._heartbeat_loop(), name="pad-heartbeat"))

    async def stop(self) -> None:
        logger.info("Stopping landing pad service")
        self._stop.set()
        await self.ingest_loop.stop()
        await self.rpc.stop()
        for t in self._tasks:
            t.cancel()
        try:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        except Exception:
            pass
        try:
            await self.bus.close()
        except Exception:
            logger.exception("bus close failed")
        try:
            await self.redis.close()
        except Exception:
            logger.exception("redis close failed")

    async def _queue_worker(self) -> None:
        logger.info("Queue worker started")
        while not self._stop.is_set():
            item = await self.queue.get()
            self.stats.set_queue_depth(self.queue.depth())
            event = await self.normalizer.reduce_and_score(item.envelope, item.channel)
            if event is None:
                continue

            await self.store.store_event(event)
            env = item.envelope.derive_child(
                kind=KIND_PAD_EVENT_V1,
                source=self._source(),
                payload=event.model_dump(mode="json"),
            )
            await self.bus.publish(self.settings.pad_output_event_channel, env)

            if event.salience >= self.settings.pad_pulse_salience:
                signal_env = env.derive_child(
                    kind=KIND_PAD_SIGNAL_V1,
                    source=self._source(),
                    payload={"event_id": event.event_id, "salience": event.salience},
                )
                await self.bus.publish(self.settings.pad_output_signal_channel, signal_env)

            self.aggregator.add_event(event)

    async def _frame_ticker(self) -> None:
        interval = max(self.settings.pad_frame_tick_ms, 100) / 1000.0
        while not self._stop.is_set():
            try:
                frame = await self.aggregator.build_frame()
                if frame:
                    env = BaseEnvelope(
                        kind=KIND_PAD_FRAME_V1,
                        source=self._source(),
                        payload=frame.model_dump(mode="json"),
                    )
                    await self.bus.publish(self.settings.pad_output_frame_channel, env)
            except Exception as exc:
                logger.exception(f"Frame build failed: {exc}")
            await asyncio.sleep(interval)

    async def _heartbeat_loop(self) -> None:
        while not self._stop.is_set():
            try:
                env = BaseEnvelope(
                    kind=self.settings.health_channel,
                    source=self._source(),
                    payload={"status": "ok", "service": self.settings.app_name, "node": self.settings.node_name},
                )
                await self.bus.publish(self.settings.health_channel, env)
            except Exception as exc:
                logger.warning(f"Heartbeat publish failed: {exc}")
            await asyncio.sleep(float(self.settings.heartbeat_interval_sec))

    async def _stats_ticker(self) -> None:
        interval = float(self.settings.pad_stats_tick_sec)
        while not self._stop.is_set():
            try:
                stats_payload = self.stats.snapshot()
                env = BaseEnvelope(
                    kind=KIND_PAD_STATS_V1,
                    source=self._source(),
                    payload=stats_payload,
                )
                await self.bus.publish(self.settings.pad_output_stats_channel, env)
            except Exception as exc:
                logger.warning(f"Stats publish failed: {exc}")
            await asyncio.sleep(interval)
