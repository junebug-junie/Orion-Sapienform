"""Durable intake: read a Redis Stream through a consumer group and hand each envelope to the
same ``handle_envelope`` path the pub/sub Hunter uses.

Why (#3920): ``orion:gpu_pool:event`` is pub/sub. Pub/sub keeps nothing, so every pool event
published while sql-writer was down (17-24 s per ``up -d --build``, 14 rebuilds in 30 hours on
2026-11-24/25) was never written to ``gpu_pool_events``. The pool now also XADDs each event to
``orion:stream:gpu_pool:event``. The stream holds entries while this service is away; the
consumer group remembers where it stopped, so the backlog is written when it comes back.

Delivery is at-least-once. That is safe here because the sink is idempotent: ``gpu_pool_events``
is insert-only on ``event_id`` (INSERT_ONLY_MODELS), so a redelivery is a duplicate skip.

Same shape as orion/spark/concept_induction/wp_stream_consumer.py (the live world-pulse stream
consumer), without its in-memory run_id guard: the table's primary key is the guard.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Optional

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.core.bus.work_queue import RedisStreamWorkQueue, StreamMessage, WorkQueueDecodeError

logger = logging.getLogger("sql-writer.durable_stream_intake")

Handler = Callable[[BaseEnvelope], Awaitable[None]]


class DurableStreamIntake:
    """One stream, one consumer group, one handler.

    ``start_id="0"`` on first group creation: the first boot after deploy also writes the entries
    the pool streamed before this group existed (the old pub/sub writer already wrote most of
    them; the primary key skips those). After that the group exists and its cursor rules.
    """

    def __init__(
        self,
        *,
        queue: RedisStreamWorkQueue,
        stream: str,
        group: str,
        consumer: str,
        dlq_stream: str,
        handler: Handler,
        max_attempts: int = 20,
        block_ms: int = 5000,
        autoclaim_idle_ms: int = 30000,
        count: int = 100,
        idle_backoff_sec: float = 1.0,
        start_id: str = "0",
        logger_: Optional[logging.Logger] = None,
    ) -> None:
        self.queue = queue
        self.stream = stream
        self.group = group
        self.consumer = consumer
        self.dlq_stream = dlq_stream
        self.handler = handler
        self.max_attempts = max(1, int(max_attempts))
        self.block_ms = int(block_ms)
        self.autoclaim_idle_ms = int(autoclaim_idle_ms)
        self.count = max(1, int(count))
        self.idle_backoff_sec = float(idle_backoff_sec)
        self.start_id = start_id
        self.log = logger_ or logger
        self._group_ready = False
        self._stopped = False
        self.written = 0
        self.failed = 0
        self.dead_lettered = 0

    async def ensure_ready(self) -> None:
        if not self._group_ready:
            await self.queue.ensure_group(self.stream, self.group, start_id=self.start_id, mkstream=True)
            self._group_ready = True

    async def _process(self, msg: StreamMessage) -> None:
        # handle_envelope routes a failed row write to bus_fallback_log itself and returns; it
        # raises only when that fallback write also fails (Postgres unreachable). Then the entry
        # stays pending and reclaim_stale retries it.
        await self.handler(msg.envelope)
        await self.queue.ack(self.stream, self.group, msg.message_id)
        self.written += 1

    async def _dlq_decode_errors(self, decode_errors: list[WorkQueueDecodeError]) -> None:
        for err in decode_errors:
            await self.queue.send_malformed_to_dlq(
                self.dlq_stream,
                original_stream=self.stream,
                original_message_id=err.message_id,
                raw_fields=err.raw_fields,
                error=err.error,
                reason="decode_failed",
            )
            await self.queue.ack(self.stream, self.group, err.message_id)
            self.dead_lettered += 1
            self.log.warning("durable_intake_dlq_malformed stream=%s message_id=%s error=%s",
                             self.stream, err.message_id, err.error)

    async def _deliveries_for(self, message_id: str) -> Optional[int]:
        rows = await self.queue.pending_range(self.stream, self.group, start=message_id, end=message_id, count=1)
        for row in rows:
            if row.message_id == message_id:
                return row.deliveries
        return None

    async def reclaim_stale(self) -> int:
        """Entries read but never acked (handler raised, or the last process died mid-batch)."""
        _next, batch = await self.queue.autoclaim(
            self.stream, self.group, self.consumer,
            min_idle_ms=self.autoclaim_idle_ms, start_id="0-0", count=self.count,
        )
        await self._dlq_decode_errors(batch.decode_errors)
        handled = len(batch.decode_errors)
        for msg in batch.messages:
            attempts = await self._deliveries_for(msg.message_id)
            if attempts is not None and attempts > self.max_attempts:
                await self.queue.send_to_dlq(self.dlq_stream, msg, error="handler_failed",
                                             reason="max_attempts_exceeded", attempt=attempts)
                await self.queue.ack(self.stream, self.group, msg.message_id)
                self.dead_lettered += 1
                self.log.warning("durable_intake_dlq stream=%s message_id=%s attempts=%s",
                                 self.stream, msg.message_id, attempts)
                handled += 1
                continue
            try:
                await self._process(msg)
                handled += 1
            except Exception:  # noqa: BLE001 -- left pending for the next pass
                self.failed += 1
                self.log.warning("durable_intake_reclaim_failed stream=%s message_id=%s (left pending)",
                                 self.stream, msg.message_id, exc_info=True)
        return handled

    async def drain_new(self) -> int:
        batch = await self.queue.read_group(self.stream, self.group, self.consumer,
                                            count=self.count, block_ms=self.block_ms)
        await self._dlq_decode_errors(batch.decode_errors)
        handled = 0
        for msg in batch.messages:
            try:
                await self._process(msg)
                handled += 1
            except Exception:  # noqa: BLE001 -- unacked: reclaim_stale retries it
                self.failed += 1
                self.log.warning("durable_intake_handle_failed stream=%s message_id=%s (left pending)",
                                 self.stream, msg.message_id, exc_info=True)
        return handled

    async def run_once(self) -> int:
        await self.ensure_ready()
        return await self.reclaim_stale() + await self.drain_new()

    def stop(self) -> None:
        self._stopped = True

    async def run_forever(self) -> None:
        self.log.warning("durable_intake_started stream=%s group=%s consumer=%s",
                         self.stream, self.group, self.consumer)
        while not self._stopped:
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 -- the intake must never die (Redis blip, etc.)
                self.log.exception("durable_intake_loop_error stream=%s", self.stream)
                await asyncio.sleep(self.idle_backoff_sec)
