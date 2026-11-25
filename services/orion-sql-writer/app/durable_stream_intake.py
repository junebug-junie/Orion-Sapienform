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
import time
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

    Dead letters go by AGE, not delivery count: an entry the handler cannot write is retried until
    it is ``dlq_after_sec`` old (default 12 h, inside the stream's ~1 day MAXLEN horizon). A count
    (20 deliveries x 30 s) would dead-letter every event of a 10-minute Postgres outage into a DLQ
    nothing reads -- the #3920 loss again, later. Undecodable entries go to the DLQ at once.
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
        dlq_after_sec: float = 43200.0,
        block_ms: int = 5000,
        autoclaim_idle_ms: int = 30000,
        count: int = 100,
        idle_backoff_sec: float = 1.0,
        start_id: str = "0",
        now_ms: Callable[[], int] | None = None,
        logger_: Optional[logging.Logger] = None,
    ) -> None:
        self.queue = queue
        self.stream = stream
        self.group = group
        self.consumer = consumer
        self.dlq_stream = dlq_stream
        self.handler = handler
        self.dlq_after_ms = int(float(dlq_after_sec) * 1000)
        self.block_ms = int(block_ms)
        self.autoclaim_idle_ms = int(autoclaim_idle_ms)
        self.count = max(1, int(count))
        self.idle_backoff_sec = float(idle_backoff_sec)
        self.start_id = start_id
        self.now_ms = now_ms or (lambda: int(time.time() * 1000))
        self.log = logger_ or logger
        self._group_ready = False
        self._recovered = False
        self._stopped = False
        self.written = 0
        self.failed = 0
        self.dead_lettered = 0

    async def ensure_ready(self) -> None:
        if not self._group_ready:
            # The intake owns its queue's connection: main.py builds the queue from a URL only. connect()
            # is a ping when already connected, and a Redis outage at boot raises here, so run_forever
            # retries it instead of failing every loop on "not connected" (live 2026-11-25 08:29-10:30Z).
            await self.queue.connect()
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

    def _age_ms(self, msg: StreamMessage) -> int:
        """Entry age from its enqueue stamp, else from the stream id's millisecond part."""
        born = msg.enqueued_at_ms
        if born is None:
            try:
                born = int(msg.message_id.split("-", 1)[0])
            except ValueError:
                return 0
        return self.now_ms() - int(born)

    async def _handle_batch(self, messages: list[StreamMessage], *, where: str) -> int:
        handled, first_error = 0, True
        for msg in messages:
            try:
                await self._process(msg)
                handled += 1
            except Exception:  # noqa: BLE001 -- unacked: reclaim_stale retries it
                self.failed += 1
                # One traceback per batch: a Postgres outage fails every entry the same way.
                self.log.warning("durable_intake_%s_failed stream=%s message_id=%s (left pending)",
                                 where, self.stream, msg.message_id, exc_info=first_error)
                first_error = False
        return handled

    async def reclaim_stale(self, *, min_idle_ms: int | None = None) -> int:
        """Entries read but never acked (handler raised, or the last process died mid-batch)."""
        _next, batch = await self.queue.autoclaim(
            self.stream, self.group, self.consumer,
            min_idle_ms=self.autoclaim_idle_ms if min_idle_ms is None else min_idle_ms,
            start_id="0-0", count=self.count,
        )
        await self._dlq_decode_errors(batch.decode_errors)
        retry: list[StreamMessage] = []
        handled = len(batch.decode_errors)
        for msg in batch.messages:
            age_ms = self._age_ms(msg)
            if age_ms > self.dlq_after_ms:
                await self.queue.send_to_dlq(self.dlq_stream, msg, error="handler_failed",
                                             reason="too_old_to_retry", attempt=0)
                await self.queue.ack(self.stream, self.group, msg.message_id)
                self.dead_lettered += 1
                self.log.warning("durable_intake_dlq stream=%s message_id=%s age_s=%.0f",
                                 self.stream, msg.message_id, age_ms / 1000)
                handled += 1
            else:
                retry.append(msg)
        return handled + await self._handle_batch(retry, where="reclaim")

    async def drain_new(self) -> int:
        batch = await self.queue.read_group(self.stream, self.group, self.consumer,
                                            count=self.count, block_ms=self.block_ms)
        await self._dlq_decode_errors(batch.decode_errors)
        return len(batch.decode_errors) + await self._handle_batch(batch.messages, where="handle")

    async def run_once(self) -> int:
        await self.ensure_ready()
        handled = 0
        if not self._recovered:
            # Boot: entries the previous process read but never acked (it died mid-batch) are
            # taken back at once, not after autoclaim_idle_ms. sql-writer is the group's only reader.
            while True:
                n = await self.reclaim_stale(min_idle_ms=0)
                handled += n
                if n < self.count:
                    break
            self._recovered = True
        return handled + await self.reclaim_stale() + await self.drain_new()

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
