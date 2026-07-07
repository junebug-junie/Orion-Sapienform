"""Durable consumer for the world-pulse run-result stream.

Why this exists: `orion:world_pulse:run:result` is a rare (once-per-run) event that
drives the autonomy episode journal. On pub/sub it is fire-and-forget with no replay,
so a busy or briefly-disconnected worker silently loses it (observed live 2026-07-07).
This consumer reads the durable Redis Stream mirror via a consumer group so delivery is
at-least-once with crash recovery (autoclaim) and a dead-letter path for poison messages.

The module is transport-only: it owns read_group/ack/autoclaim/DLQ + an idempotency
guard, and delegates all envelope processing to an injected async ``handler`` (the same
``handle_envelope`` path the pub/sub loop uses). This keeps it unit-testable with a fake
queue and isolates the durability seam from the 1000-line worker.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Awaitable, Callable, Optional

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.core.bus.work_queue import RedisStreamWorkQueue, StreamMessage

logger = logging.getLogger("orion.spark.concept_induction.wp_stream")

Handler = Callable[[BaseEnvelope], Awaitable[None]]


def extract_run_id(envelope: BaseEnvelope) -> str:
    """Stable idempotency key for a run-result envelope: the world-pulse run_id.

    Falls back to the envelope correlation_id so we always have *some* stable key even
    if the payload shape is unexpected.
    """
    payload = envelope.payload
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump()
    if isinstance(payload, dict):
        run = payload.get("run")
        if isinstance(run, dict):
            run_id = run.get("run_id")
            if run_id:
                return str(run_id)
    return str(envelope.correlation_id or "")


class WorldPulseStreamConsumer:
    """Consumer-group reader for the world-pulse run-result stream.

    Delivery contract: at-least-once. The idempotency guard (``is_processed`` /
    ``mark_processed``, keyed on world-pulse run_id) makes reprocessing safe, so a
    redelivery after a crash-before-ack does not compose a duplicate episode journal.
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
        is_processed: Callable[[str], bool],
        mark_processed: Callable[[str], None],
        max_attempts: int = 5,
        block_ms: int = 5000,
        autoclaim_idle_ms: int = 120000,
        count: int = 1,
        idle_backoff_sec: float = 1.0,
        logger_: Optional[logging.Logger] = None,
    ) -> None:
        self.queue = queue
        self.stream = stream
        self.group = group
        self.consumer = consumer
        self.dlq_stream = dlq_stream
        self.handler = handler
        self.is_processed = is_processed
        self.mark_processed = mark_processed
        self.max_attempts = max(1, int(max_attempts))
        self.block_ms = int(block_ms)
        self.autoclaim_idle_ms = int(autoclaim_idle_ms)
        self.count = max(1, int(count))
        self.idle_backoff_sec = float(idle_backoff_sec)
        self.log = logger_ or logger
        self._group_ready = False
        self._stopped = False

    async def ensure_ready(self) -> None:
        if not self._group_ready:
            # start_id="$" => only runs published after the group is first created are
            # delivered as new; anything enqueued while the group already exists is
            # retained by the stream and picked up on restart.
            await self.queue.ensure_group(self.stream, self.group, start_id="$", mkstream=True)
            self._group_ready = True

    async def _process(self, msg: StreamMessage) -> None:
        """Idempotent handle + ack for a single stream message."""
        run_id = extract_run_id(msg.envelope)
        if self.is_processed(run_id):
            self.log.info(
                "wp_stream_skip_duplicate stream=%s message_id=%s run_id=%s",
                self.stream,
                msg.message_id,
                run_id,
            )
            await self.queue.ack(self.stream, self.group, msg.message_id)
            return
        await self.handler(msg.envelope)
        self.mark_processed(run_id)
        await self.queue.ack(self.stream, self.group, msg.message_id)
        self.log.info(
            "wp_stream_processed stream=%s message_id=%s run_id=%s",
            self.stream,
            msg.message_id,
            run_id,
        )

    async def _dlq_decode_errors(self, decode_errors) -> None:
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
            self.log.warning(
                "wp_stream_dlq_malformed stream=%s message_id=%s error=%s",
                self.stream,
                err.message_id,
                err.error,
            )

    async def _deliveries_for(self, message_id: str) -> Optional[int]:
        rows = await self.queue.pending_range(
            self.stream, self.group, start=message_id, end=message_id, count=1
        )
        for row in rows:
            if row.message_id == message_id:
                return row.deliveries
        return None

    async def reclaim_stale(self) -> int:
        """Reclaim messages left pending by a crashed/slow consumer; DLQ the poisonous.

        Returns the number of messages reclaimed (processed or DLQ'd) this pass.
        """
        _next, batch = await self.queue.autoclaim(
            self.stream,
            self.group,
            self.consumer,
            min_idle_ms=self.autoclaim_idle_ms,
            start_id="0-0",
            count=self.count,
        )
        await self._dlq_decode_errors(batch.decode_errors)
        if not batch.messages:
            return len(batch.decode_errors)
        handled = 0
        for msg in batch.messages:
            # Look up the delivery count for THIS specific id (autoclaim bumps it) so the
            # DLQ decision can't be missed by a bounded PEL window.
            attempts = await self._deliveries_for(msg.message_id)
            if attempts is not None and attempts > self.max_attempts:
                await self.queue.send_to_dlq(
                    self.dlq_stream,
                    msg,
                    error="handler_failed",
                    reason="max_attempts_exceeded",
                    attempt=attempts,
                )
                await self.queue.ack(self.stream, self.group, msg.message_id)
                self.log.warning(
                    "wp_stream_dlq stream=%s message_id=%s attempts=%s",
                    self.stream,
                    msg.message_id,
                    attempts,
                )
                handled += 1
                continue
            try:
                await self._process(msg)
                handled += 1
            except Exception:  # noqa: BLE001 - leave in PEL for the next reclaim pass
                self.log.warning(
                    "wp_stream_reclaim_handle_failed stream=%s message_id=%s (left pending)",
                    self.stream,
                    msg.message_id,
                    exc_info=True,
                )
        return handled + len(batch.decode_errors)

    async def drain_new(self) -> int:
        """Read + handle new (never-delivered) messages. Returns count handled."""
        batch = await self.queue.read_group(
            self.stream,
            self.group,
            self.consumer,
            count=self.count,
            block_ms=self.block_ms,
        )
        await self._dlq_decode_errors(batch.decode_errors)
        handled = 0
        for msg in batch.messages:
            try:
                await self._process(msg)
                handled += 1
            except Exception:  # noqa: BLE001 - unacked => retried via reclaim_stale
                self.log.warning(
                    "wp_stream_handle_failed stream=%s message_id=%s (left pending for retry)",
                    self.stream,
                    msg.message_id,
                    exc_info=True,
                )
        return handled

    async def run_once(self) -> int:
        await self.ensure_ready()
        reclaimed = await self.reclaim_stale()
        drained = await self.drain_new()
        return reclaimed + drained

    def stop(self) -> None:
        self._stopped = True

    async def run_forever(self) -> None:
        self.log.info(
            "wp_stream_consumer_started stream=%s group=%s consumer=%s",
            self.stream,
            self.group,
            self.consumer,
        )
        while not self._stopped:
            try:
                await self.run_once()
            except asyncio.CancelledError:
                self.log.info("wp_stream_consumer_cancelled stream=%s", self.stream)
                raise
            except Exception:  # noqa: BLE001 - never let the durable consumer die
                self.log.exception("wp_stream_consumer_loop_error stream=%s", self.stream)
                await asyncio.sleep(self.idle_backoff_sec)
        self.log.info("wp_stream_consumer_stopped stream=%s", self.stream)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
