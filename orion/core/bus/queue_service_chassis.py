# orion/core/bus/queue_service_chassis.py
from __future__ import annotations

import asyncio
import logging
import os
import time
import traceback
from datetime import datetime, timezone
from time import perf_counter
from typing import Awaitable, Callable
from uuid import uuid4

from .async_service import OrionBusAsync
from .bus_schemas import BaseEnvelope
from .bus_service_chassis import BaseChassis, ChassisConfig
from .work_queue import (
    RedisStreamWorkQueue,
    StreamMessage,
    extract_work_metadata,
    _parse_ts,
)

logger = logging.getLogger("orion.bus.queue_rabbit")

Handler = Callable[[BaseEnvelope], Awaitable[BaseEnvelope | None]]


def _consumer_suffix() -> str:
    return f"{os.getpid()}:{uuid4().hex[:8]}"


def _attempt_from_envelope(env: BaseEnvelope) -> int:
    w = extract_work_metadata(env)
    raw = w.get("attempt", 1)
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 1


def _reply_optional(env: BaseEnvelope) -> bool:
    w = extract_work_metadata(env)
    return bool(w.get("reply_optional"))


def _not_before_ms(env: BaseEnvelope) -> int | None:
    w = extract_work_metadata(env)
    v = w.get("not_before_ms")
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _expires_at(env: BaseEnvelope) -> datetime | None:
    w = extract_work_metadata(env)
    if "expires_at_ms" in w:
        try:
            ms = int(w["expires_at_ms"])
            return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
        except (TypeError, ValueError):
            pass
    return _parse_ts(w.get("expires_at"))


class QueueRabbit(BaseChassis):
    """
    Stream consumer chassis: one worker per delivery (Redis consumer group),
    replies on existing PubSub (OrionBusAsync) reply_to channels.
    """

    def __init__(
        self,
        cfg: ChassisConfig,
        *,
        stream: str,
        group: str,
        consumer: str | None = None,
        handler: Handler,
        reply_bus: OrionBusAsync | None = None,
        work_queue: RedisStreamWorkQueue | None = None,
        concurrent_handlers: bool = False,
        max_inflight: int = 1,
        read_count: int = 1,
        block_ms: int = 5000,
        max_attempts: int = 3,
        dlq_stream: str | None = None,
        reclaim_pending: bool = True,
        reclaim_min_idle_ms: int = 120_000,
        stale_policy: str = "drop",
        heartbeat_enabled: bool = True,
        reclaim_every_loops: int = 5,
        reclaim_every_sec: float = 30.0,
    ) -> None:
        super().__init__(cfg)
        self.stream = stream
        self.group = group
        self.consumer = consumer or f"{cfg.service_name}:{cfg.node_name or 'node'}:{_consumer_suffix()}"
        self.handler = handler
        self._reply_bus = reply_bus or self.bus
        if work_queue is None:
            raise ValueError("QueueRabbit requires work_queue=RedisStreamWorkQueue(...)")
        self.work_queue = work_queue
        self.concurrent_handlers = bool(concurrent_handlers)
        self.max_inflight = max(1, int(max_inflight))
        self.read_count = max(1, int(read_count))
        self.block_ms = int(block_ms)
        self.max_attempts = int(max_attempts)
        self.dlq_stream = dlq_stream
        self.reclaim_pending = bool(reclaim_pending)
        self.reclaim_min_idle_ms = int(reclaim_min_idle_ms)
        self.stale_policy = stale_policy
        self.heartbeat_enabled = bool(heartbeat_enabled)
        self.reclaim_every_loops = max(1, int(reclaim_every_loops))
        self.reclaim_every_sec = float(reclaim_every_sec)
        self._inflight: set[asyncio.Task] = set()
        self._sem = asyncio.Semaphore(self.max_inflight)
        self._loop_iter = 0
        self._last_reclaim = perf_counter()

    async def _heartbeat_loop(self) -> None:
        if not self.heartbeat_enabled:
            await self._stop.wait()
            return
        await super()._heartbeat_loop()

    def _log_ctx(self, env: BaseEnvelope | None, sm: StreamMessage | None) -> str:
        parts: list[str] = [
            f"stream={self.stream}",
            f"group={self.group}",
            f"consumer={self.consumer}",
        ]
        if sm:
            parts.append(f"message_id={sm.message_id}")
        if env:
            wm = extract_work_metadata(env)
            parts.extend(
                [
                    f"corr={env.correlation_id}",
                    f"trace_id={(env.trace or {}).get('trace_id', '')}",
                    f"job_id={wm.get('job_id', '')}",
                    f"idempotency_key={wm.get('idempotency_key', '')}",
                    f"lane={wm.get('lane', '')}",
                    f"attempt={_attempt_from_envelope(env)}",
                ]
            )
        return " ".join(parts)

    async def _publish_reply(self, env: BaseEnvelope, out: BaseEnvelope) -> None:
        if not env.reply_to:
            return
        await self._reply_bus.publish(env.reply_to, out)
        ctx = self._log_ctx(env, None)
        logger.info(
            f"queue_rabbit_reply_publish {ctx} reply_to={env.reply_to} kind_out={out.kind}",
        )

    async def _handle_decode_error(self, err: WorkQueueDecodeError) -> None:
        logger.error(
            f"queue_rabbit_decode_error stream={err.stream} group={self.group} consumer={self.consumer} "
            f"message_id={err.message_id} error={err.error}",
        )
        if self.dlq_stream:
            try:
                await self.work_queue.send_malformed_to_dlq(
                    self.dlq_stream,
                    original_stream=err.stream,
                    original_message_id=err.message_id,
                    raw_fields=err.raw_fields,
                    error=err.error,
                    reason="decode_failed",
                )
                await self.work_queue.ack(err.stream, self.group, err.message_id)
            except Exception as e:
                logger.error(f"queue_rabbit_dlq decode_dlq_failed error={e}")
        else:
            logger.error(
                f"queue_rabbit_decode_error_no_dlq_pending stream={err.stream} message_id={err.message_id} "
                "(message stays pending)",
            )

    async def _expire_or_dlq(self, sm: StreamMessage, *, action: str, expires_at: str) -> bool:
        """Returns True if message should be acked without handler."""
        if action == "drop":
            logger.info(
                f"queue_rabbit_expired stream={self.stream} message_id={sm.message_id} "
                f"action=drop expires_at={expires_at}",
            )
            return True
        if action == "dlq" and self.dlq_stream:
            logger.info(
                f"queue_rabbit_expired stream={self.stream} message_id={sm.message_id} "
                f"action=dlq expires_at={expires_at}",
            )
            await self.work_queue.send_to_dlq(
                self.dlq_stream,
                sm,
                error="expired",
                reason="expires_at_elapsed",
                attempt=_attempt_from_envelope(sm.envelope),
            )
            return True
        if action == "run":
            logger.info(
                f"queue_rabbit_expired stream={self.stream} message_id={sm.message_id} "
                f"action=run expires_at={expires_at}",
            )
            return False
        logger.warning(
            f"queue_rabbit_expired stream={self.stream} message_id={sm.message_id} "
            f"action={action} expires_at={expires_at} (fallback no-op)",
        )
        return False

    async def _process_one(self, sm: StreamMessage) -> None:
        env = sm.envelope
        t0 = perf_counter()
        logger.info(f"queue_rabbit_handler_start {self._log_ctx(env, sm)}")

        exp = _expires_at(env)
        if exp is not None:
            now = time.time()
            if exp.timestamp() < now:
                pol = self.stale_policy
                if pol == "dlq" and not self.dlq_stream:
                    pol = "drop"
                should_ack_only = await self._expire_or_dlq(
                    sm,
                    action=pol,
                    expires_at=exp.isoformat(),
                )
                if should_ack_only:
                    await self.work_queue.ack(self.stream, self.group, sm.message_id)
                    return
                if pol != "run":
                    return

        nb = _not_before_ms(env)
        if nb is not None and nb > int(time.time() * 1000):
            await self.work_queue.requeue(
                self.stream,
                sm,
                _attempt_from_envelope(env),
                delay_ms=0,
                reason="not_before",
            )
            await self.work_queue.ack(self.stream, self.group, sm.message_id)
            logger.info(f"queue_rabbit_retry {self._log_ctx(env, sm)} reason=not_before")
            return

        out: BaseEnvelope | None
        try:
            out = await self.handler(env)
        except Exception as e:
            await self._on_handler_exception(sm, env, e)
            return

        if out is None and env.reply_to and not _reply_optional(env):
            await self._on_handler_exception(
                sm,
                env,
                RuntimeError("handler_returned_none_with_reply_to"),
            )
            return

        if out is not None and env.reply_to:
            try:
                await self._publish_reply(env, out)
            except Exception as e:
                logger.error(
                    f"queue_rabbit_handler_error {self._log_ctx(env, sm)} reason=reply_publish_failed error={e}",
                )
                await self._on_handler_exception(
                    sm,
                    env,
                    RuntimeError(f"reply_publish_failed: {e}"),
                )
                return

        await self.work_queue.ack(self.stream, self.group, sm.message_id)
        ms = (perf_counter() - t0) * 1000.0
        logger.info(f"queue_rabbit_handler_complete {self._log_ctx(env, sm)} runtime_ms={ms:.1f}")

    async def _on_handler_exception(self, sm: StreamMessage, env: BaseEnvelope, err: BaseException) -> None:
        logger.error(
            f"queue_rabbit_handler_error {self._log_ctx(env, sm)} error={err}\n{traceback.format_exc()}",
        )
        current = _attempt_from_envelope(env)
        if current < self.max_attempts:
            try:
                await self.work_queue.requeue(
                    self.stream,
                    sm,
                    current + 1,
                    delay_ms=0,
                    reason=type(err).__name__,
                )
                await self.work_queue.ack(self.stream, self.group, sm.message_id)
                logger.info(
                    f"queue_rabbit_retry {self._log_ctx(env, sm)} reason={type(err).__name__} "
                    f"next_attempt={current + 1}",
                )
            except Exception as re:
                logger.error(
                    f"queue_rabbit_handler_error {self._log_ctx(env, sm)} reason=requeue_failed error={re}",
                )
            return

        if self.dlq_stream:
            try:
                await self.work_queue.send_to_dlq(
                    self.dlq_stream,
                    sm,
                    error=str(err),
                    reason=type(err).__name__,
                    attempt=current,
                )
                await self.work_queue.ack(self.stream, self.group, sm.message_id)
                logger.info(f"queue_rabbit_dlq {self._log_ctx(env, sm)} reason=max_attempts")
            except Exception as de:
                logger.error(
                    f"queue_rabbit_handler_error {self._log_ctx(env, sm)} reason=dlq_write_failed error={de}",
                )
        else:
            logger.error(
                f"queue_rabbit_handler_error {self._log_ctx(env, sm)} reason=max_attempts_no_dlq "
                "(message stays pending)",
            )

    async def _reclaim_tick(self) -> None:
        if not self.reclaim_pending:
            return
        now = perf_counter()
        self._loop_iter += 1
        due = (self._loop_iter % self.reclaim_every_loops == 0) or (now - self._last_reclaim >= self.reclaim_every_sec)
        if not due:
            return
        self._last_reclaim = now
        try:
            _next, batch = await self.work_queue.autoclaim(
                self.stream,
                self.group,
                self.consumer,
                self.reclaim_min_idle_ms,
                start_id="0-0",
                count=self.read_count,
            )
            for err in batch.decode_errors:
                await self._handle_decode_error(err)
            claimed = batch.messages
            if claimed:
                logger.info(
                    f"queue_rabbit_claim stream={self.stream} group={self.group} consumer={self.consumer} "
                    f"count={len(claimed)} next_start={_next}",
                )
            for sm in claimed:
                await self._dispatch_sm(sm)
        except Exception:
            logger.exception("queue_rabbit_claim unexpected_error")

    def _active_inflight_count(self) -> int:
        return sum(1 for t in self._inflight if not t.done())

    def _capacity_count(self) -> int:
        if not self.concurrent_handlers:
            return 1
        free = self.max_inflight - self._active_inflight_count()
        return max(0, min(self.read_count, free))

    async def _dispatch_sm(self, sm: StreamMessage) -> None:
        if self.concurrent_handlers:
            async def _wrapped() -> None:
                async with self._sem:
                    await self._process_one(sm)

            t = asyncio.create_task(_wrapped())
            self._inflight.add(t)

            def _done(tk: asyncio.Task) -> None:
                self._inflight.discard(tk)

            t.add_done_callback(_done)
            return
        await self._process_one(sm)

    async def _run(self) -> None:
        await self.work_queue.connect()
        await self.work_queue.ensure_group(self.stream, self.group)
        logger.info(
            f"queue_rabbit_start service={self.cfg.service_name} stream={self.stream} group={self.group} "
            f"consumer={self.consumer} max_inflight={self.max_inflight}",
        )
        try:
            while not self._stop.is_set():
                await self._reclaim_tick()
                if self.concurrent_handlers:
                    while (
                        self._active_inflight_count() >= self.max_inflight
                        and self._inflight
                        and not self._stop.is_set()
                    ):
                        await asyncio.wait(
                            self._inflight,
                            return_when=asyncio.FIRST_COMPLETED,
                            timeout=1.0,
                        )
                cap = self._capacity_count()
                if cap <= 0:
                    await asyncio.sleep(0.02)
                    continue
                batch = await self.work_queue.read_group(
                    self.stream,
                    self.group,
                    self.consumer,
                    count=cap,
                    block_ms=self.block_ms,
                )
                for err in batch.decode_errors:
                    await self._handle_decode_error(err)
                msgs = batch.messages

                if not msgs:
                    continue

                for sm in msgs:
                    if self._stop.is_set():
                        break
                    await self._dispatch_sm(sm)

                if self.concurrent_handlers and self._inflight:
                    await asyncio.sleep(0)

            if self._inflight:
                for t in list(self._inflight):
                    if not t.done():
                        t.cancel()
                await asyncio.gather(*self._inflight, return_exceptions=True)
        finally:
            try:
                await self.work_queue.close()
            except Exception:
                logger.exception("queue_rabbit_work_queue_close_failed")
            logger.info(
                f"queue_rabbit_stop service={self.cfg.service_name} stream={self.stream} inflight={len(self._inflight)}",
            )
