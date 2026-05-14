# orion/core/bus/work_queue.py
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from redis import asyncio as aioredis
from redis.exceptions import ResponseError

from .async_service import OrionBusAsync
from .bus_schemas import BaseEnvelope
from .codec import OrionCodec

logger = logging.getLogger("orion.bus.work_queue")

SCHEMA_ENTRY = "work_queue.entry.v1"
SCHEMA_DLQ = "work_queue.dlq.v1"
SCHEMA_DLQ_MALFORMED = "work_queue.dlq_malformed.v1"


def _decode_any(v: Any) -> str:
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", "ignore")
    return str(v)


def _field(fields: Mapping[Any, Any], name: str) -> Any:
    if name in fields:
        return fields[name]
    nb = name.encode("utf-8")
    if nb in fields:
        return fields[nb]
    return None


def _normalize_stream_fields(raw: Any) -> dict[Any, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, (list, tuple)):
        out: dict[Any, Any] = {}
        it = iter(raw)
        for k in it:
            out[k] = next(it, None)
        return out
    return {}


def _stream_value_for_xadd(v: Any) -> bytes | str:
    if isinstance(v, (bytes, bytearray)):
        return bytes(v)
    if isinstance(v, bool):
        return b"true" if v else b"false"
    if isinstance(v, int):
        return str(v).encode("utf-8")
    if isinstance(v, float):
        return str(v).encode("utf-8")
    if isinstance(v, str):
        return v.encode("utf-8")
    return json.dumps(v, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _merge_extra_fields(base: dict[bytes, bytes], extra: Mapping[str, Any] | None) -> None:
    if not extra:
        return
    for k, v in extra.items():
        key = k.encode("utf-8") if isinstance(k, str) else bytes(k)
        base[key] = _stream_value_for_xadd(v)


def extract_work_metadata(envelope: BaseEnvelope) -> dict[str, Any]:
    """
    Read optional work metadata from trace['work'] or payload['work'] (dict only).
    Payload wins for overlapping keys after trace.
    """
    out: dict[str, Any] = {}
    tr = envelope.trace or {}
    tw = tr.get("work")
    if isinstance(tw, dict):
        out.update(tw)
    pw = (envelope.payload or {}).get("work")
    if isinstance(pw, dict):
        out.update(pw)
    return out


def copy_envelope_with_work(
    envelope: BaseEnvelope,
    *,
    work_updates: Mapping[str, Any],
) -> BaseEnvelope:
    """
    Shallow-merge work dict into trace['work'] without mutating the original envelope.

    Reads merge trace['work'] then payload['work'] (see extract_work_metadata); updates
    here only modify trace['work'] so payload-based work keys are unchanged on disk.
    """
    tr = dict(envelope.trace or {})
    cur = tr.get("work")
    merged_work: dict[str, Any] = dict(cur) if isinstance(cur, dict) else {}
    merged_work.update(dict(work_updates))
    tr["work"] = merged_work
    return envelope.model_copy(update={"trace": tr})


def _parse_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        v = float(value)
        if v > 1e12:
            return datetime.fromtimestamp(v / 1000.0, tz=timezone.utc)
        return datetime.fromtimestamp(v, tz=timezone.utc)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


@dataclass(frozen=True)
class StreamMessage:
    stream: str
    message_id: str
    envelope: BaseEnvelope
    raw_fields: Mapping[str, Any]
    enqueued_at_ms: int | None
    delivery_count: int | None


@dataclass(frozen=True)
class PendingMessage:
    stream: str
    message_id: str
    consumer: str | None
    idle_ms: int | None
    deliveries: int | None


@dataclass(frozen=True)
class RetryDecision:
    action: str  # retry | dlq | drop
    reason: str
    attempt: int
    max_attempts: int
    delay_ms: int = 0
    dlq_stream: str | None = None


class WorkQueueError(Exception):
    pass


class WorkQueueDecodeError(WorkQueueError):
    def __init__(
        self,
        *,
        stream: str,
        message_id: str,
        error: str,
        raw_fields: Mapping[str, Any],
    ) -> None:
        super().__init__(f"decode_failed stream={stream} id={message_id} error={error}")
        self.stream = stream
        self.message_id = message_id
        self.error = error
        self.raw_fields = raw_fields


@dataclass(frozen=True)
class ReadGroupBatch:
    """Result of XREADGROUP / XAUTOCLAIM: decoded messages plus per-entry decode failures."""

    messages: list[StreamMessage] = field(default_factory=list)
    decode_errors: list[WorkQueueDecodeError] = field(default_factory=list)


class WorkQueueGroupError(WorkQueueError):
    pass


class WorkQueueRetryExhausted(WorkQueueError):
    pass


class RedisStreamWorkQueue:
    """
    Redis Streams consumer-group primitive using OrionCodec-compatible envelope bytes.

    ``read_group`` / ``autoclaim`` return :class:`ReadGroupBatch` so each stream entry is
    decoded independently; malformed entries do not prevent delivery of other messages
    in the same Redis batch.
    """

    def __init__(
        self,
        redis_url: str,
        *,
        codec: OrionCodec | None = None,
        client: aioredis.Redis | None = None,
        default_maxlen: int | None = None,
        logger_: logging.Logger | None = None,
    ) -> None:
        self._redis_url = redis_url
        self.codec = codec or OrionCodec()
        self._client: aioredis.Redis | None = client
        self._owns_client = client is None
        self.default_maxlen = default_maxlen
        self.log = logger_ or logger

    @property
    def client(self) -> aioredis.Redis:
        if self._client is None:
            raise RuntimeError("RedisStreamWorkQueue not connected. Call await connect().")
        return self._client

    async def connect(self) -> None:
        if self._client is not None:
            await self._client.ping()
            return
        self._client = aioredis.from_url(self._redis_url, decode_responses=False)
        await self._client.ping()

    async def close(self) -> None:
        if self._client is None:
            return
        if self._owns_client:
            await self._client.close()
            self._client = None

    async def ensure_group(self, stream: str, group: str, start_id: str = "$", mkstream: bool = True) -> None:
        try:
            await self.client.xgroup_create(stream, group, id=start_id, mkstream=mkstream)
            self.log.info(
                "work_queue_group_ready stream=%s group=%s status=created",
                stream,
                group,
            )
        except ResponseError as e:
            if "BUSYGROUP" in str(e):
                self.log.info(
                    "work_queue_group_ready stream=%s group=%s status=exists",
                    stream,
                    group,
                )
                return
            raise WorkQueueGroupError(str(e)) from e

    def _envelope_bytes(self, envelope: BaseEnvelope) -> bytes:
        return self.codec.encode(envelope)

    def _decode_envelope_field(self, raw: Any, *, stream: str, message_id: str, raw_fields: Mapping[str, Any]) -> BaseEnvelope:
        data = raw
        if isinstance(raw, str):
            data = raw.encode("utf-8")
        if not isinstance(data, (bytes, bytearray)):
            raise WorkQueueDecodeError(
                stream=stream,
                message_id=message_id,
                error="envelope_field_not_bytes",
                raw_fields=raw_fields,
            )
        res = self.codec.decode(bytes(data))
        if not res.ok or res.envelope is None:
            raise WorkQueueDecodeError(
                stream=stream,
                message_id=message_id,
                error=res.error or "decode_failed",
                raw_fields=raw_fields,
            )
        return res.envelope

    async def enqueue(
        self,
        stream: str,
        envelope: BaseEnvelope,
        maxlen: int | None = None,
        approximate: bool = True,
        extra_fields: dict[str, str] | None = None,
    ) -> str:
        await self.connect()
        enc = self._envelope_bytes(envelope)
        now_ms = int(time.time() * 1000)
        fields: dict[bytes, bytes] = {
            b"schema_version": SCHEMA_ENTRY.encode("utf-8"),
            b"envelope": enc,
            b"enqueued_at_ms": str(now_ms).encode("utf-8"),
            b"kind": envelope.kind.encode("utf-8"),
            b"correlation_id": str(envelope.correlation_id).encode("utf-8"),
            b"reply_to": (envelope.reply_to or "").encode("utf-8"),
            b"source": json.dumps(envelope.source.model_dump(mode="json"), ensure_ascii=False).encode("utf-8"),
        }
        _merge_extra_fields(fields, extra_fields)
        trim = maxlen if maxlen is not None else self.default_maxlen
        xadd_kw: dict[str, Any] = {}
        if trim is not None:
            xadd_kw["maxlen"] = int(trim)
            xadd_kw["approximate"] = bool(approximate)
        msg_id = await self.client.xadd(stream, fields, **xadd_kw)
        mid = _decode_any(msg_id)
        wm = extract_work_metadata(envelope)
        self.log.info(
            "work_queue_enqueue stream=%s message_id=%s kind=%s corr=%s job_id=%s lane=%s",
            stream,
            mid,
            envelope.kind,
            envelope.correlation_id,
            wm.get("job_id", ""),
            wm.get("lane", ""),
        )
        return mid

    def _normalize_xread_entries(self, entries: Any) -> list[tuple[str, dict[Any, Any]]]:
        out: list[tuple[str, dict[Any, Any]]] = []
        if not entries:
            return out
        for entry in entries:
            if not entry or len(entry) < 2:
                continue
            mid, raw_fields = entry[0], entry[1]
            fields = _normalize_stream_fields(raw_fields)
            out.append((_decode_any(mid), fields))
        return out

    async def read_group(
        self,
        stream: str,
        group: str,
        consumer: str,
        count: int = 1,
        block_ms: int = 5000,
    ) -> ReadGroupBatch:
        """
        XREADGROUP for new messages. Each stream entry is decoded independently; malformed
        entries appear in ``decode_errors`` so earlier valid messages in the same batch
        are still returned and are not stranded in the PEL.
        """
        await self.connect()
        reply = await self.client.xreadgroup(
            groupname=group,
            consumername=consumer,
            streams={stream: ">"},
            count=count,
            block=block_ms,
        )
        messages: list[StreamMessage] = []
        decode_errors: list[WorkQueueDecodeError] = []
        if not reply:
            return ReadGroupBatch(messages=messages, decode_errors=decode_errors)
        for item in reply:
            if not item or len(item) < 2:
                continue
            _sname, entries = item[0], item[1]
            for mid, fields in self._normalize_xread_entries(entries):
                env_raw = _field(fields, "envelope")
                try:
                    env = self._decode_envelope_field(env_raw, stream=stream, message_id=mid, raw_fields=fields)
                except WorkQueueDecodeError as e:
                    decode_errors.append(e)
                    continue
                enq_raw = _field(fields, "enqueued_at_ms")
                enq_ms: int | None = None
                if enq_raw is not None:
                    try:
                        enq_ms = int(_decode_any(enq_raw))
                    except ValueError:
                        enq_ms = None
                messages.append(
                    StreamMessage(
                        stream=stream,
                        message_id=mid,
                        envelope=env,
                        raw_fields=fields,
                        enqueued_at_ms=enq_ms,
                        delivery_count=None,
                    )
                )
        if messages:
            self.log.info(
                "work_queue_read stream=%s group=%s consumer=%s count=%s",
                stream,
                group,
                consumer,
                len(messages),
            )
        if decode_errors:
            self.log.warning(
                "work_queue_read_decode_errors stream=%s group=%s consumer=%s n=%s",
                stream,
                group,
                consumer,
                len(decode_errors),
            )
        return ReadGroupBatch(messages=messages, decode_errors=decode_errors)

    async def ack(self, stream: str, group: str, message_id: str) -> int:
        await self.connect()
        n = int(await self.client.xack(stream, group, message_id))
        self.log.info(
            "work_queue_ack stream=%s group=%s message_id=%s acked=%s",
            stream,
            group,
            message_id,
            n,
        )
        return n

    def _normalize_pending_summary(self, raw: Any) -> dict[str, Any]:
        if raw is None:
            return {"count": 0, "min_id": None, "max_id": None, "consumers": []}
        if isinstance(raw, dict):
            count = int(raw.get(b"pending", raw.get("pending", 0)))
            min_id = raw.get(b"min", raw.get("min"))
            max_id = raw.get(b"max", raw.get("max"))
            consumers = raw.get(b"consumers", raw.get("consumers", []))
            return {
                "count": count,
                "min_id": None if min_id is None else _decode_any(min_id),
                "max_id": None if max_id is None else _decode_any(max_id),
                "consumers": list(consumers) if isinstance(consumers, (list, tuple)) else [],
            }
        if isinstance(raw, (list, tuple)) and len(raw) >= 4:
            pending, min_id, max_id, consumers = raw[0], raw[1], raw[2], raw[3]
            return {
                "count": int(pending),
                "min_id": None if min_id is None else _decode_any(min_id),
                "max_id": None if max_id is None else _decode_any(max_id),
                "consumers": list(consumers) if isinstance(consumers, (list, tuple)) else [],
            }
        return {"count": 0, "min_id": None, "max_id": None, "consumers": []}

    async def pending_summary(self, stream: str, group: str) -> dict[str, Any]:
        await self.connect()
        raw = await self.client.xpending(stream, group)
        return self._normalize_pending_summary(raw)

    def _normalize_pending_range_row(self, stream: str, row: Any) -> PendingMessage | None:
        if row is None:
            return None
        if isinstance(row, dict):
            mid = row.get("message_id") or row.get(b"message_id")
            cons = row.get("consumer") or row.get(b"consumer")
            idle = row.get("time_since_delivered") or row.get("idle") or row.get(b"time_since_delivered")
            deliv = row.get("times_delivered") or row.get(b"times_delivered")
            return PendingMessage(
                stream=stream,
                message_id=_decode_any(mid) if mid is not None else "",
                consumer=None if cons is None else _decode_any(cons),
                idle_ms=int(idle) if idle is not None else None,
                deliveries=int(deliv) if deliv is not None else None,
            )
        if isinstance(row, (list, tuple)) and len(row) >= 4:
            mid, cons, ms_since, times = row[0], row[1], row[2], row[3]
            return PendingMessage(
                stream=stream,
                message_id=_decode_any(mid),
                consumer=None if cons is None else _decode_any(cons),
                idle_ms=int(ms_since) if ms_since is not None else None,
                deliveries=int(times) if times is not None else None,
            )
        return None

    async def pending_range(
        self,
        stream: str,
        group: str,
        start: str = "-",
        end: str = "+",
        count: int = 10,
        consumer: str | None = None,
    ) -> list[PendingMessage]:
        await self.connect()
        rows = await self.client.xpending_range(stream, group, min=start, max=end, count=count, consumername=consumer)
        out: list[PendingMessage] = []
        for row in rows or []:
            pm = self._normalize_pending_range_row(stream, row)
            if pm:
                out.append(pm)
        return out

    async def autoclaim(
        self,
        stream: str,
        group: str,
        consumer: str,
        min_idle_ms: int,
        start_id: str = "0-0",
        count: int = 10,
    ) -> tuple[str, ReadGroupBatch]:
        await self.connect()
        res = await self.client.xautoclaim(
            stream,
            group,
            consumer,
            min_idle_time=int(min_idle_ms),
            start_id=start_id,
            count=count,
        )
        if not res or len(res) < 2:
            return start_id, ReadGroupBatch()
        next_start = _decode_any(res[0])
        raw_entries = res[1]
        messages: list[StreamMessage] = []
        decode_errors: list[WorkQueueDecodeError] = []
        for entry in raw_entries or []:
            if not entry or len(entry) < 2:
                continue
            mid, raw_fields = entry[0], entry[1]
            fields = _normalize_stream_fields(raw_fields)
            mid_s = _decode_any(mid)
            env_raw = _field(fields, "envelope")
            try:
                env = self._decode_envelope_field(env_raw, stream=stream, message_id=mid_s, raw_fields=fields)
            except WorkQueueDecodeError as e:
                decode_errors.append(e)
                continue
            enq_raw = _field(fields, "enqueued_at_ms")
            enq_ms: int | None = None
            if enq_raw is not None:
                try:
                    enq_ms = int(_decode_any(enq_raw))
                except ValueError:
                    enq_ms = None
            messages.append(
                StreamMessage(
                    stream=stream,
                    message_id=mid_s,
                    envelope=env,
                    raw_fields=fields,
                    enqueued_at_ms=enq_ms,
                    delivery_count=None,
                )
            )
        return next_start, ReadGroupBatch(messages=messages, decode_errors=decode_errors)

    async def send_to_dlq(
        self,
        dlq_stream: str,
        message: StreamMessage,
        error: str,
        reason: str,
        attempt: int,
    ) -> str:
        await self.connect()
        fields: dict[bytes, bytes] = {
            b"schema_version": SCHEMA_DLQ.encode("utf-8"),
            b"original_stream": message.stream.encode("utf-8"),
            b"original_message_id": message.message_id.encode("utf-8"),
            b"error": error.encode("utf-8"),
            b"reason": reason.encode("utf-8"),
            b"attempt": str(attempt).encode("utf-8"),
            b"failed_at_ms": str(int(time.time() * 1000)).encode("utf-8"),
            b"envelope": self._envelope_bytes(message.envelope),
        }
        msg_id = await self.client.xadd(dlq_stream, fields)
        mid = _decode_any(msg_id)
        self.log.info(
            "work_queue_dlq stream=%s dlq_stream=%s message_id=%s reason=%s attempt=%s",
            message.stream,
            dlq_stream,
            message.message_id,
            reason,
            attempt,
        )
        return mid

    async def send_malformed_to_dlq(
        self,
        dlq_stream: str,
        *,
        original_stream: str,
        original_message_id: str,
        raw_fields: Mapping[str, Any],
        error: str,
        reason: str,
    ) -> str:
        """DLQ path for decode failures: preserve raw stream fields (esp. envelope bytes)."""
        await self.connect()
        enc_raw = _field(raw_fields, "envelope")
        if isinstance(enc_raw, (bytes, bytearray)):
            enc_bytes = bytes(enc_raw)
        elif enc_raw is None:
            enc_bytes = b""
        else:
            enc_bytes = repr(enc_raw).encode("utf-8", errors="replace")
        fields: dict[bytes, bytes] = {
            b"schema_version": SCHEMA_DLQ_MALFORMED.encode("utf-8"),
            b"original_stream": original_stream.encode("utf-8"),
            b"original_message_id": original_message_id.encode("utf-8"),
            b"error": error.encode("utf-8"),
            b"reason": reason.encode("utf-8"),
            b"failed_at_ms": str(int(time.time() * 1000)).encode("utf-8"),
            b"envelope": enc_bytes,
        }
        msg_id = await self.client.xadd(dlq_stream, fields)
        return _decode_any(msg_id)

    async def requeue(
        self,
        stream: str,
        message: StreamMessage,
        attempt: int,
        delay_ms: int = 0,
        *,
        reason: str,
    ) -> str:
        await self.connect()
        work_updates: dict[str, Any] = {"attempt": attempt}
        if delay_ms > 0:
            nb = int(time.time() * 1000) + int(delay_ms)
            work_updates["not_before_ms"] = nb
        new_env = copy_envelope_with_work(message.envelope, work_updates=work_updates)
        old_id = message.message_id
        new_id = await self.enqueue(stream, new_env)
        self.log.info(
            "work_queue_requeue stream=%s message_id=%s new_message_id=%s attempt=%s reason=%s",
            stream,
            old_id,
            new_id,
            attempt,
            reason,
        )
        return new_id

    async def stream_info(self, stream: str) -> Any:
        await self.connect()
        return await self.client.xinfo_stream(stream)

    async def group_info(self, stream: str) -> Any:
        await self.connect()
        return await self.client.xinfo_groups(stream)

    async def consumer_info(self, stream: str, group: str) -> Any:
        await self.connect()
        return await self.client.xinfo_consumers(stream, group)


async def queue_rpc_request(
    *,
    queue: RedisStreamWorkQueue,
    reply_bus: OrionBusAsync,
    stream: str,
    envelope: BaseEnvelope,
    reply_channel: str,
    timeout_sec: float,
    maxlen: int | None = None,
    extra_fields: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Subscribe to PubSub reply_channel first, then enqueue to the stream, then await the reply.
    Mirrors OrionBusAsync.rpc_request enough for later adopters (returns raw redis message dict).
    """
    if envelope.reply_to and envelope.reply_to != reply_channel:
        logger.warning(
            "queue_rpc_request reply_channel_mismatch corr=%s envelope.reply_to=%s reply_channel=%s",
            envelope.correlation_id,
            envelope.reply_to,
            reply_channel,
        )
    corr = str(envelope.correlation_id)
    async with reply_bus.subscribe(reply_channel) as pubsub:
        await queue.enqueue(stream, envelope, maxlen=maxlen, extra_fields=extra_fields)

        async def _wait_one() -> dict:
            async for msg in reply_bus.iter_messages(pubsub):
                return msg

        return await asyncio.wait_for(_wait_one(), timeout=timeout_sec)
