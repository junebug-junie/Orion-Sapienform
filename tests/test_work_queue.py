from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

from redis.exceptions import ResponseError

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.codec import OrionCodec
from orion.core.bus.work_queue import (
    RedisStreamWorkQueue,
    StreamMessage,
    copy_envelope_with_work,
    extract_work_metadata,
    queue_rpc_request,
)


def _env(**kwargs) -> BaseEnvelope:
    base = dict(
        kind="test.kind",
        source=ServiceRef(name="svc", node="n1"),
        payload={},
        trace={},
    )
    base.update(kwargs)
    return BaseEnvelope.model_validate(base)


def test_group_create_busygroup_ok() -> None:
    async def _() -> None:
        r = AsyncMock()
        r.ping = AsyncMock()
        r.xgroup_create = AsyncMock(
            side_effect=ResponseError("BUSYGROUP Consumer Group name already exists")
        )
        wq = RedisStreamWorkQueue("redis://x", client=r)
        await wq.ensure_group("s1", "g1")
        r.xgroup_create.assert_awaited_once()

    asyncio.run(_())


def test_enqueue_encodes_base_envelope() -> None:
    async def _() -> None:
        r = AsyncMock()
        r.ping = AsyncMock()
        r.xadd = AsyncMock(return_value=b"1-0")
        codec = OrionCodec()
        env = _env()
        wq = RedisStreamWorkQueue("redis://x", client=r, codec=codec)
        mid = await wq.enqueue("jobs", env)
        assert mid == "1-0"
        args, _kwargs = r.xadd.call_args
        assert args[0] == "jobs"
        fields = args[1]
        assert fields[b"schema_version"] == b"work_queue.entry.v1"
        dec = codec.decode(fields[b"envelope"])
        assert dec.ok and dec.envelope and dec.envelope.kind == "test.kind"

    asyncio.run(_())


def test_read_group_decodes_base_envelope() -> None:
    async def _() -> None:
        r = AsyncMock()
        r.ping = AsyncMock()
        codec = OrionCodec()
        env = _env()
        raw = codec.encode(env)
        r.xreadgroup = AsyncMock(
            return_value=[
                [
                    "jobs",
                    [[b"1-0", {b"envelope": raw, b"enqueued_at_ms": b"123"}]],
                ]
            ]
        )
        wq = RedisStreamWorkQueue("redis://x", client=r, codec=codec)
        batch = await wq.read_group("jobs", "g", "c1", count=1, block_ms=1)
        assert len(batch.messages) == 1
        assert batch.messages[0].message_id == "1-0"
        assert batch.messages[0].envelope.kind == "test.kind"
        assert batch.decode_errors == []

    asyncio.run(_())


def test_read_group_decode_error_collected() -> None:
    async def _() -> None:
        r = AsyncMock()
        r.ping = AsyncMock()
        codec = OrionCodec()
        r.xreadgroup = AsyncMock(
            return_value=[
                [
                    "jobs",
                    [[b"9-0", {b"envelope": b"not-json"}]],
                ]
            ]
        )
        wq = RedisStreamWorkQueue("redis://x", client=r, codec=codec)
        batch = await wq.read_group("jobs", "g", "c1", count=1, block_ms=1)
        assert batch.messages == []
        assert len(batch.decode_errors) == 1
        assert batch.decode_errors[0].message_id == "9-0"
        assert batch.decode_errors[0].stream == "jobs"

    asyncio.run(_())


def test_read_group_batch_decodes_valid_before_invalid() -> None:
    async def _() -> None:
        r = AsyncMock()
        r.ping = AsyncMock()
        codec = OrionCodec()
        good = _env(kind="a.ok")
        raw_ok = codec.encode(good)
        r.xreadgroup = AsyncMock(
            return_value=[
                [
                    "jobs",
                    [
                        [b"1-0", {b"envelope": raw_ok}],
                        [b"2-0", {b"envelope": b"not-json"}],
                    ],
                ]
            ]
        )
        wq = RedisStreamWorkQueue("redis://x", client=r, codec=codec)
        batch = await wq.read_group("jobs", "g", "c1", count=2, block_ms=1)
        assert len(batch.messages) == 1
        assert batch.messages[0].message_id == "1-0"
        assert len(batch.decode_errors) == 1
        assert batch.decode_errors[0].message_id == "2-0"

    asyncio.run(_())


def test_ack_calls_xack() -> None:
    async def _() -> None:
        r = AsyncMock()
        r.ping = AsyncMock()
        r.xack = AsyncMock(return_value=1)
        wq = RedisStreamWorkQueue("redis://x", client=r)
        n = await wq.ack("jobs", "g", "1-0")
        assert n == 1
        r.xack.assert_awaited_once_with("jobs", "g", "1-0")

    asyncio.run(_())


def test_pending_summary_normalizes_response() -> None:
    async def _() -> None:
        r = AsyncMock()
        r.ping = AsyncMock()
        r.xpending = AsyncMock(return_value={b"pending": 3, b"min": b"1-0", b"max": b"2-0", b"consumers": []})
        wq = RedisStreamWorkQueue("redis://x", client=r)
        s = await wq.pending_summary("jobs", "g")
        assert s["count"] == 3
        assert s["min_id"] == "1-0"
        assert s["max_id"] == "2-0"

    asyncio.run(_())


def test_pending_range_normalizes_response() -> None:
    async def _() -> None:
        r = AsyncMock()
        r.ping = AsyncMock()
        r.xpending_range = AsyncMock(
            return_value=[[b"1-0", b"c1", 5000, 2]],
        )
        wq = RedisStreamWorkQueue("redis://x", client=r)
        rows = await wq.pending_range("jobs", "g", count=5)
        assert len(rows) == 1
        assert rows[0].message_id == "1-0"
        assert rows[0].consumer == "c1"
        assert rows[0].idle_ms == 5000
        assert rows[0].deliveries == 2

    asyncio.run(_())


def test_autoclaim_decodes_messages() -> None:
    async def _() -> None:
        r = AsyncMock()
        r.ping = AsyncMock()
        codec = OrionCodec()
        env = _env()
        raw = codec.encode(env)
        r.xautoclaim = AsyncMock(return_value=[b"0-0", [[b"2-0", {b"envelope": raw}]]])
        wq = RedisStreamWorkQueue("redis://x", client=r, codec=codec)
        nxt, batch = await wq.autoclaim("jobs", "g", "c1", 1000, count=2)
        assert nxt == "0-0"
        assert len(batch.messages) == 1
        assert batch.messages[0].message_id == "2-0"
        assert batch.decode_errors == []

    asyncio.run(_())


def test_requeue_copies_envelope_and_increments_attempt() -> None:
    async def _() -> None:
        r = AsyncMock()
        r.ping = AsyncMock()
        r.xadd = AsyncMock(return_value=b"2-0")
        codec = OrionCodec()
        env = _env(trace={"work": {"attempt": 1}})
        wq = RedisStreamWorkQueue("redis://x", client=r, codec=codec)
        sm = StreamMessage(
            stream="jobs",
            message_id="1-0",
            envelope=env,
            raw_fields={},
            enqueued_at_ms=None,
            delivery_count=None,
        )
        new_id = await wq.requeue("jobs", sm, 2, reason="retry")
        assert new_id == "2-0"
        fields = r.xadd.call_args[0][1]
        dec = codec.decode(fields[b"envelope"])
        assert dec.ok and dec.envelope
        assert extract_work_metadata(dec.envelope).get("attempt") == 2

    asyncio.run(_())


def test_dlq_writes_original_envelope_and_error_metadata() -> None:
    async def _() -> None:
        r = AsyncMock()
        r.ping = AsyncMock()
        r.xadd = AsyncMock(return_value=b"99-0")
        codec = OrionCodec()
        env = _env()
        wq = RedisStreamWorkQueue("redis://x", client=r, codec=codec)
        sm = StreamMessage(
            stream="jobs",
            message_id="1-0",
            envelope=env,
            raw_fields={},
            enqueued_at_ms=None,
            delivery_count=None,
        )
        mid = await wq.send_to_dlq("dlq", sm, error="boom", reason="TestErr", attempt=3)
        assert mid == "99-0"
        fields = r.xadd.call_args[0][1]
        assert fields[b"schema_version"] == b"work_queue.dlq.v1"
        assert fields[b"original_message_id"] == b"1-0"
        dec = codec.decode(fields[b"envelope"])
        assert dec.ok and dec.envelope.kind == "test.kind"

    asyncio.run(_())


def test_send_malformed_to_dlq_preserves_non_bytes_envelope() -> None:
    async def _() -> None:
        r = AsyncMock()
        r.ping = AsyncMock()
        r.xadd = AsyncMock(return_value=b"m-1")
        wq = RedisStreamWorkQueue("redis://x", client=r)
        await wq.send_malformed_to_dlq(
            "dlq",
            original_stream="jobs",
            original_message_id="1-0",
            raw_fields={b"envelope": 12345},
            error="bad",
            reason="decode",
        )
        fields = r.xadd.call_args[0][1]
        assert fields[b"envelope"] == repr(12345).encode("utf-8", errors="replace")

    asyncio.run(_())


def test_queue_rpc_request_subscribe_before_enqueue_order() -> None:
    async def _() -> None:
        order: list[str] = []

        class Q:
            async def enqueue(self, *a, **k):
                order.append("enqueue")
                return "1-0"

        q = Q()
        pubsub = MagicMock()

        @asynccontextmanager
        async def sub_cm(*ch, **kw):
            order.append("subscribe_ctx")
            yield pubsub

        reply_bus = MagicMock()
        reply_bus.subscribe = sub_cm

        async def it_messages(_p):
            order.append("iter")
            yield {"type": "message", "data": b"{}"}

        reply_bus.iter_messages = it_messages
        env = _env()
        await queue_rpc_request(
            queue=q,  # type: ignore[arg-type]
            reply_bus=reply_bus,  # type: ignore[arg-type]
            stream="jobs",
            envelope=env,
            reply_channel="rep:1",
            timeout_sec=2.0,
        )
        assert order[0] == "subscribe_ctx"
        assert order.index("subscribe_ctx") < order.index("enqueue")

    asyncio.run(_())


def test_copy_envelope_with_work_no_mutation() -> None:
    env = _env(trace={"work": {"attempt": 1}, "x": 1})
    e2 = copy_envelope_with_work(env, work_updates={"attempt": 2})
    assert extract_work_metadata(env).get("attempt") == 1
    assert extract_work_metadata(e2).get("attempt") == 2
    assert env.trace["x"] == 1
