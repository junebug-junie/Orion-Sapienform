from __future__ import annotations

import asyncio
import os
import time
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.bus_service_chassis import ChassisConfig
from orion.core.bus.codec import OrionCodec
from orion.core.bus.queue_service_chassis import QueueRabbit
from orion.core.bus.work_queue import StreamMessage, WorkQueueDecodeError


async def _noop_handler(_e: BaseEnvelope) -> None:
    return None


def _cfg() -> ChassisConfig:
    return ChassisConfig(
        service_name="test-svc",
        service_version="0",
        node_name="node-a",
        bus_url="redis://localhost:6379/0",
        bus_enabled=False,
    )


def _env(**kwargs) -> BaseEnvelope:
    base = dict(
        kind="q.test",
        source=ServiceRef(name="svc", node="n1"),
        payload={},
        trace={},
    )
    base.update(kwargs)
    return BaseEnvelope.model_validate(base)


def _sm(env: BaseEnvelope, mid: str = "1-0") -> StreamMessage:
    return StreamMessage(
        stream="jobs",
        message_id=mid,
        envelope=env,
        raw_fields={},
        enqueued_at_ms=None,
        delivery_count=None,
    )


def _qr(
    *,
    handler,
    wq: AsyncMock | None = None,
    reply_bus=None,
    max_attempts: int = 3,
    dlq_stream: str | None = None,
    stale_policy: str = "drop",
    concurrent_handlers: bool = False,
    max_inflight: int = 1,
) -> QueueRabbit:
    wq_supplied = wq is not None
    wq = wq or AsyncMock()
    if not wq_supplied:
        wq.connect = AsyncMock()
        wq.ensure_group = AsyncMock()
        wq.ack = AsyncMock(return_value=1)
        wq.requeue = AsyncMock(return_value="2-0")
        wq.send_to_dlq = AsyncMock(return_value="d-0")
        wq.send_malformed_to_dlq = AsyncMock(return_value="m-0")
    bus_supplied = reply_bus is not None
    rb = reply_bus or MagicMock()
    if not bus_supplied:
        rb.publish = AsyncMock()
    return QueueRabbit(
        _cfg(),
        stream="jobs",
        group="g1",
        consumer="c-fixed",
        handler=handler,
        reply_bus=rb,
        work_queue=wq,  # type: ignore[arg-type]
        max_attempts=max_attempts,
        dlq_stream=dlq_stream,
        stale_policy=stale_policy,
        concurrent_handlers=concurrent_handlers,
        max_inflight=max_inflight,
        heartbeat_enabled=False,
        reclaim_pending=False,
    )


def test_handler_success_no_reply_acks() -> None:
    async def _() -> None:
        wq = AsyncMock()
        wq.connect = AsyncMock()
        wq.ensure_group = AsyncMock()
        wq.ack = AsyncMock(return_value=1)
        wq.requeue = AsyncMock()
        wq.send_to_dlq = AsyncMock()
        wq.send_malformed_to_dlq = AsyncMock()
        env = _env()
        calls = []

        async def h(e):
            calls.append("h")
            return None

        qr = _qr(handler=h, wq=wq)
        await qr._process_one(_sm(env))
        wq.ack.assert_awaited_once_with("jobs", "g1", "1-0")
        assert calls == ["h"]

    asyncio.run(_())


def test_handler_success_with_reply_publishes_then_acks() -> None:
    async def _() -> None:
        wq = AsyncMock()
        wq.connect = AsyncMock()
        wq.ensure_group = AsyncMock()
        wq.ack = AsyncMock(return_value=1)
        wq.requeue = AsyncMock()
        wq.send_to_dlq = AsyncMock()
        wq.send_malformed_to_dlq = AsyncMock()
        rb = MagicMock()
        rb.publish = AsyncMock()
        env = _env(reply_to="rep:x")
        out = _env(kind="q.resp", reply_to=None)

        async def h(e):
            return out

        qr = _qr(handler=h, wq=wq, reply_bus=rb)
        await qr._process_one(_sm(env))
        rb.publish.assert_awaited_once_with("rep:x", out)
        wq.ack.assert_awaited_once_with("jobs", "g1", "1-0")

    asyncio.run(_())


def test_reply_publish_failure_triggers_retry() -> None:
    async def _() -> None:
        wq = AsyncMock()
        wq.connect = AsyncMock()
        wq.ensure_group = AsyncMock()
        wq.ack = AsyncMock(return_value=1)
        wq.requeue = AsyncMock(return_value="2-0")
        wq.send_to_dlq = AsyncMock()
        wq.send_malformed_to_dlq = AsyncMock()
        rb = MagicMock()
        rb.publish = AsyncMock(side_effect=RuntimeError("pubfail"))
        env = _env(reply_to="rep:x")

        async def h(e):
            return _env(kind="out")

        qr = _qr(handler=h, wq=wq, reply_bus=rb)
        await qr._process_one(_sm(env))
        wq.requeue.assert_awaited()
        wq.ack.assert_awaited_once_with("jobs", "g1", "1-0")

    asyncio.run(_())


def test_handler_returns_none_with_reply_to_is_error() -> None:
    async def _() -> None:
        wq = AsyncMock()
        wq.connect = AsyncMock()
        wq.ensure_group = AsyncMock()
        wq.ack = AsyncMock(return_value=1)
        wq.requeue = AsyncMock(return_value="2-0")
        wq.send_to_dlq = AsyncMock()
        wq.send_malformed_to_dlq = AsyncMock()
        env = _env(reply_to="rep:x")

        async def h(e):
            return None

        qr = _qr(handler=h, wq=wq, max_attempts=3)
        await qr._process_one(_sm(env))
        wq.requeue.assert_awaited()
        wq.ack.assert_awaited()

    asyncio.run(_())


def test_handler_failure_requeues_before_ack() -> None:
    async def _() -> None:
        wq = AsyncMock()
        wq.connect = AsyncMock()
        wq.ensure_group = AsyncMock()
        wq.ack = AsyncMock(return_value=1)
        wq.requeue = AsyncMock(return_value="2-0")
        wq.send_to_dlq = AsyncMock()
        wq.send_malformed_to_dlq = AsyncMock()
        env = _env()

        async def h(e):
            raise ValueError("nope")

        qr = _qr(handler=h, wq=wq)
        await qr._process_one(_sm(env))
        wq.requeue.assert_awaited()
        wq.ack.assert_awaited_once_with("jobs", "g1", "1-0")

    asyncio.run(_())


def test_handler_failure_dlq_after_max_attempts() -> None:
    async def _() -> None:
        wq = AsyncMock()
        wq.connect = AsyncMock()
        wq.ensure_group = AsyncMock()
        wq.ack = AsyncMock(return_value=1)
        wq.requeue = AsyncMock()
        wq.send_to_dlq = AsyncMock(return_value="d-0")
        wq.send_malformed_to_dlq = AsyncMock()
        env = _env(trace={"work": {"attempt": 3}})

        async def h(e):
            raise ValueError("nope")

        qr = _qr(handler=h, wq=wq, max_attempts=3, dlq_stream="dlq")
        await qr._process_one(_sm(env))
        wq.send_to_dlq.assert_awaited()
        wq.requeue.assert_not_called()
        wq.ack.assert_awaited_once_with("jobs", "g1", "1-0")

    asyncio.run(_())


def test_dlq_failure_does_not_ack_original() -> None:
    async def _() -> None:
        wq = AsyncMock()
        wq.connect = AsyncMock()
        wq.ensure_group = AsyncMock()
        wq.ack = AsyncMock(return_value=1)
        wq.requeue = AsyncMock()
        wq.send_to_dlq = AsyncMock(side_effect=RuntimeError("dlqfail"))
        wq.send_malformed_to_dlq = AsyncMock()
        env = _env(trace={"work": {"attempt": 3}})

        async def h(e):
            raise ValueError("nope")

        qr = _qr(handler=h, wq=wq, max_attempts=3, dlq_stream="dlq")
        await qr._process_one(_sm(env))
        wq.ack.assert_not_called()

    asyncio.run(_())


def test_expired_drop_acks_without_handler() -> None:
    async def _() -> None:
        wq = AsyncMock()
        wq.connect = AsyncMock()
        wq.ensure_group = AsyncMock()
        wq.ack = AsyncMock(return_value=1)
        wq.requeue = AsyncMock()
        wq.send_to_dlq = AsyncMock()
        wq.send_malformed_to_dlq = AsyncMock()
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        env = _env(trace={"work": {"expires_at": past}})
        ran = []

        async def h(e):
            ran.append(1)
            return None

        qr = _qr(handler=h, wq=wq, stale_policy="drop")
        await qr._process_one(_sm(env))
        assert ran == []
        wq.ack.assert_awaited_once_with("jobs", "g1", "1-0")

    asyncio.run(_())


def test_expired_dlq_acks_after_dlq() -> None:
    async def _() -> None:
        wq = AsyncMock()
        wq.connect = AsyncMock()
        wq.ensure_group = AsyncMock()
        wq.ack = AsyncMock(return_value=1)
        wq.requeue = AsyncMock()
        wq.send_to_dlq = AsyncMock()
        wq.send_malformed_to_dlq = AsyncMock()
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        env = _env(trace={"work": {"expires_at": past}})
        ran = []

        async def h(e):
            ran.append(1)
            return None

        qr = _qr(handler=h, wq=wq, stale_policy="dlq", dlq_stream="dlq")
        await qr._process_one(_sm(env))
        assert ran == []
        wq.send_to_dlq.assert_awaited()
        wq.ack.assert_awaited_once_with("jobs", "g1", "1-0")

    asyncio.run(_())


def test_not_before_requeues_and_acks_original() -> None:
    async def _() -> None:
        wq = AsyncMock()
        wq.connect = AsyncMock()
        wq.ensure_group = AsyncMock()
        wq.ack = AsyncMock(return_value=1)
        wq.requeue = AsyncMock(return_value="2-0")
        wq.send_to_dlq = AsyncMock()
        wq.send_malformed_to_dlq = AsyncMock()
        nb = int(time.time() * 1000) + 3600_000
        env = _env(trace={"work": {"attempt": 2, "not_before_ms": nb}})
        ran = []

        async def h(e):
            ran.append(1)
            return None

        qr = _qr(handler=h, wq=wq)
        await qr._process_one(_sm(env))
        assert ran == []
        wq.requeue.assert_awaited()
        wq.ack.assert_awaited_once_with("jobs", "g1", "1-0")

    asyncio.run(_())


def test_concurrent_max_inflight() -> None:
    async def _() -> None:
        qr = _qr(handler=_noop_handler, concurrent_handlers=True, max_inflight=3)
        t1 = asyncio.create_task(asyncio.sleep(60))
        t2 = asyncio.create_task(asyncio.sleep(60))
        qr._inflight.update({t1, t2})
        assert qr._active_inflight_count() == 2
        assert qr._capacity_count() == 1
        t1.cancel()
        t2.cancel()
        with suppress(asyncio.CancelledError):
            await t1
        with suppress(asyncio.CancelledError):
            await t2

    asyncio.run(_())


def test_shutdown_does_not_ack_cancelled_task() -> None:
    async def _() -> None:
        wq = AsyncMock()
        wq.connect = AsyncMock()
        wq.ensure_group = AsyncMock()
        wq.ack = AsyncMock(return_value=1)
        wq.requeue = AsyncMock()
        wq.send_to_dlq = AsyncMock()
        wq.send_malformed_to_dlq = AsyncMock()
        env = _env()

        async def h(e):
            await asyncio.Event().wait()

        qr = _qr(handler=h, wq=wq)
        t = asyncio.create_task(qr._process_one(_sm(env)))
        await asyncio.sleep(0.05)
        t.cancel()
        with suppress(asyncio.CancelledError):
            await t
        wq.ack.assert_not_called()

    asyncio.run(_())


def test_decode_failure_dlq_and_ack_when_dlq_configured() -> None:
    async def _() -> None:
        wq = AsyncMock()
        wq.connect = AsyncMock()
        wq.ensure_group = AsyncMock()
        wq.ack = AsyncMock(return_value=1)
        wq.requeue = AsyncMock()
        wq.send_to_dlq = AsyncMock()
        wq.send_malformed_to_dlq = AsyncMock(return_value="m-1")
        err = WorkQueueDecodeError(
            stream="jobs",
            message_id="9-0",
            error="bad",
            raw_fields={b"envelope": b"x"},
        )
        qr = _qr(handler=_noop_handler, wq=wq, dlq_stream="dlq")
        await qr._handle_decode_error(err)
        wq.send_malformed_to_dlq.assert_awaited()
        wq.ack.assert_awaited_once_with("jobs", "g1", "9-0")

    asyncio.run(_())


def test_decode_failure_no_dlq_leaves_pending() -> None:
    async def _() -> None:
        wq = AsyncMock()
        wq.connect = AsyncMock()
        wq.ensure_group = AsyncMock()
        wq.ack = AsyncMock(return_value=1)
        wq.requeue = AsyncMock()
        wq.send_to_dlq = AsyncMock()
        wq.send_malformed_to_dlq = AsyncMock()
        err = WorkQueueDecodeError(
            stream="jobs",
            message_id="9-0",
            error="bad",
            raw_fields={b"envelope": b"x"},
        )
        qr = _qr(handler=_noop_handler, wq=wq, dlq_stream=None)
        await qr._handle_decode_error(err)
        wq.send_malformed_to_dlq.assert_not_called()
        wq.ack.assert_not_called()

    asyncio.run(_())


def test_reply_optional_none_with_reply_does_not_error() -> None:
    async def _() -> None:
        wq = AsyncMock()
        wq.connect = AsyncMock()
        wq.ensure_group = AsyncMock()
        wq.ack = AsyncMock(return_value=1)
        wq.requeue = AsyncMock()
        wq.send_to_dlq = AsyncMock()
        wq.send_malformed_to_dlq = AsyncMock()
        rb = MagicMock()
        rb.publish = AsyncMock()
        env = _env(reply_to="rep:x", trace={"work": {"reply_optional": True}})

        async def h(e):
            return None

        qr = _qr(handler=h, wq=wq, reply_bus=rb)
        await qr._process_one(_sm(env))
        rb.publish.assert_not_called()
        wq.ack.assert_awaited_once_with("jobs", "g1", "1-0")

    asyncio.run(_())


@pytest.mark.skipif(
    not os.environ.get("ORION_REDIS_STREAM_TEST_URL"),
    reason="ORION_REDIS_STREAM_TEST_URL not set",
)
def test_live_two_consumers_one_message() -> None:
    async def _() -> None:
        import os

        from redis import asyncio as aioredis

        from orion.core.bus.work_queue import RedisStreamWorkQueue

        url = os.environ["ORION_REDIS_STREAM_TEST_URL"]
        stream = f"orion:test:stream:{os.getpid()}"
        group = "g"
        codec = OrionCodec()
        client = aioredis.from_url(url, decode_responses=False)
        wq = RedisStreamWorkQueue(url, client=client, codec=codec)
        await wq.connect()
        try:
            await client.delete(stream)
        except Exception:
            pass
        await wq.ensure_group(stream, group, start_id="0", mkstream=True)
        env = _env()
        await wq.enqueue(stream, env)
        got: list[str] = []

        async def read_once(name: str) -> int:
            batch = await wq.read_group(stream, group, name, count=1, block_ms=2000)
            got.extend([name] * len(batch.messages))
            for m in batch.messages:
                await wq.ack(stream, group, m.message_id)
            return len(batch.messages)

        n1, n2 = await asyncio.gather(read_once("c1"), read_once("c2"))
        assert n1 + n2 == 1
        assert len(got) == 1
        pend = await wq.pending_summary(stream, group)
        assert pend.get("count", 0) == 0
        await client.delete(stream)
        await client.aclose()

    asyncio.run(_())
