"""#3920 regression: pool events published while sql-writer is down are written when it returns.

Before: sql-writer read orion:gpu_pool:event over pub/sub. Pub/sub keeps nothing, so every
``up -d --build`` (17-24 s) dropped the pool events published meanwhile -- 68 of 69 lease grants
missing from gpu_pool_events on 2026-11-24/25 fell inside a sql-writer rebuild or manual
redeploy window. Now the pool XADDs each event to orion:stream:gpu_pool:event and sql-writer
reads it through a consumer group.

These drive the real intake (app/durable_stream_intake.py) over fakeredis, and the real
handle_envelope -> _write_row insert-only path into a SQLite gpu_pool_events table.
"""
from __future__ import annotations

import asyncio
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import fakeredis.aioredis
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef  # noqa: E402
from orion.core.bus.work_queue import RedisStreamWorkQueue  # noqa: E402
from orion.schemas.gpu_pool import (  # noqa: E402
    GPU_POOL_EVENT_CHANNEL, GPU_POOL_EVENT_KIND, GPU_POOL_EVENT_STREAM, GPU_POOL_EVENT_STREAM_DLQ, GpuPoolEventV1,
)

from app import worker  # noqa: E402
from app.durable_stream_intake import DurableStreamIntake  # noqa: E402
from app.models.fallback_log import BusFallbackLog  # noqa: E402
from app.models.gpu_pool_event import GpuPoolEventSQL  # noqa: E402
from app.settings import Settings  # noqa: E402

GROUP = "cg:sql-writer"


def _event(lease: str, name: str = "granted") -> BaseEnvelope:
    ev = GpuPoolEventV1(event=name, lease_id=lease, holder="cortex-exec", work_class="fast", priority="interactive",
                        role="fast", generated_at=datetime.now(timezone.utc))
    return BaseEnvelope(kind=GPU_POOL_EVENT_KIND, source=ServiceRef(name="orion-gpu-pool"),
                        correlation_id=uuid.uuid4(), payload=ev.model_dump(mode="json"))


@pytest.fixture()
def table(monkeypatch):
    """gpu_pool_events in SQLite, behind the worker's own session seam (PK on event_id)."""
    # One shared connection: _write_row runs in a worker thread (asyncio.to_thread).
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    GpuPoolEventSQL.__table__.create(engine)
    BusFallbackLog.__table__.create(engine)
    maker = sessionmaker(bind=engine)
    monkeypatch.setattr(worker, "get_session", maker)
    monkeypatch.setattr(worker, "remove_session", lambda: None)

    def rows() -> list[str]:
        with maker() as s:
            assert s.query(BusFallbackLog).count() == 0  # written for real, not parked in the fallback log
            return [r.event_id for r in s.query(GpuPoolEventSQL).order_by(GpuPoolEventSQL.generated_at)]
    return rows


def _queue(redis) -> RedisStreamWorkQueue:
    return RedisStreamWorkQueue("redis://unused", client=redis)


def _intake(queue: RedisStreamWorkQueue, handler=None, **kw) -> DurableStreamIntake:
    async def real(env):
        await worker.handle_envelope(env)
    return DurableStreamIntake(queue=queue, stream=GPU_POOL_EVENT_STREAM, group=GROUP, consumer="sql-writer-test",
                               dlq_stream=GPU_POOL_EVENT_STREAM_DLQ, handler=handler or real, block_ms=10, **kw)


def test_events_published_while_the_writer_is_down_are_written_when_it_returns(table) -> None:
    async def go():
        redis = fakeredis.aioredis.FakeRedis()
        pool = _queue(redis)  # the producer side, exactly as orion-gpu-pool's main.py builds it

        before = _event("lease-before")
        await pool.enqueue(GPU_POOL_EVENT_STREAM, before)
        up = _intake(_queue(redis))
        assert await up.run_once() == 1

        # sql-writer goes down (up -d --build): nothing is reading. The pool keeps granting.
        during = [_event(f"lease-down-{i}") for i in range(70)]
        for env in during:
            await pool.enqueue(GPU_POOL_EVENT_STREAM, env)

        # A fresh process (new container), same group: the backlog is written, once.
        back = _intake(_queue(redis))
        while await back.run_once():
            pass
        ids = table()
        assert ids == [before.payload["event_id"]] + [e.payload["event_id"] for e in during]
        pending = await back.queue.pending_summary(GPU_POOL_EVENT_STREAM, GROUP)
        assert pending["count"] == 0
    asyncio.run(go())


def test_first_boot_writes_entries_streamed_before_the_group_existed_and_dedupes_the_rest(table) -> None:
    """Deploy order: the pool streams first; the old pub/sub writer already wrote some of those rows."""
    async def go():
        redis = fakeredis.aioredis.FakeRedis()
        pool = _queue(redis)
        envs = [_event(f"lease-{i}") for i in range(5)]
        for env in envs:
            await pool.enqueue(GPU_POOL_EVENT_STREAM, env)
        await worker.handle_envelope(envs[0])  # written earlier by the pub/sub intake
        intake = _intake(_queue(redis))
        await intake.run_once()
        assert sorted(table()) == sorted(e.payload["event_id"] for e in envs)
        assert intake.written == 5  # the duplicate is acked too, not left pending
    asyncio.run(go())


def test_postgres_down_leaves_the_entry_pending_through_the_real_handler_then_writes_it(table, monkeypatch) -> None:
    """The real handle_envelope: the row write AND its bus_fallback_log write both fail -> it
    raises, the intake does not ack, and the entry is written once Postgres is back."""
    async def go():
        redis = fakeredis.aioredis.FakeRedis()
        env = _event("lease-db-down")
        await _queue(redis).enqueue(GPU_POOL_EVENT_STREAM, env)
        working = worker.get_session

        def down():
            raise ConnectionError("postgres unreachable")

        monkeypatch.setattr(worker, "get_session", down)
        intake = _intake(_queue(redis), autoclaim_idle_ms=0)
        await intake.run_once()
        assert intake.failed == 1 and intake.written == 0
        assert (await intake.queue.pending_summary(GPU_POOL_EVENT_STREAM, GROUP))["count"] == 1

        monkeypatch.setattr(worker, "get_session", working)
        await intake.run_once()
        assert table() == [env.payload["event_id"]]
        assert (await intake.queue.pending_summary(GPU_POOL_EVENT_STREAM, GROUP))["count"] == 0
        assert await redis.xlen(GPU_POOL_EVENT_STREAM_DLQ) == 0
    asyncio.run(go())


def test_an_outage_is_waited_out_and_only_an_entry_older_than_dlq_after_is_dead_lettered(table) -> None:
    """By age, not delivery count: many failed passes inside the window never dead-letter."""
    async def go():
        redis = fakeredis.aioredis.FakeRedis()
        env = _event("lease-long-outage")
        await _queue(redis).enqueue(GPU_POOL_EVENT_STREAM, env)
        clock = {"ms": int(datetime.now(timezone.utc).timestamp() * 1000)}

        async def down(e):
            raise ConnectionError("postgres unreachable")

        intake = _intake(_queue(redis), handler=down, autoclaim_idle_ms=0, dlq_after_sec=3600,
                         now_ms=lambda: clock["ms"])
        for _ in range(30):  # 30 failed passes, entry still young
            await intake.run_once()
        assert await redis.xlen(GPU_POOL_EVENT_STREAM_DLQ) == 0
        assert (await intake.queue.pending_summary(GPU_POOL_EVENT_STREAM, GROUP))["count"] == 1

        clock["ms"] += 3601 * 1000
        await intake.run_once()
        assert await redis.xlen(GPU_POOL_EVENT_STREAM_DLQ) == 1
        assert (await intake.queue.pending_summary(GPU_POOL_EVENT_STREAM, GROUP))["count"] == 0
        assert intake.dead_lettered == 1 and table() == []
    asyncio.run(go())


def test_entries_read_but_not_acked_by_a_crashed_process_are_written_at_boot(table) -> None:
    """The container dies mid-batch: delivered to sql-writer-<node>, never acked. The next boot
    takes them back at once, not after autoclaim_idle_ms (30 s)."""
    async def go():
        redis = fakeredis.aioredis.FakeRedis()
        envs = [_event(f"lease-crash-{i}") for i in range(3)]
        q = _queue(redis)
        await q.ensure_group(GPU_POOL_EVENT_STREAM, GROUP, start_id="0")
        for env in envs:
            await q.enqueue(GPU_POOL_EVENT_STREAM, env)
        crashed = await q.read_group(GPU_POOL_EVENT_STREAM, GROUP, "sql-writer-test", count=10, block_ms=10)
        assert len(crashed.messages) == 3  # read, never acked

        fresh = _intake(_queue(redis))  # default autoclaim_idle_ms=30000
        await fresh.run_once()
        assert sorted(table()) == sorted(e.payload["event_id"] for e in envs)
        assert (await fresh.queue.pending_summary(GPU_POOL_EVENT_STREAM, GROUP))["count"] == 0
    asyncio.run(go())


def test_an_undecodable_entry_goes_to_the_dlq_and_does_not_block_the_rest(table) -> None:
    async def go():
        redis = fakeredis.aioredis.FakeRedis()
        intake = _intake(_queue(redis))
        await intake.ensure_ready()
        await redis.xadd(GPU_POOL_EVENT_STREAM, {b"envelope": b"not an envelope"})
        good = _event("lease-good")
        await _queue(redis).enqueue(GPU_POOL_EVENT_STREAM, good)
        await intake.run_once()
        assert table() == [good.payload["event_id"]]
        assert await redis.xlen(GPU_POOL_EVENT_STREAM_DLQ) == 1
    asyncio.run(go())


def test_stream_on_drops_the_pubsub_subscription_even_from_a_stale_env_list() -> None:
    stale = Settings(SQL_WRITER_SUBSCRIBE_CHANNELS=[GPU_POOL_EVENT_CHANNEL, "orion:biometrics:summary"])
    assert stale.gpu_pool_event_stream_enabled is True
    assert GPU_POOL_EVENT_CHANNEL not in stale.effective_subscribe_channels
    assert stale.gpu_pool_event_stream_key == GPU_POOL_EVENT_STREAM
    assert stale.gpu_pool_event_stream_dlq == GPU_POOL_EVENT_STREAM_DLQ
    assert stale.effective_stream_channels == [GPU_POOL_EVENT_STREAM]
    off = Settings(SQL_WRITER_SUBSCRIBE_CHANNELS=["orion:biometrics:summary"],
                   SQL_WRITER_GPU_POOL_EVENT_STREAM_ENABLED=False)
    assert GPU_POOL_EVENT_CHANNEL in off.effective_subscribe_channels  # rollback: pub/sub again
