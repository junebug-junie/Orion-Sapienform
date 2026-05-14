from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from app import introspection_guard as ig


def test_try_claim_inflight_without_redis() -> None:
    async def _run() -> None:
        class S:
            spark_introspection_idempotency_enable = False
            spark_introspection_key_prefix = "spark:introspection"
            spark_introspection_inflight_ttl_sec = 60

        ok = await ig.try_claim_inflight(None, settings=S(), trace_id="t1", owner="n:1")
        assert ok is True

    asyncio.run(_run())


def test_mark_done_release_no_redis() -> None:
    async def _run() -> None:
        class S:
            spark_introspection_key_prefix = "spark:introspection"
            spark_introspection_done_ttl_sec = 60

        await ig.mark_done(None, settings=S(), trace_id="t1")
        await ig.release_inflight(None, settings=S(), trace_id="t1")

    asyncio.run(_run())


def test_is_done_with_redis_mock() -> None:
    async def _run() -> None:
        r = AsyncMock()
        r.exists = AsyncMock(return_value=1)

        class S:
            spark_introspection_key_prefix = "p"

        assert await ig.is_done(r, settings=S(), trace_id="x") is True

    asyncio.run(_run())
