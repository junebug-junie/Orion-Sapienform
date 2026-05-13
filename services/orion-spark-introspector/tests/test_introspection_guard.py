from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from app import introspection_guard as ig


def test_try_claim_inflight_setnx_true():
    async def _go() -> None:
        redis = MagicMock()
        redis.set = AsyncMock(return_value=True)
        assert await ig.try_claim_inflight(redis, prefix="p", trace_id="t1", owner="n1", ttl_sec=30) is True
        redis.set.assert_awaited_once()
        assert redis.set.await_args.kwargs.get("nx") is True

    asyncio.run(_go())


def test_try_claim_inflight_setnx_false():
    async def _go() -> None:
        redis = MagicMock()
        redis.set = AsyncMock(return_value=None)
        assert await ig.try_claim_inflight(redis, prefix="p", trace_id="t1", owner="n1", ttl_sec=30) is False

    asyncio.run(_go())


def test_is_done_true_when_key_present():
    async def _go() -> None:
        redis = MagicMock()
        redis.get = AsyncMock(return_value=b"ok")
        assert await ig.is_done(redis, prefix="p", trace_id="t1") is True

    asyncio.run(_go())


def test_release_deletes_inflight():
    async def _go() -> None:
        redis = MagicMock()
        redis.delete = AsyncMock(return_value=1)
        await ig.release_inflight(redis, prefix="p", trace_id="t1")
        redis.delete.assert_awaited_once()

    asyncio.run(_go())
