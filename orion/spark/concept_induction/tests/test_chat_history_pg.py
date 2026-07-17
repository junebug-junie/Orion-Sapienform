from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from orion.spark.concept_induction import chat_history_pg


class _FakeRow(dict):
    pass


class _FakePool:
    def __init__(self, row):
        self._row = row
        self.fetch_calls = 0
        self.closed = False

    async def fetchrow(self, query, correlation_id):
        self.fetch_calls += 1
        return self._row

    async def close(self):
        self.closed = True


class _FakeAsyncpg:
    def __init__(self, row=None, raise_on_create_pool=False, raise_on_fetch=False):
        self._row = row
        self._raise_on_create_pool = raise_on_create_pool
        self._raise_on_fetch = raise_on_fetch
        self.pools_created = []

    async def create_pool(self, dsn, min_size, max_size, timeout, command_timeout):
        if self._raise_on_create_pool:
            raise RuntimeError("connection refused")
        pool = _FakePool(self._row)
        if self._raise_on_fetch:
            async def _raise_fetchrow(query, correlation_id):
                pool.fetch_calls += 1
                raise RuntimeError("query failed")

            pool.fetchrow = _raise_fetchrow  # type: ignore[assignment]
        self.pools_created.append(pool)
        return pool


class ChatHistoryPgTests(unittest.TestCase):
    def setUp(self):
        chat_history_pg._pool = None
        chat_history_pg._pool_dsn = None

    def tearDown(self):
        chat_history_pg._pool = None
        chat_history_pg._pool_dsn = None

    def test_returns_prompt_and_response_on_hit(self):
        fake = _FakeAsyncpg(row=_FakeRow(prompt="hello orion", response="hello juniper"))
        with patch.object(chat_history_pg, "asyncpg", fake):
            result = asyncio.run(
                chat_history_pg.fetch_chat_turn_by_correlation_id(
                    "corr-1", dsn="postgresql://x", retries=1
                )
            )
        self.assertEqual(result, ("hello orion", "hello juniper"))

    def test_returns_none_without_correlation_id(self):
        fake = _FakeAsyncpg(row=_FakeRow(prompt="x", response="y"))
        with patch.object(chat_history_pg, "asyncpg", fake):
            result = asyncio.run(
                chat_history_pg.fetch_chat_turn_by_correlation_id("", dsn="postgresql://x")
            )
        self.assertIsNone(result)

    def test_fails_open_when_pool_creation_fails(self):
        fake = _FakeAsyncpg(raise_on_create_pool=True)
        with patch.object(chat_history_pg, "asyncpg", fake):
            result = asyncio.run(
                chat_history_pg.fetch_chat_turn_by_correlation_id(
                    "corr-1", dsn="postgresql://x", retries=2, retry_delay_sec=0.0
                )
            )
        self.assertIsNone(result)

    def test_retries_on_query_error_not_just_on_miss(self):
        fake = _FakeAsyncpg(raise_on_fetch=True)
        with patch.object(chat_history_pg, "asyncpg", fake):
            result = asyncio.run(
                chat_history_pg.fetch_chat_turn_by_correlation_id(
                    "corr-1", dsn="postgresql://x", retries=3, retry_delay_sec=0.0
                )
            )
        self.assertIsNone(result)
        # The query is retried on a transient error, not just on a clean miss.
        self.assertEqual(fake.pools_created[0].fetch_calls, 3)

    def test_retries_then_falls_back_when_row_never_found(self):
        fake = _FakeAsyncpg(row=None)
        with patch.object(chat_history_pg, "asyncpg", fake):
            result = asyncio.run(
                chat_history_pg.fetch_chat_turn_by_correlation_id(
                    "corr-1", dsn="postgresql://x", retries=3, retry_delay_sec=0.0
                )
            )
        self.assertIsNone(result)
        self.assertEqual(fake.pools_created[0].fetch_calls, 3)

    def test_pool_is_reused_across_calls_with_same_dsn(self):
        fake = _FakeAsyncpg(row=_FakeRow(prompt="a", response="b"))
        with patch.object(chat_history_pg, "asyncpg", fake):
            asyncio.run(
                chat_history_pg.fetch_chat_turn_by_correlation_id(
                    "corr-1", dsn="postgresql://x", retries=1
                )
            )
            asyncio.run(
                chat_history_pg.fetch_chat_turn_by_correlation_id(
                    "corr-2", dsn="postgresql://x", retries=1
                )
            )
        self.assertEqual(len(fake.pools_created), 1)

    def test_returns_none_when_asyncpg_unavailable(self):
        with patch.object(chat_history_pg, "asyncpg", None):
            result = asyncio.run(
                chat_history_pg.fetch_chat_turn_by_correlation_id("corr-1", dsn="postgresql://x")
            )
        self.assertIsNone(result)

    def test_close_pool_resets_state(self):
        fake = _FakeAsyncpg(row=_FakeRow(prompt="a", response="b"))
        with patch.object(chat_history_pg, "asyncpg", fake):
            asyncio.run(
                chat_history_pg.fetch_chat_turn_by_correlation_id(
                    "corr-1", dsn="postgresql://x", retries=1
                )
            )
            self.assertIsNotNone(chat_history_pg._pool)
            asyncio.run(chat_history_pg.close_pool())
        self.assertIsNone(chat_history_pg._pool)
        self.assertIsNone(chat_history_pg._pool_dsn)
        self.assertTrue(fake.pools_created[0].closed)


if __name__ == "__main__":
    unittest.main()
