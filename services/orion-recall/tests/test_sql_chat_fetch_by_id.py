"""fetch_chat_turns_by_id: the Postgres text join fetch_falkor_chatturn_fragments
relies on (Falkor's ChatTurn node has no prompt/response text)."""

from __future__ import annotations

import asyncio

from app import sql_chat


def _run(coro):
    return asyncio.run(coro)


def test_empty_turn_ids_short_circuits(monkeypatch) -> None:
    monkeypatch.setattr(sql_chat, "asyncpg", object())  # non-None, but never called
    out = _run(sql_chat.fetch_chat_turns_by_id([]))
    assert out == {}


def test_asyncpg_unavailable_returns_empty(monkeypatch) -> None:
    monkeypatch.setattr(sql_chat, "asyncpg", None)
    out = _run(sql_chat.fetch_chat_turns_by_id(["turn-1"]))
    assert out == {}


def test_connection_failure_degrades_to_empty(monkeypatch) -> None:
    class _RaisingAsyncpg:
        @staticmethod
        async def connect(dsn):
            raise RuntimeError("db down")

    monkeypatch.setattr(sql_chat, "asyncpg", _RaisingAsyncpg())
    out = _run(sql_chat.fetch_chat_turns_by_id(["turn-1"]))
    assert out == {}


def test_returns_prompt_response_client_meta_tuple_by_id(monkeypatch) -> None:
    """client_meta is the third tuple element (added 2026-07-31, see
    app/chat_source_tagging.py) so callers can tell ai-town-sourced turns
    apart from real Juniper/hub conversation instead of quoting every row
    identically."""
    class _FakeConn:
        async def fetch(self, query, ids):
            assert ids == ["turn-1"]
            return [
                {
                    "id": "turn-1",
                    "prompt": "hi Circe",
                    "response": "hello",
                    "client_meta": {"external_room": {"platform": "aitown"}},
                }
            ]

        async def close(self):
            pass

    class _FakeAsyncpg:
        @staticmethod
        async def connect(dsn):
            return _FakeConn()

    monkeypatch.setattr(sql_chat, "asyncpg", _FakeAsyncpg())
    out = _run(sql_chat.fetch_chat_turns_by_id(["turn-1"]))
    assert out == {"turn-1": ("hi Circe", "hello", {"external_room": {"platform": "aitown"}})}


def test_queries_both_tables_separately(monkeypatch) -> None:
    """AI Town chat-history table split (docs/superpowers/specs/2026-08-19-
    aitown-table-split-phase2-recall-migration-design.md): a turn id can
    live in aitown_chat_history_log instead of chat_history_log. Queries
    both tables as two separate calls (not a single UNION ALL) -- see
    fetch_chat_turns_by_id's docstring for why this deliberately does not
    assume an id lives in exactly one table."""
    captured: list = []

    class _FakeConn:
        async def fetch(self, query, ids):
            captured.append(query)
            return []

        async def close(self):
            pass

    class _FakeAsyncpg:
        @staticmethod
        async def connect(dsn):
            return _FakeConn()

    monkeypatch.setattr(sql_chat, "asyncpg", _FakeAsyncpg())
    _run(sql_chat.fetch_chat_turns_by_id(["turn-1"]))
    assert len(captured) == 2
    assert "chat_history_log" in captured[0] and "aitown_chat_history_log" not in captured[0]
    assert "aitown_chat_history_log" in captured[1]
    assert "union all" not in captured[0].lower() and "union all" not in captured[1].lower()


def test_mirror_table_wins_on_id_conflict(monkeypatch) -> None:
    """The load-bearing guarantee this two-query-merge exists for: if the
    same id somehow exists in both tables (a real possibility while
    orion-sql-writer's Phase 1 dual-write is the live write path), the
    mirror table's value must deterministically win, not whichever table
    a plain UNION ALL happened to return last."""

    class _FakeConn:
        def __init__(self):
            self._call = 0

        async def fetch(self, query, ids):
            self._call += 1
            if self._call == 1:
                return [{"id": "turn-1", "prompt": "primary version", "response": "old", "client_meta": None}]
            return [{"id": "turn-1", "prompt": "mirror version", "response": "new", "client_meta": {"external_room": {"platform": "aitown"}}}]

        async def close(self):
            pass

    class _FakeAsyncpg:
        @staticmethod
        async def connect(dsn):
            return _FakeConn()

    monkeypatch.setattr(sql_chat, "asyncpg", _FakeAsyncpg())
    out = _run(sql_chat.fetch_chat_turns_by_id(["turn-1"]))
    assert out["turn-1"][0] == "mirror version"


def test_mirror_query_failure_preserves_primary_results(monkeypatch) -> None:
    """Regression (code review, 2026-08-19): the two queries used to share
    one try/except around both calls, so a mirror-table failure (missing
    table, permission error, transient fault) discarded an
    already-successful primary_rows result too, silently emptying the whole
    response instead of degrading to primary-only."""

    class _FakeConn:
        def __init__(self):
            self._call = 0

        async def fetch(self, query, ids):
            self._call += 1
            if self._call == 1:
                return [{"id": "turn-1", "prompt": "primary version", "response": "ok", "client_meta": None}]
            raise RuntimeError("aitown_chat_history_log unavailable")

        async def close(self):
            pass

    class _FakeAsyncpg:
        @staticmethod
        async def connect(dsn):
            return _FakeConn()

    monkeypatch.setattr(sql_chat, "asyncpg", _FakeAsyncpg())
    out = _run(sql_chat.fetch_chat_turns_by_id(["turn-1"]))
    assert out == {"turn-1": ("primary version", "ok", None)}


def test_primary_query_failure_still_returns_mirror_results(monkeypatch) -> None:
    """Symmetric case: a primary-table failure must not discard an
    already-successful mirror result either."""

    class _FakeConn:
        def __init__(self):
            self._call = 0

        async def fetch(self, query, ids):
            self._call += 1
            if self._call == 1:
                raise RuntimeError("chat_history_log unavailable")
            return [{"id": "turn-1", "prompt": "mirror version", "response": "ok", "client_meta": None}]

        async def close(self):
            pass

    class _FakeAsyncpg:
        @staticmethod
        async def connect(dsn):
            return _FakeConn()

    monkeypatch.setattr(sql_chat, "asyncpg", _FakeAsyncpg())
    out = _run(sql_chat.fetch_chat_turns_by_id(["turn-1"]))
    assert out == {"turn-1": ("mirror version", "ok", None)}


def test_timestamps_mirror_query_failure_preserves_primary_results(monkeypatch) -> None:
    """Same regression, same fix, for fetch_chat_turn_timestamps."""

    class _FakeConn:
        def __init__(self):
            self._call = 0

        async def fetch(self, query, ids):
            self._call += 1
            if self._call == 1:
                return [{"id": "turn-1", "created_at": "2026-08-19T00:00:00+00:00"}]
            raise RuntimeError("aitown_chat_history_log unavailable")

        async def close(self):
            pass

    class _FakeAsyncpg:
        @staticmethod
        async def connect(dsn):
            return _FakeConn()

    monkeypatch.setattr(sql_chat, "asyncpg", _FakeAsyncpg())
    out = _run(sql_chat.fetch_chat_turn_timestamps(["turn-1"], 60))
    assert "turn-1" in out


def test_connection_close_failure_still_returns_already_fetched_results(monkeypatch) -> None:
    """Regression (code review, 2026-08-19): the finally: await conn.close()
    block had no enclosing except, so a close() failure (asyncpg's own
    Connection.close() can raise on a failed graceful close) propagated
    uncaught AFTER both queries already succeeded, discarding the results
    this whole fix exists to preserve. close() failures must be swallowed
    (best-effort), not left to blow away already-built results."""

    class _FakeConn:
        async def fetch(self, query, ids):
            return [{"id": "turn-1", "prompt": "hi", "response": "ok", "client_meta": None}]

        async def close(self):
            raise RuntimeError("connection reset by peer")

    class _FakeAsyncpg:
        @staticmethod
        async def connect(dsn):
            return _FakeConn()

    monkeypatch.setattr(sql_chat, "asyncpg", _FakeAsyncpg())
    out = _run(sql_chat.fetch_chat_turns_by_id(["turn-1"]))
    assert out == {"turn-1": ("hi", "ok", None)}


def test_timestamps_queries_both_tables_separately(monkeypatch) -> None:
    captured: list = []

    class _FakeConn:
        async def fetch(self, query, ids):
            captured.append(query)
            return []

        async def close(self):
            pass

    class _FakeAsyncpg:
        @staticmethod
        async def connect(dsn):
            return _FakeConn()

    monkeypatch.setattr(sql_chat, "asyncpg", _FakeAsyncpg())
    _run(sql_chat.fetch_chat_turn_timestamps(["turn-1"], 60))
    assert len(captured) == 2
    assert "aitown_chat_history_log" in captured[1]
    assert "union all" not in captured[0].lower() and "union all" not in captured[1].lower()
