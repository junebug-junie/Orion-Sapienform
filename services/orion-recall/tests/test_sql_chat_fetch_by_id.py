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


def test_query_unions_the_aitown_table(monkeypatch) -> None:
    """AI Town chat-history table split (docs/superpowers/specs/2026-08-19-
    aitown-table-split-phase2-recall-migration-design.md): a turn id can
    live in aitown_chat_history_log instead of chat_history_log (orion-
    sql-writer routes each row to exactly one table). Without unioning it
    in, any AI-Town-originated turn id would silently resolve to nothing
    here -- the exact failure mode the design doc's audit flagged."""
    captured: dict = {}

    class _FakeConn:
        async def fetch(self, query, ids):
            captured["query"] = query
            return []

        async def close(self):
            pass

    class _FakeAsyncpg:
        @staticmethod
        async def connect(dsn):
            return _FakeConn()

    monkeypatch.setattr(sql_chat, "asyncpg", _FakeAsyncpg())
    _run(sql_chat.fetch_chat_turns_by_id(["turn-1"]))
    assert "aitown_chat_history_log" in captured["query"]
    assert "union all" in captured["query"].lower()


def test_timestamps_query_unions_the_aitown_table(monkeypatch) -> None:
    captured: dict = {}

    class _FakeConn:
        async def fetch(self, query, ids):
            captured["query"] = query
            return []

        async def close(self):
            pass

    class _FakeAsyncpg:
        @staticmethod
        async def connect(dsn):
            return _FakeConn()

    monkeypatch.setattr(sql_chat, "asyncpg", _FakeAsyncpg())
    _run(sql_chat.fetch_chat_turn_timestamps(["turn-1"], 60))
    assert "aitown_chat_history_log" in captured["query"]
    assert "union all" in captured["query"].lower()
