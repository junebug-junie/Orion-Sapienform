"""Regression: one asyncpg connection must never run two queries at once.

Live 2026-09-29: orion-athena-recall logged 1,075
``InterfaceError: cannot perform operation: another operation is in progress``
in 24h -- every call to fetch_chat_turns_by_id (796) and
fetch_chat_turn_timestamps (279) lost its AI Town mirror-table rows, because
_fetch_primary_and_mirror_rows ran both table queries through
``asyncio.gather`` on the SAME connection. A real asyncpg connection rejects
a second operation while the first is in flight; the earlier fakes in
test_sql_chat_fetch_by_id.py did not, which is why they passed.

The fake below behaves like asyncpg: a second fetch that starts before the
first finishes raises immediately.
"""

from __future__ import annotations

import asyncio

import pytest

from app import sql_chat


class InterfaceError(Exception):
    pass


class _ExclusiveConn:
    """Mimics asyncpg's _stmt_exclusive_section on a single connection."""

    def __init__(self, rows_by_table: dict) -> None:
        self._busy = False
        self._rows_by_table = rows_by_table
        self.queries: list = []

    async def fetch(self, query, ids):
        if self._busy:
            raise InterfaceError("cannot perform operation: another operation is in progress")
        self._busy = True
        try:
            await asyncio.sleep(0.01)  # a real round-trip yields to the loop
            self.queries.append(query)
            for table, rows in self._rows_by_table.items():
                if f"FROM {table}\n" in query:
                    return rows
            return []
        finally:
            self._busy = False

    async def close(self):
        pass


def _install(monkeypatch, conn):
    class _FakeAsyncpg:
        @staticmethod
        async def connect(dsn):
            return conn

    monkeypatch.setattr(sql_chat, "asyncpg", _FakeAsyncpg())


def test_fetch_chat_turns_by_id_returns_mirror_rows_on_one_connection(monkeypatch) -> None:
    primary = sql_chat.settings.RECALL_SQL_CHAT_TABLE
    mirror = sql_chat.settings.RECALL_SQL_AITOWN_CHAT_TABLE
    conn = _ExclusiveConn(
        {
            primary: [{"id": "t-hub", "prompt": "p", "response": "r", "client_meta": None}],
            mirror: [{"id": "t-town", "prompt": "pt", "response": "rt", "client_meta": {"x": 1}}],
        }
    )
    _install(monkeypatch, conn)
    out = asyncio.run(sql_chat.fetch_chat_turns_by_id(["t-hub", "t-town"]))
    assert out == {"t-hub": ("p", "r", None), "t-town": ("pt", "rt", {"x": 1})}
    assert len(conn.queries) == 2


def test_fetch_chat_turn_timestamps_returns_mirror_rows_on_one_connection(monkeypatch) -> None:
    primary = sql_chat.settings.RECALL_SQL_CHAT_TABLE
    mirror = sql_chat.settings.RECALL_SQL_AITOWN_CHAT_TABLE
    conn = _ExclusiveConn(
        {
            primary: [{"id": "shared", "created_at": 100.0}, {"id": "t-hub", "created_at": 50.0}],
            mirror: [{"id": "shared", "created_at": 200.0}],
        }
    )
    _install(monkeypatch, conn)
    out = asyncio.run(sql_chat.fetch_chat_turn_timestamps(["shared", "t-hub"], since_minutes=60))
    # mirror still wins a conflict, and both tables contributed
    assert out == {"shared": 200.0, "t-hub": 50.0}


@pytest.mark.parametrize("failing", ["primary", "mirror"])
def test_one_table_failure_does_not_discard_the_other(monkeypatch, failing) -> None:
    """The 2026-08-19 isolation contract survives serialization: a failure in
    one table still leaves the other table's rows."""
    primary = sql_chat.settings.RECALL_SQL_CHAT_TABLE
    mirror = sql_chat.settings.RECALL_SQL_AITOWN_CHAT_TABLE
    bad = primary if failing == "primary" else mirror

    class _Conn(_ExclusiveConn):
        async def fetch(self, query, ids):
            # Fail INSIDE the busy section (like a server-side error on a
            # real connection), so the other query must run on the same
            # connection after an error on it.
            if self._busy:
                raise InterfaceError("cannot perform operation: another operation is in progress")
            if f"FROM {bad}\n" in query:
                self._busy = True
                try:
                    await asyncio.sleep(0.01)
                    raise RuntimeError("relation does not exist")
                finally:
                    self._busy = False
            return await super().fetch(query, ids)

    conn = _Conn(
        {
            primary: [{"id": "a", "created_at": 1.0}],
            mirror: [{"id": "b", "created_at": 2.0}],
        }
    )
    _install(monkeypatch, conn)
    out = asyncio.run(sql_chat.fetch_chat_turn_timestamps(["a", "b"], since_minutes=60))
    assert out == ({"b": 2.0} if failing == "primary" else {"a": 1.0})
