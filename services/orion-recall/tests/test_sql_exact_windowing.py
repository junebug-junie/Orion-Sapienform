"""SQL anchor exact-match recall must honor the profile time window.

fetch_exact_fragments had no temporal filter, so anchor rails (sql_timeline_anchor)
could surface months-old chat turns into journal/metacog recall when query expansion
tokens matched generic words (connection, coding, etc.). RDF chat-turn windowing was
fixed in PR #815; this closes the parallel SQL leak.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from app.sql_timeline import _epoch, fetch_exact_fragments


def test_epoch_parses_datetime_objects() -> None:
    ts = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert _epoch(ts) == ts.timestamp()


def test_epoch_parses_iso_strings() -> None:
    assert _epoch("2026-07-01T12:00:00+00:00") > 1_700_000_000


def test_fetch_exact_fragments_passes_since_minutes_to_query(monkeypatch) -> None:
    captured: dict = {}

    class _FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, sql, params):
            captured["sql"] = sql
            captured["params"] = params

        @property
        def description(self):
            return []

        def fetchall(self):
            return []

    class _FakeConn:
        def cursor(self):
            return _FakeCursor()

        def close(self):
            pass

    monkeypatch.setattr("app.sql_timeline._connect", lambda: _FakeConn())
    monkeypatch.setattr("app.sql_timeline.settings.RECALL_SQL_TIMELINE_TABLE", "chat_history_log")
    monkeypatch.setattr("app.sql_timeline.settings.RECALL_SQL_CHAT_TABLE", "chat_history_log")
    monkeypatch.setattr("app.sql_timeline._pick_id_col", lambda *_a, **_k: "id")
    monkeypatch.setattr("app.sql_timeline._pick_session_col", lambda *_a, **_k: None)
    monkeypatch.setattr("app.sql_timeline._memory_filter_clause", lambda *_a, **_k: "")

    asyncio.run(
        fetch_exact_fragments(
            tokens=["connection"],
            session_id=None,
            node_id=None,
            limit=5,
            since_minutes=1440,
        )
    )

    assert "INTERVAL" in captured["sql"]
    assert 1440 in captured["params"]
