"""/chat/messages?status=open&tag=... -- the Hub tray's durable Orion-message list.

"Open" must mean "Juniper has not dismissed it". Hub sends a `seen` receipt
automatically when a chat message arrives, so the existing "unread" filter
would empty the tray seconds after each message. These tests compile the real
SQLAlchemy query against the Postgres dialect and check what it asks for.
"""

from __future__ import annotations

import asyncio

from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Query, Session

import app.api_notify as api_notify


def _captured_sql(monkeypatch, **params) -> str:
    captured: dict = {}

    def fake_all(self):
        compiled = self.statement.compile(dialect=postgresql.dialect())
        # Inline the bound values by hand (JSONB has no literal renderer).
        sql = str(compiled)
        for key, value in compiled.params.items():
            sql = sql.replace(f"%({key})s", repr(value))
        captured["sql"] = sql
        return []

    monkeypatch.setattr(Query, "all", fake_all)
    monkeypatch.setattr(api_notify, "get_session", lambda: Session())
    monkeypatch.setattr(api_notify, "remove_session", lambda: None)
    result = asyncio.run(api_notify.list_chat_messages(**params))
    assert result == []
    return captured["sql"]


def test_open_excludes_dismissed_rows_and_dismissed_receipts(monkeypatch) -> None:
    sql = _captured_sql(monkeypatch, limit=20, status="open")
    assert "notify_requests.message_dismissed_at IS NULL" in sql
    assert "NOT (EXISTS" in sql
    assert "notify_receipts.receipt_type = 'dismissed'" in sql
    # A seen/opened receipt must not close a message.
    assert "'seen'" not in sql
    assert "message_first_seen_at" not in sql.split("WHERE", 1)[1]


def test_tag_filter_uses_jsonb_containment(monkeypatch) -> None:
    sql = _captured_sql(monkeypatch, limit=20, status="open", tag="endogenous_outreach")
    assert "notify_requests.tags @>" in sql
    assert "['endogenous_outreach']" in sql


def test_no_status_and_no_tag_leaves_the_old_query_alone(monkeypatch) -> None:
    sql = _captured_sql(monkeypatch, limit=20)
    assert "@>" not in sql
    assert "notify_receipts" not in sql
    assert "notify_requests.message_id IS NOT NULL" in sql
