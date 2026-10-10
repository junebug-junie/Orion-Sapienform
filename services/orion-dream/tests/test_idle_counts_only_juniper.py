"""Regression (Temporal Self rev 4, R2 repair 1): Orion's own outreach must not reset the
dream's idle clock. Live 10-09: 24 of 52 chat_history_log rows in a week were Orion's
promptless outreach, and the old query read max(created_at) over all of them."""
from __future__ import annotations

import os
from urllib.parse import urlparse

import pytest

from orion.regulation.juniper_turns import JUNIPER_IDLE_MINUTES_SQL, JUNIPER_TURN_PREDICATE


def test_dream_idle_query_is_the_shared_juniper_rule():
    from app import cycle_store

    assert cycle_store.IDLE_MINUTES_SQL == JUNIPER_IDLE_MINUTES_SQL
    assert JUNIPER_TURN_PREDICATE in cycle_store.IDLE_MINUTES_SQL


def test_outreach_row_does_not_reset_idle_postgres(monkeypatch):
    uri = os.environ.get("REGULATION_HISTORY_TEST_POSTGRES_URI")
    if not uri:
        pytest.skip("requires isolated regulation_history_test Postgres")
    assert urlparse(uri).path == "/regulation_history_test", "refuse any other database"
    from sqlalchemy import create_engine, text

    from app import cycle_store

    engine = create_engine(uri)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS chat_history_log"))
        # The live columns this query touches (timestamp WITHOUT time zone, defaulted by now()).
        conn.execute(text("""CREATE TABLE chat_history_log (
            id varchar PRIMARY KEY, source varchar, prompt text, response text,
            client_meta jsonb, created_at timestamp DEFAULT now())"""))
        conn.execute(text("""INSERT INTO chat_history_log (id, source, prompt, response, client_meta, created_at) VALUES
            ('juniper', 'hub_orion', 'good night', 'sleep well', '{}', LOCALTIMESTAMP - interval '120 minutes'),
            ('button', NULL, 'Run your dream cycle.', 'ok', NULL, LOCALTIMESTAMP - interval '180 minutes'),
            ('outreach', NULL, NULL, 'I have been thinking...', '{"unsolicited": true}', LOCALTIMESTAMP - interval '2 minutes'),
            ('blank', 'hub_orion', '   ', 'x', NULL, LOCALTIMESTAMP - interval '1 minutes'),
            ('marked', 'hub_orion', 'x', 'y', '{"unsolicited": "true"}', LOCALTIMESTAMP - interval '1 minutes')"""))
    monkeypatch.setattr(cycle_store, "_engine", engine)
    try:
        idle = cycle_store.load_idle_minutes()
        assert idle is not None and 119.0 < idle < 122.0, idle
        # The old query would have read the outreach row: about 1 minute, never idle.
        with engine.connect() as conn:
            old = conn.execute(text(
                "SELECT EXTRACT(EPOCH FROM (LOCALTIMESTAMP - max(created_at))) / 60.0 FROM chat_history_log")).scalar()
        assert float(old) < 3.0
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM chat_history_log WHERE prompt IS NOT NULL"))
        assert cycle_store.load_idle_minutes() is None   # outreach only: unknown, as an empty log was
    finally:
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS chat_history_log"))
        engine.dispose()
