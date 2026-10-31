"""Startup schema step takes no ACCESS EXCLUSIVE lock once the schema is current (2026-10-31).

ALTER TABLE ... ADD COLUMN IF NOT EXISTS locks the table exclusively even when the column exists;
live, a 27-minute pg_dump held this service's startup at the first ALTER for 11+ minutes.
"""
from app.storage import repository


class FakeCursor:
    def __init__(self, present):
        self.present = present
        self.sql = []

    def execute(self, sql, params=None):
        self.sql.append(sql)

    def fetchall(self):
        return list(self.present)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeConn:
    def __init__(self, cur):
        self.cur = cur

    def cursor(self):
        return self.cur

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _run(monkeypatch, present):
    cur = FakeCursor(present)
    monkeypatch.setattr(repository, "pg_conn", lambda: FakeConn(cur))
    repository.ensure_tables()
    return [s for s in cur.sql if s.startswith("ALTER TABLE")]


def test_a_current_schema_runs_no_alter(monkeypatch):
    every = [(table, column) for table, column, _ in repository.ADDED_COLUMNS]
    assert _run(monkeypatch, every) == []


def test_a_missing_column_is_still_added(monkeypatch):
    present = [(t, c) for t, c, _ in repository.ADDED_COLUMNS if c != "stage"]
    assert _run(monkeypatch, present) == [
        "ALTER TABLE topic_foundry_runs ADD COLUMN IF NOT EXISTS stage VARCHAR"]
