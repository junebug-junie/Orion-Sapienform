"""Startup schema step takes no ACCESS EXCLUSIVE lock once the schema is current (2026-10-31).

ALTER TABLE ... ADD COLUMN IF NOT EXISTS locks the table exclusively even when the column exists;
live, a 27-minute pg_dump held this service's startup at the first ALTER for 11+ minutes.
"""
from app.storage import repository


class FakeCursor:
    """Answers the information_schema and pg_indexes reads; records every statement and its params."""

    def __init__(self, present, indexes=None):
        self.present = present
        self.indexes = [(name,) for name, _ in repository.INDEXES] if indexes is None else indexes
        self.sql = []
        self.params = []
        self._last = ""

    def execute(self, sql, params=None):
        self.sql.append(sql)
        self.params.append(params)
        self._last = sql

    def fetchall(self):
        return list(self.indexes) if "pg_indexes" in self._last else list(self.present)

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


def _run(monkeypatch, present, indexes=None):
    cur = FakeCursor(present, indexes)
    monkeypatch.setattr(repository, "pg_conn", lambda: FakeConn(cur))
    repository.ensure_tables()
    return cur


def _ddl(cur):
    return [s for s in cur.sql if s.startswith(("ALTER TABLE", "CREATE INDEX", "SET LOCAL"))]


EVERY = [(table, column) for table, column, _ in repository.ADDED_COLUMNS]


def test_a_current_schema_runs_no_lock_taking_ddl(monkeypatch):
    cur = _run(monkeypatch, EVERY)
    assert _ddl(cur) == []
    # the reads are scoped to this schema and pass a list (psycopg2 adapts it to ARRAY for ANY)
    reads = [(sql, params) for sql, params in zip(cur.sql, cur.params) if "ANY(%s)" in sql]
    assert len(reads) == 2 and all("current_schema()" in sql and isinstance(p[0], list) for sql, p in reads)


def test_a_missing_column_is_added_under_a_lock_timeout(monkeypatch):
    present = [(t, c) for t, c in EVERY if c != "stage"]
    assert _ddl(_run(monkeypatch, present)) == [
        f"SET LOCAL lock_timeout = '{repository.SCHEMA_LOCK_TIMEOUT}'",
        "ALTER TABLE topic_foundry_runs ADD COLUMN IF NOT EXISTS stage VARCHAR"]


def test_a_missing_index_is_created_under_a_lock_timeout(monkeypatch):
    indexes = [(name,) for name, _ in repository.INDEXES if name != "ix_topic_foundry_runs_spec_hash"]
    assert _ddl(_run(monkeypatch, EVERY, indexes)) == [
        f"SET LOCAL lock_timeout = '{repository.SCHEMA_LOCK_TIMEOUT}'",
        repository.INDEXES[0][1]]
