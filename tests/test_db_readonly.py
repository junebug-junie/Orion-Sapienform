"""Tests for orion/db_readonly.py -- the canonical read-only Postgres helper.

Moved out of scripts/analysis/_pg_readonly.py (see that module's docstring);
this is the first dedicated test coverage either version has had. DB access
is mocked throughout.
"""
from __future__ import annotations

from unittest import mock

from orion.db_readonly import open_readonly_connection


class _FakeCursor:
    def __init__(self, fetchone_result=None):
        self.fetchone_result = fetchone_result
        self.executed: list[tuple] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, params=None):
        self.executed.append((query, params))

    def fetchone(self):
        return self.fetchone_result


class _FakeConn:
    def __init__(self, cursor: _FakeCursor):
        self._cursor = cursor
        self.closed = False

    def cursor(self):
        return self._cursor

    def close(self):
        self.closed = True


def test_returns_none_when_psycopg2_missing():
    import builtins

    real_import = builtins.__import__

    def _fail_import(name, *a, **kw):
        if name == "psycopg2":
            raise ImportError("no psycopg2 here")
        return real_import(name, *a, **kw)

    with mock.patch.object(builtins, "__import__", _fail_import):
        assert open_readonly_connection("postgresql://x") is None


def test_returns_none_on_connect_failure():
    fake_psycopg2 = mock.Mock()
    fake_psycopg2.connect.side_effect = Exception("connection refused")
    with mock.patch.dict("sys.modules", {"psycopg2": fake_psycopg2}):
        assert open_readonly_connection("postgresql://x") is None


def test_closes_and_returns_none_if_not_readonly():
    conn = _FakeConn(_FakeCursor(fetchone_result=("off",)))
    fake_psycopg2 = mock.Mock()
    fake_psycopg2.connect.return_value = conn
    with mock.patch.dict("sys.modules", {"psycopg2": fake_psycopg2}):
        result = open_readonly_connection("postgresql://x")
    assert result is None
    assert conn.closed is True


def test_returns_connection_when_readonly_confirmed():
    conn = _FakeConn(_FakeCursor(fetchone_result=("on",)))
    fake_psycopg2 = mock.Mock()
    fake_psycopg2.connect.return_value = conn
    with mock.patch.dict("sys.modules", {"psycopg2": fake_psycopg2}):
        result = open_readonly_connection("postgresql://x")
    assert result is conn
    fake_psycopg2.connect.assert_called_once_with("postgresql://x")


def test_connect_timeout_omitted_by_default_for_backward_compatibility():
    """Existing callers (measure_goal_provenance_streak_distribution.py via
    the _pg_readonly.py re-export) never passed connect_timeout -- must keep
    getting psycopg2's own default (no timeout), not a new one silently
    injected under them."""
    conn = _FakeConn(_FakeCursor(fetchone_result=("on",)))
    fake_psycopg2 = mock.Mock()
    fake_psycopg2.connect.return_value = conn
    with mock.patch.dict("sys.modules", {"psycopg2": fake_psycopg2}):
        open_readonly_connection("postgresql://x")
    fake_psycopg2.connect.assert_called_once_with("postgresql://x")


def test_connect_timeout_passed_through_when_given():
    conn = _FakeConn(_FakeCursor(fetchone_result=("on",)))
    fake_psycopg2 = mock.Mock()
    fake_psycopg2.connect.return_value = conn
    with mock.patch.dict("sys.modules", {"psycopg2": fake_psycopg2}):
        open_readonly_connection("postgresql://x", connect_timeout=5)
    fake_psycopg2.connect.assert_called_once_with("postgresql://x", connect_timeout=5)


def test_statement_timeout_not_set_by_default():
    """Existing callers (measure_goal_provenance_streak_distribution.py) that
    never passed statement_timeout_ms must keep running unbounded queries --
    no SET statement_timeout should be issued unless explicitly asked for."""
    cur = _FakeCursor(fetchone_result=("on",))
    conn = _FakeConn(cur)
    fake_psycopg2 = mock.Mock()
    fake_psycopg2.connect.return_value = conn
    with mock.patch.dict("sys.modules", {"psycopg2": fake_psycopg2}):
        open_readonly_connection("postgresql://x")
    executed_queries = [q for q, _params in cur.executed]
    assert not any("statement_timeout" in q for q in executed_queries)


def test_statement_timeout_set_when_given():
    cur = _FakeCursor(fetchone_result=("on",))
    conn = _FakeConn(cur)
    fake_psycopg2 = mock.Mock()
    fake_psycopg2.connect.return_value = conn
    with mock.patch.dict("sys.modules", {"psycopg2": fake_psycopg2}):
        open_readonly_connection("postgresql://x", statement_timeout_ms=10_000)
    assert (
        "SET statement_timeout = %s;",
        (10_000,),
    ) in cur.executed
