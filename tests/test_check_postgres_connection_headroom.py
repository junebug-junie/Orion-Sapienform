"""Tests for the Postgres connection-headroom gate.

The numbers in the incident tests are the real 2026-08-31 reading (max_connections
100, superuser_reserved_connections 3, 97 backends in use), hand-computed rather
than taken from the code under test.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.check_postgres_connection_headroom import (  # noqa: E402
    Headroom,
    is_saturation_error,
    main,
    resolve_dsn,
)


# --- the arithmetic that made 97/100 mean "full" ----------------------------------


def test_the_reserved_slots_are_not_available_to_services():
    # 100 - 3 = 97. A service that sees "3 slots left" on max_connections has none.
    h = Headroom(max_connections=100, reserved_for_superuser=3, used=90)
    assert h.service_ceiling == 97
    assert h.free_for_services == 7


def test_the_incident_reading_reports_no_headroom_at_all():
    # The live reading on 2026-08-31, which was refusing connections.
    h = Headroom(max_connections=100, reserved_for_superuser=3, used=97)
    assert h.free_for_services == 0
    assert h.free_pct == 0.0


def test_a_reading_below_the_ceiling_is_not_reported_as_full():
    # Guards against an off-by-one that would cry wolf one slot early.
    h = Headroom(max_connections=100, reserved_for_superuser=3, used=96)
    assert h.free_for_services == 1
    assert h.free_pct == pytest.approx(100 / 97, rel=1e-6)


def test_the_new_ceiling_gives_services_297_slots():
    # POSTGRES_MAX_CONNECTIONS=300 as shipped in services/orion-sql-db/.env_example.
    h = Headroom(max_connections=300, reserved_for_superuser=3, used=97)
    assert h.service_ceiling == 297
    assert h.free_for_services == 200


def test_superusers_eating_reserved_slots_cannot_produce_negative_free():
    h = Headroom(max_connections=100, reserved_for_superuser=3, used=99)
    assert h.free_for_services == 0


def test_the_clamp_never_hides_the_raw_count_it_clamped():
    # free_for_services is clamped because negative free slots are meaningless, but
    # the reading that caused the clamp must stay visible or the instrument lies.
    h = Headroom(max_connections=100, reserved_for_superuser=3, used=99)
    assert "99" in h.summary()


def test_the_summary_names_the_ceiling_services_hit_not_max_connections():
    h = Headroom(max_connections=100, reserved_for_superuser=3, used=97)
    # "97/97", not "97/100" -- the whole point of the reserved subtraction.
    assert "97/97 used" in h.summary()


# --- saturation detection ----------------------------------------------------------


class _PgError(Exception):
    def __init__(self, message, pgcode=None):
        super().__init__(message)
        self.pgcode = pgcode


def test_saturation_is_detected_by_sqlstate():
    assert is_saturation_error(_PgError("irgendein fehler", pgcode="53300"))


def test_saturation_is_detected_from_the_message_when_pgcode_is_absent():
    # psycopg2 raises a bare OperationalError with no pgcode when the refusal lands
    # during the startup handshake -- the exact case this gate exists for.
    assert is_saturation_error(_PgError("FATAL:  sorry, too many clients already"))


def test_an_unrelated_connection_error_is_not_mistaken_for_saturation():
    assert not is_saturation_error(_PgError("password authentication failed"))
    assert not is_saturation_error(_PgError("could not translate host name"))


def test_a_different_sqlstate_is_not_saturation():
    assert not is_saturation_error(_PgError("boom", pgcode="28P01"))


# --- the behaviour that makes the check survive the thing it checks for ------------


def _fake_psycopg2(connect):
    module = types.ModuleType("psycopg2")
    module.connect = connect
    return module


class _FakeCursor:
    def __init__(self, rows, executed=None):
        self._rows = rows
        self._row = None
        self.executed = executed if executed is not None else []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.executed.append(sql)
        self._row = self._rows.pop(0)

    def fetchone(self):
        return self._row

    def fetchall(self):
        return self._row


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows
        self.closed = False
        self.executed: list[str] = []

    def cursor(self):
        return _FakeCursor(self._rows, self.executed)

    def close(self):
        self.closed = True


def test_the_check_reports_instead_of_crashing_when_the_database_is_full(
    monkeypatch, capsys
):
    """A gate that dies on saturation goes quiet exactly when it should shout."""

    def connect(*a, **kw):
        raise _PgError("FATAL:  sorry, too many clients already")

    monkeypatch.setitem(sys.modules, "psycopg2", _fake_psycopg2(connect))
    rc = main(["--dsn", "postgresql://x/y", "--gate"])
    assert rc == 1
    assert "SATURATED" in capsys.readouterr().err


def test_an_unrelated_connection_failure_is_not_laundered_into_the_saturation_path(
    monkeypatch,
):
    """A wrong password must not be reported as 'the database is full'."""

    def connect(*a, **kw):
        raise _PgError("password authentication failed for user")

    monkeypatch.setitem(sys.modules, "psycopg2", _fake_psycopg2(connect))
    with pytest.raises(_PgError):
        main(["--dsn", "postgresql://x/y"])


def test_the_gate_fails_when_headroom_is_below_the_threshold(monkeypatch, capsys):
    conn = _FakeConn([(100, 3, 97)])
    monkeypatch.setitem(sys.modules, "psycopg2", _fake_psycopg2(lambda *a, **k: conn))
    rc = main(["--dsn", "postgresql://x/y", "--gate", "--min-free-pct", "15"])
    assert rc == 1
    assert "FAIL" in capsys.readouterr().err
    assert conn.closed


def test_the_gate_passes_with_the_raised_ceiling_on_the_same_load(monkeypatch):
    # Same 97 backends that failed above, against max_connections=300.
    conn = _FakeConn([(300, 3, 97)])
    monkeypatch.setitem(sys.modules, "psycopg2", _fake_psycopg2(lambda *a, **k: conn))
    assert main(["--dsn", "postgresql://x/y", "--gate", "--min-free-pct", "15"]) == 0


def test_without_the_gate_flag_a_saturated_reading_still_exits_zero(monkeypatch):
    # Reporting mode must stay usable as a plain read, or nobody will run it.
    conn = _FakeConn([(100, 3, 97)])
    monkeypatch.setitem(sys.modules, "psycopg2", _fake_psycopg2(lambda *a, **k: conn))
    assert main(["--dsn", "postgresql://x/y"]) == 0


def test_the_connection_is_closed_even_when_the_query_raises(monkeypatch):
    conn = _FakeConn([])  # pop from an empty list -> IndexError inside the try
    monkeypatch.setitem(sys.modules, "psycopg2", _fake_psycopg2(lambda *a, **k: conn))
    with pytest.raises(IndexError):
        main(["--dsn", "postgresql://x/y"])
    assert conn.closed


# --- DSN resolution ---------------------------------------------------------------


def test_an_explicit_dsn_wins_over_the_environment(monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://from-env/db")
    assert resolve_dsn("postgresql://explicit/db") == "postgresql://explicit/db"


def test_database_url_is_the_fallback_when_postgres_uri_is_unset(monkeypatch):
    monkeypatch.delenv("POSTGRES_URI", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://fallback/db")
    assert resolve_dsn() == "postgresql://fallback/db"


def test_no_dsn_anywhere_is_a_clear_exit_not_a_traceback(monkeypatch):
    monkeypatch.delenv("POSTGRES_URI", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(SystemExit) as excinfo:
        resolve_dsn()
    assert "no DSN" in str(excinfo.value)


# --- background processes are not connections -------------------------------------


def test_only_client_backends_are_counted_against_max_connections(monkeypatch):
    """pg_stat_activity lists background processes that hold no connection slot.

    Counting every row made this script print "102/97 used" against a server whose
    max_connections is 100: the live server had 91 client backends plus a
    checkpointer, walwriter, background writer, autovacuum launcher and logical
    replication launcher. Those five are budgeted by max_worker_processes and
    friends, not by max_connections.
    """
    conn = _FakeConn([(100, 3, 91)])
    monkeypatch.setitem(sys.modules, "psycopg2", _fake_psycopg2(lambda *a, **k: conn))
    assert main(["--dsn", "postgresql://x/y"]) == 0
    assert any("backend_type = 'client backend'" in sql for sql in conn.executed)


def test_every_pg_stat_activity_read_filters_out_background_processes(monkeypatch):
    """The idle split and the per-client breakdown must agree with the headline."""
    conn = _FakeConn([(100, 3, 91), (94, 8), [("172.18.0.1", 24)]])
    monkeypatch.setitem(sys.modules, "psycopg2", _fake_psycopg2(lambda *a, **k: conn))
    assert main(["--dsn", "postgresql://x/y", "--verbose"]) == 0
    reads = [sql for sql in conn.executed if "pg_stat_activity" in sql]
    assert len(reads) == 3
    for sql in reads:
        assert "backend_type = 'client backend'" in sql
