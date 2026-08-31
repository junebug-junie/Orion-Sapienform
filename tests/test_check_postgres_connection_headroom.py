"""Tests for the Postgres connection-headroom gate.

Numbers come from the real 2026-08-31 incident and are hand-computed, not taken
from the code under test: max_connections 100, superuser_reserved_connections 3,
every client backend connecting as `postgres` (a superuser), 217
`sorry, too many clients already` in the server log over five days and zero
`remaining connection slots are reserved ...`.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.check_postgres_connection_headroom import (  # noqa: E402
    EXIT_ALARM,
    EXIT_CANNOT_CHECK,
    EXIT_OK,
    Headroom,
    connection_params,
    is_saturation_error,
    main,
    read_headroom,
)


# --- which ceiling actually applies ------------------------------------------------


def test_a_superuser_client_is_refused_at_max_connections_not_at_the_reserve():
    """The reserve does not hold slots back from superusers.

    Every client backend on this deployment is `postgres`. The log proves the wall
    is 100: 217 refusals, all "too many clients", none of the non-superuser form.
    """
    h = Headroom(max_connections=100, reserved_for_superuser=3, used=98, superuser_used=98)
    assert h.free == 2


def test_the_lower_ceiling_for_ordinary_roles_is_still_reported():
    h = Headroom(max_connections=100, reserved_for_superuser=3, used=50, superuser_used=50)
    assert h.nonsuperuser_ceiling == 97


def test_the_incident_reading_reports_no_headroom():
    h = Headroom(max_connections=100, reserved_for_superuser=3, used=100, superuser_used=100)
    assert h.free == 0
    assert h.free_pct == 0.0


def test_free_is_measured_against_max_connections_not_the_reserve():
    # 100 - 97 = 3 free. Measuring against 97 would call this full and cry wolf.
    h = Headroom(max_connections=100, reserved_for_superuser=3, used=97, superuser_used=97)
    assert h.free == 3
    assert h.free_pct == pytest.approx(3.0)


def test_the_raised_ceiling_gives_200_more_slots_on_the_same_load():
    h = Headroom(max_connections=300, reserved_for_superuser=3, used=97, superuser_used=97)
    assert h.free == 203


def test_free_never_goes_negative():
    h = Headroom(max_connections=100, reserved_for_superuser=3, used=101, superuser_used=101)
    assert h.free == 0


def test_the_clamp_never_hides_the_raw_count_it_clamped():
    h = Headroom(max_connections=100, reserved_for_superuser=3, used=101, superuser_used=101)
    assert "101" in h.summary()


# --- the reserve being decorative is itself the finding ----------------------------


def test_an_all_superuser_deployment_has_no_emergency_door():
    h = Headroom(max_connections=100, reserved_for_superuser=3, used=94, superuser_used=94)
    assert h.reserve_is_decorative


def test_a_deployment_with_ordinary_roles_still_has_its_reserve():
    h = Headroom(max_connections=100, reserved_for_superuser=3, used=94, superuser_used=2)
    assert not h.reserve_is_decorative


def test_an_idle_server_is_not_reported_as_having_lost_its_reserve():
    h = Headroom(max_connections=100, reserved_for_superuser=3, used=0, superuser_used=0)
    assert not h.reserve_is_decorative


def test_the_warning_names_the_hazard_when_the_reserve_is_spent(monkeypatch, capsys):
    conn = _FakeConn([(100, 3, 94, 94)])
    monkeypatch.setitem(sys.modules, "psycopg2", _fake_psycopg2(lambda *a, **k: conn))
    main(["--dsn", "postgresql://x/y"])
    assert "no emergency door" in capsys.readouterr().out


# --- saturation detection ----------------------------------------------------------


class _PgError(Exception):
    def __init__(self, message, pgcode=None):
        super().__init__(message)
        self.pgcode = pgcode


def test_saturation_is_detected_by_sqlstate():
    assert is_saturation_error(_PgError("irgendein fehler", pgcode="53300"))


def test_the_superuser_refusal_message_is_detected_without_a_pgcode():
    assert is_saturation_error(_PgError("FATAL:  sorry, too many clients already"))


def test_the_ordinary_role_refusal_message_is_also_detected():
    """The message a non-superuser gets at max_connections - reserved.

    It has never appeared in this deployment's log, and will not until a service is
    moved off the `postgres` superuser -- which is the fix for the missing emergency
    door. This branch must already work when that happens.
    """
    assert is_saturation_error(
        _PgError(
            "FATAL:  remaining connection slots are reserved for "
            "non-replication superuser connections"
        )
    )


def test_an_unrelated_connection_error_is_not_mistaken_for_saturation():
    assert not is_saturation_error(_PgError("password authentication failed"))
    assert not is_saturation_error(_PgError("could not translate host name"))


def test_a_different_sqlstate_is_not_saturation():
    assert not is_saturation_error(_PgError("boom", pgcode="28P01"))


# --- fakes -------------------------------------------------------------------------


def _fake_psycopg2(connect):
    module = types.ModuleType("psycopg2")
    module.connect = connect
    return module


class _FakeCursor:
    def __init__(self, rows, executed):
        self._rows = rows
        self._row = None
        self.executed = executed

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

    def set_session(self, **kwargs):
        self.session_kwargs = kwargs

    def close(self):
        self.closed = True


# --- exit codes: an alarm must be distinguishable from a broken check --------------


def test_saturation_is_reported_as_an_alarm_not_a_crash(monkeypatch, capsys):
    """A gate that dies on saturation goes quiet exactly when it should shout."""

    def connect(*a, **kw):
        raise _PgError("FATAL:  sorry, too many clients already")

    monkeypatch.setitem(sys.modules, "psycopg2", _fake_psycopg2(connect))
    assert main(["--dsn", "postgresql://x/y", "--gate"]) == EXIT_ALARM
    assert "SATURATED" in capsys.readouterr().err


def test_an_unrelated_connection_failure_is_not_laundered_into_an_alarm(
    monkeypatch, capsys
):
    """A wrong password must not page someone about a full database."""

    def connect(*a, **kw):
        raise _PgError("password authentication failed for user")

    monkeypatch.setitem(sys.modules, "psycopg2", _fake_psycopg2(connect))
    assert main(["--dsn", "postgresql://x/y", "--gate"]) == EXIT_CANNOT_CHECK
    assert "SATURATED" not in capsys.readouterr().err


def test_with_nothing_configured_it_targets_the_published_host_port(monkeypatch):
    """The root .env's POSTGRES_URI is a docker-internal hostname that does not
    resolve from the host, so a bare invocation must not default to it."""
    for key in ("POSTGRES_URI", "DATABASE_URL", "ORION_PG_HOST", "ORION_PG_PORT"):
        monkeypatch.delenv(key, raising=False)
    params = connection_params()
    assert params["host"] == "localhost"
    assert params["port"] == 55432


def test_a_missing_driver_cannot_be_confused_with_a_full_database(monkeypatch):
    monkeypatch.setitem(sys.modules, "psycopg2", None)

    def boom(name, *a, **k):
        if name == "psycopg2":
            raise ImportError("no psycopg2")
        return _real_import(name, *a, **k)

    import builtins

    _real_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", boom)
    assert main(["--dsn", "postgresql://x/y", "--gate"]) == EXIT_CANNOT_CHECK


def test_the_gate_fails_when_headroom_is_below_the_threshold(monkeypatch, capsys):
    conn = _FakeConn([(100, 3, 97, 97)])
    monkeypatch.setitem(sys.modules, "psycopg2", _fake_psycopg2(lambda *a, **k: conn))
    assert main(["--dsn", "postgresql://x/y", "--gate", "--min-free-pct", "15"]) == EXIT_ALARM
    assert "FAIL" in capsys.readouterr().err
    assert conn.closed


def test_the_gate_passes_with_the_raised_ceiling_on_the_same_load(monkeypatch):
    conn = _FakeConn([(300, 3, 97, 97)])
    monkeypatch.setitem(sys.modules, "psycopg2", _fake_psycopg2(lambda *a, **k: conn))
    assert main(["--dsn", "postgresql://x/y", "--gate", "--min-free-pct", "15"]) == EXIT_OK


def test_reporting_mode_without_the_gate_flag_does_not_alarm(monkeypatch):
    """Deliberate: --verbose is for humans reading output, --gate is for cron.

    Anything scheduled MUST pass --gate or it cannot fire. The compose comment and
    the Makefile target both pass it.
    """
    conn = _FakeConn([(100, 3, 100, 100)])
    monkeypatch.setitem(sys.modules, "psycopg2", _fake_psycopg2(lambda *a, **k: conn))
    assert main(["--dsn", "postgresql://x/y"]) == EXIT_OK


def test_the_connection_is_closed_even_when_the_query_raises(monkeypatch):
    conn = _FakeConn([])
    monkeypatch.setitem(sys.modules, "psycopg2", _fake_psycopg2(lambda *a, **k: conn))
    with pytest.raises(IndexError):
        main(["--dsn", "postgresql://x/y"])
    assert conn.closed


# --- DSN resolution ---------------------------------------------------------------


def test_an_explicit_dsn_wins_over_the_environment(monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://from-env/db")
    assert connection_params("postgresql://explicit/db") == "postgresql://explicit/db"


def test_database_url_is_the_fallback_when_postgres_uri_is_unset(monkeypatch):
    monkeypatch.delenv("POSTGRES_URI", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://fallback/db")
    assert connection_params() == "postgresql://fallback/db"


# --- background processes are not connections --------------------------------------
#
# This is the bug that made an earlier version print "102/97 used" on a server whose
# max_connections is 100. A unit test cannot catch it: asserting the SQL *contains*
# "backend_type = 'client backend'" is defeated by appending "OR 1=1", which is
# exactly the mutation that matters. Only a real server can tell the difference, so
# this is a live test that skips when there is no database.

_LIVE_DSN = os.environ.get("ORION_TEST_POSTGRES_URI") or os.environ.get("POSTGRES_URI")


@pytest.mark.skipif(not _LIVE_DSN, reason="no live Postgres (set ORION_TEST_POSTGRES_URI)")
def test_live_background_processes_are_excluded_from_the_count():
    psycopg2 = pytest.importorskip("psycopg2")
    conn = psycopg2.connect(_LIVE_DSN, connect_timeout=10)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM pg_stat_activity")
            everything = int(cur.fetchone()[0])
            cur.execute(
                "SELECT count(*) FROM pg_stat_activity "
                "WHERE backend_type <> 'client backend'"
            )
            background = int(cur.fetchone()[0])
        measured = read_headroom(conn).used
    finally:
        conn.close()

    # A live server always runs a checkpointer, walwriter, background writer and at
    # least one launcher, so this is never a vacuous comparison.
    assert background > 0
    assert measured == everything - background
    assert measured < everything


@pytest.mark.skipif(not _LIVE_DSN, reason="no live Postgres (set ORION_TEST_POSTGRES_URI)")
def test_live_the_superuser_count_never_exceeds_the_total():
    psycopg2 = pytest.importorskip("psycopg2")
    conn = psycopg2.connect(_LIVE_DSN, connect_timeout=10)
    try:
        h = read_headroom(conn)
    finally:
        conn.close()
    assert 0 <= h.superuser_used <= h.used
    assert h.max_connections > 0


def test_the_session_is_opened_read_only(monkeypatch):
    """This script runs against production; it must never be able to write."""
    conn = _FakeConn([(100, 3, 94, 94)])
    monkeypatch.setitem(sys.modules, "psycopg2", _fake_psycopg2(lambda *a, **k: conn))
    main(["--dsn", "postgresql://x/y"])
    assert conn.session_kwargs["readonly"] is True
