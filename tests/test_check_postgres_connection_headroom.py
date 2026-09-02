"""Tests for the Postgres connection-headroom gate.

Numbers come from the real 2026-08-31 incident and are hand-computed, not taken
from the code under test: max_connections 100, superuser_reserved_connections 3,
every client backend connecting as `postgres` (a superuser), 217
`sorry, too many clients already` in the server log over five days and zero
`remaining connection slots are reserved ...`.
"""

from __future__ import annotations

import os
import subprocess
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


# --- escalation: an alarm nobody is told about is not a gate -----------------------
#
# The gate has exited non-zero since PR #2010, but nothing was scheduled to run it
# and it had no way to reach a human. These cover the --notify path added when it
# was finally put on cron: the debounce, and specifically the rule that a card
# which FAILED to send must not be treated as delivered.


class _NotifyAccepted:
    def __init__(self, ok: bool) -> None:
        self.ok = ok


def _notify_stub_module(recorder, *, ok=True):
    """Stand in for orion.notify.client, which notify_alarm imports lazily."""
    module = types.ModuleType("orion.notify.client")

    class _StubClient:
        def __init__(self, **kwargs):
            recorder.setdefault("init", []).append(kwargs)

        def attention_request(self, **kwargs):
            recorder.setdefault("calls", []).append(kwargs)
            return _NotifyAccepted(ok)

    module.NotifyClient = _StubClient
    return module


def _alarming_conn():
    """280 of 300 slots used -> 6.7% free, under the 15% default threshold."""
    return _FakeConn([(300, 3, 280, 280)])


def _healthy_conn():
    return _FakeConn([(300, 3, 10, 10)])


def _run(monkeypatch, conn, recorder, state_file, *, extra=(), ok=True, gate=True):
    monkeypatch.setitem(sys.modules, "psycopg2", _fake_psycopg2(lambda *a, **k: conn))
    monkeypatch.setitem(
        sys.modules, "orion.notify.client", _notify_stub_module(recorder, ok=ok)
    )
    argv = ["--dsn", "postgresql://x/y", "--state-file", str(state_file), *extra]
    if gate:
        argv.insert(2, "--gate")
    return main(argv)


def test_no_card_fires_without_the_notify_flag(monkeypatch, tmp_path):
    """Escalation is opt-in; the bare gate must stay usable by hand and in CI."""
    recorder = {}
    assert (
        _run(monkeypatch, _alarming_conn(), recorder, tmp_path / "s.json") == EXIT_ALARM
    )
    assert recorder.get("calls", []) == []


def test_a_low_headroom_alarm_raises_one_card(monkeypatch, tmp_path):
    recorder = {}
    rc = _run(
        monkeypatch, _alarming_conn(), recorder, tmp_path / "s.json", extra=["--notify"]
    )
    assert rc == EXIT_ALARM
    assert len(recorder["calls"]) == 1
    call = recorder["calls"][0]
    assert call["severity"] == "warning"
    assert "headroom low" in call["message"]
    # The reserve hazard is named when every backend is a superuser (280 of 280).
    # Assert the CLAUSE, not the word "superuser": headroom.summary() always
    # emits "superuser clients={n}", so a substring check on "superuser" passes
    # even with the whole conditional clause deleted.
    assert "will not hold a door open" in call["message"]


def test_a_second_tick_while_still_alarming_does_not_re_card(monkeypatch, tmp_path):
    """A human acks these cards. Re-firing every 10 minutes is noise, not signal."""
    recorder = {}
    state = tmp_path / "s.json"
    _run(monkeypatch, _alarming_conn(), recorder, state, extra=["--notify"])
    _run(monkeypatch, _alarming_conn(), recorder, state, extra=["--notify"])
    assert len(recorder["calls"]) == 1


def test_a_card_that_failed_to_send_is_retried_on_the_next_tick(monkeypatch, tmp_path):
    """The rule that is easy to get wrong.

    NotifyClient does not raise when orion-notify is down -- it returns ok=False.
    If state recorded "notified" on the attempt rather than the confirmation, an
    alarm that first fired during a notify outage would be debounced into
    permanent silence, and no card would ever land for that episode.
    """
    recorder = {}
    state = tmp_path / "s.json"
    _run(monkeypatch, _alarming_conn(), recorder, state, extra=["--notify"], ok=False)
    assert len(recorder["calls"]) == 1
    _run(monkeypatch, _alarming_conn(), recorder, state, extra=["--notify"], ok=True)
    assert len(recorder["calls"]) == 2, "a failed send must be retried, not swallowed"
    # and once it lands, the debounce takes over
    _run(monkeypatch, _alarming_conn(), recorder, state, extra=["--notify"], ok=True)
    assert len(recorder["calls"]) == 2


def test_recovery_clears_the_episode_so_the_next_alarm_cards_again(
    monkeypatch, tmp_path
):
    recorder = {}
    state = tmp_path / "s.json"
    _run(monkeypatch, _alarming_conn(), recorder, state, extra=["--notify"])
    assert _run(monkeypatch, _healthy_conn(), recorder, state, extra=["--notify"]) == EXIT_OK
    assert len(recorder["calls"]) == 1, "recovery is silent"
    _run(monkeypatch, _alarming_conn(), recorder, state, extra=["--notify"])
    assert len(recorder["calls"]) == 2, "a new episode must be able to card again"


def test_saturation_cards_at_critical_severity(monkeypatch, tmp_path):
    """Refused-because-full is the worst case: the door is already shut."""
    recorder = {}

    def connect(*a, **kw):
        raise _PgError("FATAL:  sorry, too many clients already")

    monkeypatch.setitem(sys.modules, "psycopg2", _fake_psycopg2(connect))
    monkeypatch.setitem(sys.modules, "orion.notify.client", _notify_stub_module(recorder))
    rc = main(
        [
            "--dsn",
            "postgresql://x/y",
            "--gate",
            "--notify",
            "--state-file",
            str(tmp_path / "s.json"),
        ]
    )
    assert rc == EXIT_ALARM
    assert len(recorder["calls"]) == 1
    assert recorder["calls"][0]["severity"] == "critical"


def _saturated_tick(monkeypatch, recorder, state_file, *, ok=True):
    """One tick where this script's own connection attempt lost the race."""

    def connect(*a, **kw):
        raise _PgError("FATAL:  sorry, too many clients already")

    monkeypatch.setitem(sys.modules, "psycopg2", _fake_psycopg2(connect))
    monkeypatch.setitem(
        sys.modules, "orion.notify.client", _notify_stub_module(recorder, ok=ok)
    )
    return main(
        [
            "--dsn",
            "postgresql://x/y",
            "--gate",
            "--notify",
            "--state-file",
            str(state_file),
        ]
    )


def test_a_flapping_incident_does_not_card_every_tick(monkeypatch, tmp_path):
    """`saturated` and `headroom_low` are one incident, not two.

    At the wall, whether a given tick gets a slot is close to a coin flip, so the
    two readings alternate. Keying the debounce on `reason` made that fire a card
    EVERY tick, indefinitely, during exactly the incident this exists for. The
    episode is keyed on severity rank instead.
    """
    recorder = {}
    state = tmp_path / "s.json"
    for _ in range(4):
        _saturated_tick(monkeypatch, recorder, state)
        _run(monkeypatch, _alarming_conn(), recorder, state, extra=["--notify"])
    severities = [c["severity"] for c in recorder["calls"]]
    assert severities == ["critical"], (
        f"8 ticks of one incident should card once, got {severities}"
    )


def test_escalation_to_critical_still_fires_once(monkeypatch, tmp_path):
    """Silencing the flap must not silence a genuine escalation."""
    recorder = {}
    state = tmp_path / "s.json"
    _run(monkeypatch, _alarming_conn(), recorder, state, extra=["--notify"])
    _saturated_tick(monkeypatch, recorder, state)
    assert [c["severity"] for c in recorder["calls"]] == ["warning", "critical"]
    # ...and does not re-fire once raised
    _saturated_tick(monkeypatch, recorder, state)
    _run(monkeypatch, _alarming_conn(), recorder, state, extra=["--notify"])
    assert len(recorder["calls"]) == 2


def test_notify_without_gate_does_not_wipe_a_live_alarm(monkeypatch, tmp_path):
    """--gate decides the exit code; it must not decide whether the alarm is real.

    While these were fused, a --notify run without --gate fell through to
    clear_alarm() on a still-alarming reading, so the next gated tick raised a
    second card for the same unbroken episode.
    """
    recorder = {}
    state = tmp_path / "s.json"
    _run(monkeypatch, _alarming_conn(), recorder, state, extra=["--notify"])
    assert len(recorder["calls"]) == 1
    rc = _run(
        monkeypatch, _alarming_conn(), recorder, state, extra=["--notify"], gate=False
    )
    assert rc == EXIT_OK, "without --gate an alarm must not change the exit code"
    _run(monkeypatch, _alarming_conn(), recorder, state, extra=["--notify"])
    assert len(recorder["calls"]) == 1, "the ungated tick wiped the episode"


def test_escalation_failure_is_not_reported_as_an_alarm(monkeypatch, tmp_path, capsys):
    """An unwritable state dir must not masquerade as a full database.

    notify_alarm used to let this escape as an unhandled traceback, whose Python
    exit status 1 is indistinguishable from EXIT_ALARM.
    """
    recorder = {}
    unwritable = tmp_path / "nope"
    unwritable.write_text("i am a file, not a directory")
    rc = _run(
        monkeypatch,
        _healthy_conn(),
        recorder,
        unwritable / "s.json",
        extra=["--notify"],
    )
    assert rc == EXIT_OK
    assert recorder.get("calls", []) == []


def test_the_notify_client_is_importable_when_run_as_a_script(tmp_path):
    """Regression: the escalation path shipped dead on arrival.

    Invoked as `python scripts/check_postgres_connection_headroom.py` -- exactly
    how `make postgres-headroom-watch`, and therefore cron, invokes it --
    sys.path[0] is `scripts/`, not the repo root. The lazy
    `from orion.notify.client import NotifyClient` inside notify_alarm() then
    raised ModuleNotFoundError, and every alarm printed "notify unavailable;
    alarm not escalated" instead of raising a card. The gate ran, reported
    correctly, and escalated nothing.

    This has to be a SUBPROCESS from a foreign cwd with PYTHONPATH cleared.
    Every other test in this file passed straight through the bug: this module
    inserts REPO_ROOT into sys.path at import time, and then monkeypatches
    sys.modules["orion.notify.client"] anyway, so the import being stood in for
    could not fail no matter how broken the script's own path setup was.
    """
    script = REPO_ROOT / "scripts" / "check_postgres_connection_headroom.py"
    probe = (
        "import runpy\n"
        # run_name != "__main__" so module-level setup executes but main() does not
        f"runpy.run_path({str(script)!r}, run_name='_probe')\n"
        "import orion.notify.client\n"
        "print('NOTIFY_IMPORT_OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": ""},
    )
    assert "NOTIFY_IMPORT_OK" in result.stdout, (
        "the script's own sys.path setup did not make `orion` importable, so "
        "--notify would report 'notify unavailable' in production\n"
        f"stdout={result.stdout!r}\nstderr={result.stderr[-900:]!r}"
    )
