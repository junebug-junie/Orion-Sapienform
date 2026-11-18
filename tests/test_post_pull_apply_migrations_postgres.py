"""Post-pull migration apply against a REAL Postgres: the migration lands before the rebuild runs.

Runs the real scripts/rebuild_services_from_git_diff.sh, the real applier and the real checker in
a scratch git repo. Only the classifier and safe_docker_build.sh are fakes: the fake build asks
the database, at the moment it is called, whether the migrated table already exists.

Database: POST_PULL_MIGRATIONS_TEST_POSTGRES_URI (CI: a throwaway service container), else a
throwaway cluster started with the local initdb/pg_ctl in tmp_path, else skipped. Never the live
database.
"""
from __future__ import annotations

import glob
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest

psycopg2 = pytest.importorskip("psycopg2")

REPO_ROOT = Path(__file__).resolve().parents[1]
MIG_DIR = "services/orion-sql-db"
HDR = "-- ORION-MIGRATION-REQUIRED-BY: orion-hub\n"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def pg(tmp_path):
    """dict(host, port, user, password, dbname) of a throwaway database."""
    uri = os.environ.get("POST_PULL_MIGRATIONS_TEST_POSTGRES_URI", "").strip()
    if uri:
        u = urlparse(uri)
        yield dict(host=u.hostname or "localhost", port=u.port or 5432, user=unquote(u.username or "postgres"),
                   password=unquote(u.password or ""), dbname=u.path.lstrip("/") or "postgres")
        return
    bins = sorted(glob.glob("/usr/lib/postgresql/*/bin/initdb"))
    if not bins or not shutil.which("psql"):
        if os.environ.get("CI"):
            pytest.fail("CI must provide POST_PULL_MIGRATIONS_TEST_POSTGRES_URI; a skip here hides the only real-DB proof")
        pytest.skip("no POST_PULL_MIGRATIONS_TEST_POSTGRES_URI and no local initdb/psql")
    bindir = Path(bins[-1]).parent
    data, sock = tmp_path / "pgdata", tmp_path / "sock"
    sock.mkdir()
    subprocess.run([str(bindir / "initdb"), "-D", str(data), "-U", "postgres", "-A", "trust"],
                   check=True, capture_output=True)
    port = _free_port()
    subprocess.run([str(bindir / "pg_ctl"), "-D", str(data), "-w", "-l", str(tmp_path / "pg.log"),
                    "-o", f"-k {sock} -p {port} -c listen_addresses=''", "start"],
                   check=True, capture_output=True)
    try:
        yield dict(host=str(sock), port=port, user="postgres", password="", dbname="postgres")
    finally:
        subprocess.run([str(bindir / "pg_ctl"), "-D", str(data), "-m", "immediate", "stop"],
                       capture_output=True)


def _q(pg, sql):
    conn = psycopg2.connect(**pg)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            return cur.fetchall() if cur.description else None
    finally:
        conn.close()


_FAKE_CLASSIFIER = """\
import os, sys
args = sys.argv[1:]
if "--running-profile-services" in args or "--pool-held-services" in args or "--running-extra-compose" in args:
    raise SystemExit(0)
if "--migrations" in args:
    print("\\n".join(x for x in os.environ["FAKE_MIGRATIONS"].split(";") if x))
    raise SystemExit(0)
print("orion-hub\\tbuild")
"""

# At the moment of the rebuild, record which probe tables already exist.
_FAKE_SAFE_BUILD = """#!/bin/sh
SEEN=$(psql -X -At -h "$ORION_PG_HOST" -p "$ORION_PG_PORT" -U "$ORION_PG_USER" -d "$ORION_PG_DB" \\
  -c "SELECT coalesce(string_agg(relname, ',' ORDER BY relname), '-') FROM pg_class WHERE relname LIKE 'pp_probe_%' AND relkind = 'r'")
printf 'BUILD %s tables=%s\\n' "$*" "$SEEN" >> "$REBUILD_TEST_CAPTURE"
"""


@pytest.fixture()
def scratch(tmp_path, pg):
    repo = tmp_path / "repo"
    for rel in ("scripts/rebuild_services_from_git_diff.sh", "scripts/apply_post_pull_migrations.py",
                "scripts/check_sql_migrations_applied.py", "orion/sql_migration_drift.py"):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / rel, repo / rel)
    (repo / "orion" / "__init__.py").write_text("")
    (repo / "scripts" / "rebuild_affected_services.py").write_text(_FAKE_CLASSIFIER)
    sb = repo / "scripts" / "safe_docker_build.sh"
    sb.write_text(_FAKE_SAFE_BUILD)
    sb.chmod(0o755)
    (repo / MIG_DIR).mkdir(parents=True)
    # The primary main checkout synchronized with origin/main: the only place the applier writes.
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True)
    for args in (["init", "-q", "-b", "main"], ["config", "user.email", "t@t"], ["config", "user.name", "t"],
                 ["remote", "add", "origin", str(origin)]):
        subprocess.run(["git", "-C", str(repo), *args], check=True)
    (repo / ".gitignore").write_text("__pycache__/\n")

    shim = tmp_path / "bin"
    shim.mkdir()
    # The interpreter that has psycopg2. A wrapper, not a symlink: a symlink to a venv's python
    # loses the venv (pyvenv.cfg is found next to the invoked path).
    py = shim / "python3"
    py.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    py.chmod(0o755)
    env = os.environ.copy()
    for k in ("ORION_POST_MERGE_REBUILD", "ORION_POST_PULL_APPLY_MIGRATIONS"):
        env.pop(k, None)
    env.update({
        "PATH": f"{shim}:{env['PATH']}",
        "ORION_PG_HOST": pg["host"], "ORION_PG_PORT": str(pg["port"]), "ORION_PG_USER": pg["user"],
        "ORION_PG_PASSWORD": pg["password"], "PGPASSWORD": pg["password"], "ORION_PG_DB": pg["dbname"],
        "ORION_MIGRATION_APPLY_PSQL": f"psql -h {pg['host']} -p {pg['port']} -U {pg['user']} -d {pg['dbname']}",
        "ORION_APPLIED_MIGRATIONS_LOG": str(tmp_path / "applied.log"),
        "ORION_REBUILD_LOG": str(tmp_path / "post_pull_rebuild.log"),
        "REBUILD_TEST_CAPTURE": str(tmp_path / "capture.txt"),
    })

    class S:
        pass

    s = S()
    s.repo, s.env, s.pg, s.log, s.capture = repo, env, pg, tmp_path / "applied.log", tmp_path / "capture.txt"

    def add(name, sql):
        (repo / MIG_DIR / name).write_text(sql)
        subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-qm", name], check=True)
        subprocess.run(["git", "-C", str(repo), "push", "-q", "origin", "main"], check=True)
        return f"{MIG_DIR}/{name}"

    def pull(paths):
        s.env["FAKE_MIGRATIONS"] = ";".join(paths)
        p = subprocess.run([str(repo / "scripts" / "rebuild_services_from_git_diff.sh")], cwd=repo,
                           env=s.env, capture_output=True, text=True, timeout=300)
        builds = s.capture.read_text().splitlines() if s.capture.exists() else []
        return p, builds

    s.add, s.pull = add, pull
    yield s
    _q(pg, "DROP TABLE IF EXISTS pp_probe_a, pp_probe_b, pp_probe_c, pp_probe_keep")


# lock_timeout must reach the session that runs the file, not just the command line: the column
# default is evaluated by the INSERT, inside that session.
_RECORD_LOCK_TIMEOUT = ("CREATE TABLE IF NOT EXISTS pp_probe_a (id int PRIMARY KEY, "
                        "lt text DEFAULT current_setting('lock_timeout'));\n"
                        "INSERT INTO pp_probe_a (id) VALUES (1) ON CONFLICT DO NOTHING;\n")


def test_migrations_land_before_the_rebuild(scratch):
    a = scratch.add("manual_migration_pp_a.sql", HDR + _RECORD_LOCK_TIMEOUT +
                    "ALTER TABLE pp_probe_a ADD COLUMN IF NOT EXISTS note text;\n")
    b = scratch.add("manual_migration_pp_b.sql", HDR +
                    "CREATE TABLE IF NOT EXISTS pp_probe_b (id int);\n"
                    "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_pp_probe_b ON pp_probe_b (id);\n")
    p, builds = scratch.pull([b, a])
    assert p.returncode == 0, p.stdout + p.stderr
    assert builds == ["BUILD orion-hub up -d --build tables=pp_probe_a,pp_probe_b"], p.stdout
    assert "verified: services/orion-sql-db/manual_migration_pp_a.sql is APPLIED" in p.stdout
    assert "no transaction wrapper: CREATE INDEX CONCURRENTLY" in p.stdout
    assert _q(scratch.pg, "SELECT lt FROM pp_probe_a") == [("10s",)]
    assert _q(scratch.pg, "SELECT indisvalid FROM pg_index WHERE indexrelid = 'ix_pp_probe_b'::regclass") == [(True,)]
    log = scratch.log.read_text()
    assert log.index("APPLIED services/orion-sql-db/manual_migration_pp_a.sql") < log.index("manual_migration_pp_b")

    # Second pull of the same files: already applied, nothing re-run.
    scratch.capture.unlink()
    p, builds = scratch.pull([a, b])
    assert "manual_migration_pp_a.sql: APPLIED" in p.stdout
    assert scratch.log.read_text() == log


def test_failed_apply_rolls_back_holds_later_files_and_still_rebuilds(scratch):
    bad = scratch.add("manual_migration_pp_a.sql", HDR +
                      "CREATE TABLE pp_probe_a (id int);\nCREATE INDEX ix_pp_missing ON pp_no_such_table (id);\n")
    later = scratch.add("manual_migration_pp_b.sql", HDR + "CREATE TABLE pp_probe_b (id int);\n")
    p, builds = scratch.pull([bad, later])
    assert p.returncode == 0, p.stdout + p.stderr
    assert builds == ["BUILD orion-hub up -d --build tables=-"], p.stdout
    assert "FAILED" in p.stdout and "HELD, not applied" in p.stdout
    assert "live state: NOT APPLIED" in p.stdout  # the read-only report still names the command
    assert "FAILED services/orion-sql-db/manual_migration_pp_a.sql" in scratch.log.read_text()


def test_destructive_content_is_never_applied(scratch):
    _q(scratch.pg, "CREATE TABLE pp_probe_keep (id int); INSERT INTO pp_probe_keep VALUES (1)")
    f = scratch.add("manual_migration_pp_c.sql", HDR +
                    "CREATE TABLE pp_probe_c (id int);\nDELETE FROM pp_probe_keep;\n")
    p, builds = scratch.pull([f])
    assert "not safe to auto-apply: statement not on the allowlist: DELETE FROM pp_probe_keep" in p.stdout
    assert _q(scratch.pg, "SELECT count(*) FROM pp_probe_keep") == [(1,)]
    assert builds == ["BUILD orion-hub up -d --build tables=pp_probe_keep"]


def test_kill_switch_leaves_the_database_alone(scratch):
    a = scratch.add("manual_migration_pp_a.sql", HDR + "CREATE TABLE pp_probe_a (id int);\n")
    scratch.env["ORION_POST_PULL_APPLY_MIGRATIONS"] = "0"
    p, builds = scratch.pull([a])
    assert "auto-apply OFF" in p.stdout
    assert builds == ["BUILD orion-hub up -d --build tables=-"]


# ------------------------------------------- a running backup (incident 2026-10-17 ~03:56 UTC)
#
# The nightly pg_dump (application_name 'pg_dump') held AccessShareLocks ~45 min and the apply hit
# lock_timeout. Here a session NAMED like a backup holds a real lock on the table the migration
# alters; the real applier, real psql, real pg_stat_activity / pg_blocking_pids decide. The applier
# runs in-process so the poll and backoff can be short; lock_timeout itself stays 10s.

import importlib.util  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402


def _load_applier():
    spec = importlib.util.spec_from_file_location("apply_post_pull_migrations_pg",
                                                  REPO_ROOT / "scripts" / "apply_post_pull_migrations.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


ap = _load_applier()


class _Holder:
    """A session with `appname` holding ACCESS SHARE on pp_probe_keep (what pg_dump takes on every
    table it dumps) for `hold_s` seconds after start(), then disconnecting."""

    def __init__(self, pg, appname, hold_s):
        self.pg, self.appname, self.hold_s = pg, appname, hold_s
        self.locked = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        conn = psycopg2.connect(application_name=self.appname, **self.pg)
        try:
            with conn.cursor() as cur:
                cur.execute("LOCK TABLE pp_probe_keep IN ACCESS SHARE MODE")
                self.locked.set()
                time.sleep(self.hold_s)
            conn.rollback()
        finally:
            conn.close()

    def __enter__(self):
        self.thread.start()
        assert self.locked.wait(10)
        return self

    def __exit__(self, *a):
        self.thread.join(timeout=60)


def _connect(pg):
    def connect():
        conn = psycopg2.connect(**pg)
        conn.set_session(readonly=True, autocommit=True)
        return conn
    return connect


_ADD_COLUMN = HDR + "ALTER TABLE pp_probe_keep ADD COLUMN IF NOT EXISTS note text;\n"


def _has_note(pg):
    return _q(pg, "SELECT count(*) FROM information_schema.columns "
                  "WHERE table_name = 'pp_probe_keep' AND column_name = 'note'") == [(1,)]


def _apply(scratch, paths, **kw):
    lines = []
    kw.setdefault("poll_s", 0.5)
    kw.setdefault("watch_interval_s", 0.5)
    rc = ap.run(paths, repo=scratch.repo, env=scratch.env, connect_fn=_connect(scratch.pg),
                out=lines.append, **kw)
    return rc, "\n".join(lines)


def test_running_pg_dump_is_waited_for_then_the_migration_lands(scratch):
    _q(scratch.pg, "CREATE TABLE pp_probe_keep (id int)")
    f = scratch.add("manual_migration_pp_a.sql", _ADD_COLUMN)
    with _Holder(scratch.pg, "pg_dump", 3):
        t0 = time.monotonic()
        rc, text = _apply(scratch, [f])
        waited = time.monotonic() - t0
    assert rc == 0, text
    assert "waiting for a running backup (pg_dump pid=" in text
    assert "backup finished after a" in text
    assert "lock timeout" not in text   # it never even tried while the backup ran
    assert 2.0 < waited < 9.5, waited   # waited for the holder, not for lock_timeout
    assert _has_note(scratch.pg)
    log = scratch.log.read_text()
    assert log.index("BACKUP-WAIT ") < log.index("BACKUP-WAIT-DONE") < log.index("APPLIED ")


def test_idle_pg_dump_backend_is_not_waited_for(scratch):
    """An orphaned pg_dump backend (2026-08-16..18) sits idle and holds no locks: apply at once."""
    _q(scratch.pg, "CREATE TABLE pp_probe_keep (id int)")
    f = scratch.add("manual_migration_pp_a.sql", _ADD_COLUMN)
    idle = psycopg2.connect(application_name="pg_dump", **scratch.pg)
    idle.autocommit = True
    try:
        assert _q(scratch.pg, "SELECT count(*) FROM pg_stat_activity WHERE application_name = 'pg_dump'") == [(1,)]
        rc, text = _apply(scratch, [f], max_wait_s=0)
    finally:
        idle.close()
    assert rc == 0, text
    assert "waiting for a running backup" not in text
    assert _has_note(scratch.pg)


def test_pg_dump_outlasting_the_max_wait_fails_and_holds(scratch):
    _q(scratch.pg, "CREATE TABLE pp_probe_keep (id int)")
    f = scratch.add("manual_migration_pp_a.sql", _ADD_COLUMN)
    with _Holder(scratch.pg, "pg_dump", 4):
        rc, text = _apply(scratch, [f], max_wait_s=1)
    assert rc == 1, text
    assert "a backup is still running (pg_dump pid=" in text
    assert not _has_note(scratch.pg)
    assert "BACKUP-WAIT-EXCEEDED" in scratch.log.read_text()


def test_lock_timeout_behind_pg_dump_is_retried_then_lands(scratch):
    """The backup starts after the pre-apply look (here: the first look is made to miss it), so the
    apply really hits lock_timeout; the watcher must name pg_dump via pg_blocking_pids."""
    _q(scratch.pg, "CREATE TABLE pp_probe_keep (id int)")
    f = scratch.add("manual_migration_pp_a.sql", _ADD_COLUMN)
    real = _connect(scratch.pg)
    looks = []

    class _MissFirstLook:
        def __init__(self, conn):
            self.conn = conn

        def cursor(self):
            cur = self.conn.cursor()
            outer = self

            class C:
                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    cur.close()
                    return False

                def execute(self, sql, params=None):
                    self.miss = sql == ap.BACKUP_SQL and not looks
                    if sql == ap.BACKUP_SQL:
                        looks.append(1)
                    cur.execute(sql, params)

                def fetchall(self):
                    rows = cur.fetchall()
                    return [] if self.miss else rows
            return C()

        def close(self):
            self.conn.close()

    lines = []
    with _Holder(scratch.pg, "pg_dump", 12):
        rc = ap.run([f], repo=scratch.repo, env=scratch.env, connect_fn=lambda: _MissFirstLook(real()),
                    out=lines.append, poll_s=0.5, watch_interval_s=0.5, backoff_s=(0.5, 0.5, 0.5))
    text = "\n".join(lines)
    assert rc == 0, text
    assert "lock timeout behind a backup (pg_dump pid=" in text and "retry 1/3" in text
    assert _has_note(scratch.pg)
    log = scratch.log.read_text()
    assert "RETRY " in log and "retries=1" in log


def test_lock_timeout_behind_a_non_backup_session_fails_at_once(scratch):
    _q(scratch.pg, "CREATE TABLE pp_probe_keep (id int)")
    f = scratch.add("manual_migration_pp_a.sql", _ADD_COLUMN)
    with _Holder(scratch.pg, "orion-some-reader", 11):
        rc, text = _apply(scratch, [f], backoff_s=(0.5, 0.5, 0.5))
    assert rc == 1, text
    assert "canceling statement due to lock timeout" in text
    assert "blocker: orion-some-reader pid=" in text and "not a backup, not retried" in text
    assert "RETRY" not in scratch.log.read_text()
    assert "FAILED services/orion-sql-db/manual_migration_pp_a.sql" in scratch.log.read_text()
    assert not _has_note(scratch.pg)
