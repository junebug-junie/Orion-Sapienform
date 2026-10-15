"""Post-pull auto-apply of hand-written SQL migrations (scripts/apply_post_pull_migrations.py).

Juniper, 2026-10-15: a pull that brings a migration starts the rebuilds at once, and the deploy
gate refuses the consumers until someone applies it by hand. The applier runs first, but only on
files that cannot lose data. These tests pin the safety filter, the ordering, the stop-on-failure
rule, the kill switch and the host gate without a database (fake psql + fake catalog);
tests/test_post_pull_apply_migrations_postgres.py runs the same path against a real server.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion import sql_migration_drift as d  # noqa: E402


def _load():
    spec = importlib.util.spec_from_file_location("apply_post_pull_migrations",
                                                  REPO_ROOT / "scripts" / "apply_post_pull_migrations.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


ap = _load()
HDR = "-- ORION-MIGRATION-REQUIRED-BY: orion-hub\n"


# --------------------------------------------------------------------------- safety filter


@pytest.mark.parametrize("sql, reason", [
    ("DROP TABLE foo;", "drops or renames table foo"),
    ("DROP TABLE IF EXISTS a, b CASCADE;", "drops or renames table a"),
    ("ALTER TABLE t DROP COLUMN c;", "drops or renames column t.c"),
    ("ALTER TABLE t RENAME COLUMN a TO b;", "drops or renames column t.a"),
    ("ALTER TABLE t RENAME TO u;", "drops or renames table t"),
    ("DROP INDEX IF EXISTS ix_old;", "drops or renames index ix_old"),
    ("DROP SCHEMA s CASCADE;", "DROP SCHEMA statement"),
    ("DROP VIEW v;", "DROP VIEW statement"),
    ("TRUNCATE t;", "TRUNCATE statement"),
    ("DELETE FROM t;", "DELETE with no WHERE"),
    ("UPDATE t SET a = 1;", "UPDATE with no WHERE"),
    ("UPDATE ONLY public.t AS x SET a = 1;", "UPDATE with no WHERE"),
    # The real wipe in manual_migration_action_value_control_arm.sql sits inside a guarded DO.
    ("DO $$ BEGIN IF NOT EXISTS (SELECT 1) THEN DELETE FROM posterior; END IF; END $$;",
     "DELETE with no WHERE"),
    ("CREATE FUNCTION f() RETURNS void AS $fn$ BEGIN TRUNCATE t; END $fn$ LANGUAGE plpgsql;",
     "TRUNCATE statement"),
])
def test_hazards_are_refused(sql, reason):
    assert reason in d.auto_apply_hazards(sql)


@pytest.mark.parametrize("sql", [
    "CREATE TABLE IF NOT EXISTS t (id int);",
    "ALTER TABLE t ADD COLUMN IF NOT EXISTS c int;",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix ON t (c);",
    "DELETE FROM t WHERE created_at < now() - interval '30 days';",
    "UPDATE t SET a = 1 WHERE a IS NULL;",
    "INSERT INTO t (id) VALUES (1) ON CONFLICT (id) DO UPDATE SET id = excluded.id;",
    "ALTER TABLE t DROP CONSTRAINT IF EXISTS t_pkey;",
    "ALTER TABLE t ALTER COLUMN c DROP NOT NULL;",
    "DROP TRIGGER IF EXISTS trg ON t; CREATE TRIGGER trg BEFORE UPDATE ON t FOR EACH ROW EXECUTE FUNCTION f();",
    "DROP FUNCTION IF EXISTS f();",
    # Prose and strings never count: the header comment and a literal mention a DROP.
    "-- this used to DROP TABLE foo and TRUNCATE bar\nCOMMENT ON TABLE t IS 'we never DELETE FROM t';",
])
def test_safe_statements_pass(sql):
    assert d.auto_apply_hazards(sql) == []


def test_every_hazard_in_the_corpus_is_a_drop_rename_or_wipe():
    """Calibration against the real corpus: every non-rollback, non-DESTRUCTIVE file the filter
    refuses is refused for a stated reason, and the known guarded wipe is among them."""
    flagged = {}
    for p in sorted((REPO_ROOT / d.MIGRATION_SUBDIR).glob("*.sql")):
        h = d.auto_apply_hazards(p.read_text(errors="replace"))
        if h:
            flagged[p.name] = h
    assert "DELETE with no WHERE" in flagged["manual_migration_action_value_control_arm.sql"]
    assert any("drops or renames table" in r
               for r in flagged["manual_migration_gpu_pool_stage5_drop_legacy_tables.sql"])
    assert "manual_migration_vision_ask_names_v1.sql" not in flagged


@pytest.mark.parametrize("sql, expect", [
    ("CREATE TABLE t (id int);", []),
    ("CREATE INDEX CONCURRENTLY IF NOT EXISTS ix ON t (c);", ["CREATE INDEX CONCURRENTLY IF"]),
    ("BEGIN; CREATE TABLE t (id int); COMMIT;",
     ["own transaction control (BEGIN)", "own transaction control (COMMIT)"]),
    ("VACUUM ANALYZE t;", ["VACUUM ANALYZE T"]),
    # BEGIN/END inside a DO body is plpgsql, not transaction control.
    ("DO $$ BEGIN PERFORM 1; END $$;", []),
])
def test_transaction_incompatible(sql, expect):
    assert d.transaction_incompatible(sql) == expect


# --------------------------------------------------------------------------- classify


def _fr(name, status="MISSING", **kw):
    from datetime import datetime, timezone
    return d.FileResult(name=name, in_window=True, changed_at=datetime.now(timezone.utc), status=status, **kw)


REL = "services/orion-sql-db/manual_migration_x.sql"


def test_classify_safe_missing_file_is_applied():
    it = ap.classify(REL, HDR + "CREATE TABLE x (id int);", _fr("manual_migration_x.sql"))
    assert it.action == "apply" and it.no_txn == []


def test_classify_concurrent_index_runs_without_wrapper():
    it = ap.classify(REL, HDR + "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix ON t (c);",
                     _fr("manual_migration_x.sql"))
    assert it.action == "apply" and it.no_txn == ["CREATE INDEX CONCURRENTLY IF"]


@pytest.mark.parametrize("rel, text, result, action, fragment", [
    ("services/orion-sql-db/manual_migration_x_rollback.sql", HDR + "DROP TABLE x;", None,
     "operator", "rollback file"),
    (REL, "-- DESTRUCTIVE\n-- ORION-MIGRATION-REQUIRED-BY: none retire\nDROP TABLE x;",
     _fr("manual_migration_x.sql"), "operator", "DESTRUCTIVE"),
    (REL, "CREATE TABLE x (id int);", _fr("manual_migration_x.sql"), "refuse", "no -- ORION-MIGRATION-REQUIRED-BY"),
    (REL, "-- ORION-MIGRATION-REQUIRED-BY: none\nCREATE TABLE x (id int);", _fr("manual_migration_x.sql"),
     "refuse", "invalid REQUIRED-BY"),
    (REL, HDR + "CREATE TABLE x (id int); DELETE FROM y;", _fr("manual_migration_x.sql"),
     "refuse", "DELETE with no WHERE"),
    (REL, HDR + "CREATE TABLE x (id int);", _fr("manual_migration_x.sql", header_script="scripts/apply_x.sh"),
     "refuse", "scripts/apply_x.sh"),
    (REL, HDR + "CREATE INDEX ix ON t (c);", _fr("manual_migration_x.sql", status="INVALID"),
     "refuse", "INVALID"),
    (REL, HDR + "CREATE TABLE x (id int);", _fr("manual_migration_x.sql", status="APPLIED"), "skip", "APPLIED"),
    (REL, "-- ORION-MIGRATION-REQUIRED-BY: none data\nUPDATE t SET a = 1 WHERE a IS NULL;",
     _fr("manual_migration_x.sql", status="DATA"), "skip", "UNVERIFIABLE"),
    (REL, HDR + "CREATE TABLE x (id int);", None, "refuse", "did not load"),
])
def test_classify_refusals_and_skips(rel, text, result, action, fragment):
    it = ap.classify(rel, text, result)
    assert it.action == action, it
    assert fragment in it.reason
    if action == "operator":
        assert it.manual == ""  # never print a command that runs a rollback/DESTRUCTIVE file


# ----------------------------------------------------- run(): fake psql + fake catalog, real git


_FAKE_PSQL = r'''
import json, os, sys
sys.path.insert(0, os.environ["ORION_TEST_REPO_ROOT"])
from orion import sql_migration_drift as d
sql = sys.stdin.read()
state_path = os.environ["FAKE_DB_STATE"]
state = json.load(open(state_path))
with open(os.environ["FAKE_PSQL_CALLS"], "a") as fh:
    fh.write(json.dumps({"argv": sys.argv[1:], "first_line": sql.splitlines()[0] if sql else "",
                         "pgoptions": os.environ.get("PGOPTIONS", "")}) + "\n")
if "FAKE-FAIL" in sql:
    print("ERROR:  canceling statement due to lock timeout", file=sys.stderr)
    raise SystemExit(3)
if "FAKE-NOOP" not in sql:
    for e in d.parse_migration(sql).effects:
        if e.op == "create":
            key = {"table": "tables", "column": "columns", "index": "indexes", "sequence": "sequences"}[e.kind]
            state[key].append(e.name)
json.dump(state, open(state_path, "w"))
'''


class _FakeCursor:
    def __init__(self, state):
        self.state, self.rows = state, []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql):
        st = self.state
        if sql == d.LIVE_STATE_SQL["tables"]:
            self.rows = [(t,) for t in st["tables"]]
        elif sql == d.LIVE_STATE_SQL["columns"]:
            self.rows = [tuple(c.split(".", 1)) for c in st["columns"]]
        elif sql == d.LIVE_STATE_SQL["indexes"]:
            self.rows = [(i, True) for i in st["indexes"]]
        else:
            self.rows = [(s,) for s in st["sequences"]]

    def fetchall(self):
        return self.rows


class _FakeConn:
    def __init__(self, path):
        self.state = json.loads(Path(path).read_text())

    def cursor(self):
        return _FakeCursor(self.state)

    def close(self):
        pass


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture()
def world(tmp_path):
    repo = tmp_path / "repo"
    (repo / d.MIGRATION_SUBDIR).mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    state = tmp_path / "db.json"
    state.write_text(json.dumps({"tables": [], "columns": [], "indexes": [], "sequences": []}))
    fake = tmp_path / "fake_psql.py"
    fake.write_text(_FAKE_PSQL)
    calls = tmp_path / "calls.jsonl"
    env = {
        "ORION_TEST_REPO_ROOT": str(REPO_ROOT),
        "FAKE_DB_STATE": str(state),
        "FAKE_PSQL_CALLS": str(calls),
        "ORION_MIGRATION_APPLY_PSQL": f"{sys.executable} {fake}",
        "ORION_APPLIED_MIGRATIONS_LOG": str(tmp_path / "applied.log"),
        "PATH": os.environ["PATH"],
    }

    class W:
        pass

    w = W()
    w.repo, w.env, w.calls, w.state, w.log = repo, env, calls, state, tmp_path / "applied.log"
    w.lines = []

    def add(name, sql):
        (repo / d.MIGRATION_SUBDIR / name).write_text(sql)
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", name)
        return f"{d.MIGRATION_SUBDIR}/{name}"

    def run(paths, **kw):
        w.lines = []
        return ap.run(paths, repo=repo, env=env, connect_fn=lambda: _FakeConn(state),
                      out=w.lines.append, **kw)

    def called():
        if not calls.exists():
            return []
        return [json.loads(x) for x in calls.read_text().splitlines()]

    w.add, w.run, w.called = add, run, called
    return w


def test_applies_in_filename_order_with_lock_timeout_and_logs(world):
    b = world.add("manual_migration_b.sql", "-- b\n" + HDR + "CREATE TABLE tb (id int);")
    a = world.add("manual_migration_a.sql", "-- a\n" + HDR + "CREATE TABLE ta (id int);")
    rc = world.run([b, a])
    assert rc == 0, world.lines
    calls = world.called()
    assert [c["first_line"] for c in calls] == ["-- a", "-- b"]
    for c in calls:
        assert "--single-transaction" in c["argv"] and "ON_ERROR_STOP=1" in c["argv"]
        assert "lock_timeout=10s" in c["pgoptions"]
    log = world.log.read_text().splitlines()
    assert len(log) == 2 and "APPLIED services/orion-sql-db/manual_migration_a.sql" in log[0]
    assert any("verified: services/orion-sql-db/manual_migration_b.sql is APPLIED" in x for x in world.lines)


def test_concurrent_index_file_runs_without_single_transaction(world):
    t = world.add("manual_migration_a.sql", HDR + "CREATE TABLE ta (id int);")
    i = world.add("manual_migration_b.sql", "-- ORION-MIGRATION-REQUIRED-BY: orion-hub\n"
                                            "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_ta ON ta (id);")
    assert world.run([t, i]) == 0, world.lines
    argvs = [c["argv"] for c in world.called()]
    assert "--single-transaction" in argvs[0]
    assert "--single-transaction" not in argvs[1]


def test_failure_stops_every_later_file(world):
    a = world.add("manual_migration_a.sql", HDR + "-- FAKE-FAIL\nCREATE TABLE ta (id int);")
    b = world.add("manual_migration_b.sql", HDR + "CREATE TABLE tb (id int);")
    assert world.run([a, b]) == 1
    assert len(world.called()) == 1
    text = "\n".join(world.lines)
    assert "lock timeout" in text and "HELD, not applied" in text
    assert "FAILED services/orion-sql-db/manual_migration_a.sql" in world.log.read_text()


def test_refused_file_holds_later_files_but_destructive_and_rollback_do_not(world):
    r = world.add("manual_migration_a_rollback.sql", HDR + "DROP TABLE tz;")
    x = world.add("manual_migration_b.sql", "-- DESTRUCTIVE\n-- ORION-MIGRATION-REQUIRED-BY: none retire\n"
                                            "DROP TABLE tz;")
    ok = world.add("manual_migration_c.sql", HDR + "CREATE TABLE tc (id int);")
    assert world.run([r, x, ok]) == 0, world.lines
    assert len(world.called()) == 1

    world.calls.unlink()
    bad = world.add("manual_migration_d.sql", HDR + "CREATE TABLE td (id int); TRUNCATE tc;")
    later = world.add("manual_migration_e.sql", HDR + "CREATE TABLE te (id int);")
    assert world.run([bad, later]) == 1
    assert world.called() == []
    text = "\n".join(world.lines)
    assert "NOT auto-applied -- not safe to auto-apply: TRUNCATE statement" in text
    assert "HELD, not applied -- manual_migration_d.sql comes first" in text


def test_already_applied_is_skipped(world):
    a = world.add("manual_migration_a.sql", HDR + "CREATE TABLE ta (id int);")
    world.state.write_text(json.dumps({"tables": ["ta"], "columns": [], "indexes": [], "sequences": []}))
    assert world.run([a]) == 0
    assert world.called() == []
    assert any(x.endswith("manual_migration_a.sql: APPLIED") for x in world.lines)


def test_psql_success_that_the_checker_does_not_confirm_is_unverified(world):
    a = world.add("manual_migration_a.sql", HDR + "-- FAKE-NOOP\nCREATE TABLE ta (id int);")
    assert world.run([a]) == 1
    assert any("NOT verified" in x for x in world.lines)
    assert "UNVERIFIED" in world.log.read_text()


def test_kill_switch_applies_nothing(world):
    a = world.add("manual_migration_a.sql", HDR + "CREATE TABLE ta (id int);")
    world.env["ORION_POST_PULL_APPLY_MIGRATIONS"] = "0"
    assert world.run([a]) == 0
    assert world.called() == []
    assert "auto-apply OFF" in world.lines[0]


def test_dry_run_applies_nothing(world):
    a = world.add("manual_migration_a.sql", HDR + "CREATE TABLE ta (id int);")
    assert world.run([a], dry_run=True) == 0
    assert world.called() == []
    assert any("would apply (single transaction" in x for x in world.lines)
    assert not world.log.exists()


def test_unreachable_database_applies_nothing(world):
    a = world.add("manual_migration_a.sql", HDR + "CREATE TABLE ta (id int);")

    def boom():
        raise RuntimeError("could not connect to localhost:55432")

    lines = []
    rc = ap.run([a], repo=world.repo, env=world.env, connect_fn=boom, out=lines.append)
    assert rc == 2 and world.called() == []
    assert "UNKNOWN" in lines[0]


def test_default_command_refuses_when_the_db_container_is_not_on_this_host(world, tmp_path):
    """Host gate, second layer: with the default `docker exec` prefix nothing runs unless
    orion-athena-sql-db is running here (circe has no database)."""
    a = world.add("manual_migration_a.sql", HDR + "CREATE TABLE ta (id int);")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text("#!/bin/sh\necho \"$*\" >> \"$FAKE_PSQL_CALLS\"\n"
                      "[ \"$1\" = inspect ] && { echo 'Error: No such object' >&2; exit 1; }\nexit 0\n")
    docker.chmod(0o755)
    world.env.pop("ORION_MIGRATION_APPLY_PSQL")
    world.env["PATH"] = f"{bin_dir}:{os.environ['PATH']}"
    assert world.run([a]) == 2
    assert world.calls.read_text().splitlines()[0].startswith("inspect")
    assert len(world.calls.read_text().splitlines()) == 1  # inspect only, never exec
    assert any("is not running on this host" in x for x in world.lines)


def test_default_prefix_passes_lock_timeout_into_the_container():
    assert ap.DEFAULT_PSQL[:5] == ["docker", "exec", "-i", "-e", f"PGOPTIONS={ap.PGOPTIONS}"]
    assert "lock_timeout=10s" in ap.PGOPTIONS
