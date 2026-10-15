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
#
# An allowlist. Every case below that is refused includes the review probes that slipped past the
# first (denylist) draft: dynamic SQL, psql meta-commands, type changes, WHERE hiding in a
# subquery, CTE writes, MERGE, CASCADE, function calls.


@pytest.mark.parametrize("sql", [
    "DROP TABLE foo;",
    "DROP TABLE IF EXISTS a, b CASCADE;",
    "ALTER TABLE t DROP COLUMN c;",
    "ALTER TABLE t RENAME COLUMN a TO b;",
    "ALTER TABLE t RENAME TO u;",
    "DROP INDEX IF EXISTS ix_old;",
    "DROP SCHEMA s CASCADE;",
    "TRUNCATE t;",
    "DELETE FROM t;",
    "DELETE FROM t WHERE created_at < now();",
    "UPDATE t SET a = 1 WHERE a IS NULL;",
    "UPDATE t SET c = (SELECT 1 FROM y WHERE y.id = 1);",
    "WITH d AS (DELETE FROM t RETURNING id) INSERT INTO log SELECT id FROM d WHERE id > 0;",
    "DELETE FROM t USING u WHERE true;",
    "MERGE INTO t USING s ON t.id = s.id WHEN MATCHED THEN DELETE;",
    "ALTER TABLE t ALTER COLUMN c TYPE int USING NULL;",
    "ALTER TABLE t ALTER c SET DATA TYPE varchar(3);",
    "ALTER TABLE t DROP CONSTRAINT pk CASCADE;",
    "ALTER TABLE t DETACH PARTITION p;",
    "ALTER VIEW v RENAME TO w;",
    "DROP FUNCTION f() CASCADE;",
    "DROP TRIGGER IF EXISTS trg ON t;",
    "DO $$ BEGIN EXECUTE 'DROP TABLE x'; END $$;",
    "DO $$ BEGIN EXECUTE format('TRUNCATE %I', 't'); END $$;",
    "DO $$ BEGIN IF NOT EXISTS (SELECT 1) THEN DELETE FROM posterior; END IF; END $$;",
    "CREATE FUNCTION f() RETURNS void AS $fn$ ALTER TABLE t DROP COLUMN c $fn$ LANGUAGE sql; SELECT f();",
    "CALL proc();",
    "SELECT f();",
    "CREATE TABLE t2 AS SELECT * FROM t;",
    "INSERT INTO t SELECT * FROM u ON CONFLICT DO NOTHING;",
    "INSERT INTO t (id) VALUES (1);",
    "INSERT INTO t (id) VALUES (1) ON CONFLICT (id) DO UPDATE SET id = excluded.id;",
    "INSERT INTO t (id) VALUES (1) ON CONFLICT DO NOTHING RETURNING id;",
    "\\i other.sql\nCREATE TABLE x (id int);",
    "\\! rm -rf /\nCREATE TABLE x (id int);",
    "\\set AUTOCOMMIT off\nCREATE TABLE x (id int);",
    "CREATE TABLE x (id int);\nSELECT 'DROP TABLE x' \\gexec",
    "VACUUM t;",
    "GRANT ALL ON t TO PUBLIC;",
    "SET search_path = evil;",
    "CREATE TEMP TABLE x (id int);",
    # second review: calls riding after an allowed prefix run per existing row
    "ALTER TABLE t ADD COLUMN c int DEFAULT f();",
    "ALTER TABLE t ADD COLUMN c int GENERATED ALWAYS AS (a * 2) STORED;",
    "ALTER TABLE t ADD CONSTRAINT k CHECK (f(a) > 0) NOT VALID;",
    "INSERT INTO t (a) VALUES (f()) ON CONFLICT DO NOTHING;",
    # quoted identifiers can spell an allowed prefix
    'ALTER TABLE "t add column x" DROP COLUMN secret;',
    'ALTER TABLE t ALTER COLUMN "c set default" TYPE text;',
    # not additive
    "ALTER TABLE t DROP CONSTRAINT t_pkey;",
    "ALTER TABLE t DROP CONSTRAINT IF EXISTS t_old_chk;",
    "ALTER TABLE t ALTER COLUMN a DROP DEFAULT;",
    "ALTER TABLE t ALTER COLUMN a DROP NOT NULL;",
    "ALTER TABLE t ALTER COLUMN a SET NOT NULL;",
    # a file must not lift the wrapper's timeouts
    "SET lock_timeout = 0;",
    "SET LOCAL statement_timeout = 0;",
    "BEGIN;\nCREATE TABLE x (id int);",
    # long write locks on a table that already exists (no catalog given = every table exists)
    "CREATE INDEX ix ON t (c);",
    "ALTER TABLE t ADD CONSTRAINT t_c_chk CHECK (c > 0);",
    "ALTER TABLE t ADD COLUMN c int UNIQUE;",
    "ALTER TABLE t ADD COLUMN c int REFERENCES u (id);",
    "CREATE INDEX CONCURRENTLY ix ON t ((f(a)));",
])
def test_anything_off_the_allowlist_is_refused(sql):
    assert d.auto_apply_refusals(sql), sql


def test_fresh_table_gets_plain_indexes_and_constraints_existing_one_does_not():
    sql = ("CREATE TABLE IF NOT EXISTS t (id int PRIMARY KEY, n text CHECK (length(n) < 9));\n"
           "CREATE INDEX IF NOT EXISTS ix_t ON t (lower(n));\n"
           "ALTER TABLE t ADD COLUMN c int REFERENCES u (id);\n")
    assert d.auto_apply_refusals(sql, existing_tables=set()) == []
    refused = d.auto_apply_refusals(sql, existing_tables={"t"})
    assert any("function call lower()" in r for r in refused)
    assert any("new column with a constraint on existing table t" in r for r in refused)
    plain = d.auto_apply_refusals("CREATE INDEX IF NOT EXISTS ix_t ON t (n);", existing_tables={"t"})
    assert plain and "without CONCURRENTLY" in plain[0]


@pytest.mark.parametrize("sql", [
    "CREATE TABLE IF NOT EXISTS t (id int PRIMARY KEY, note text DEFAULT 'x');",
    "ALTER TABLE t ADD COLUMN IF NOT EXISTS c int;",
    "ALTER TABLE t ADD COLUMN a int, ADD COLUMN b text;",
    "ALTER TABLE IF EXISTS t ADD c int;",
    "ALTER TABLE t ADD CONSTRAINT t_c_chk CHECK (c > 0) NOT VALID;",
    "ALTER TABLE t ADD CONSTRAINT t_u_fk FOREIGN KEY (u) REFERENCES u (id) NOT VALID;",
    "ALTER TABLE t ALTER COLUMN c SET DEFAULT 0, ALTER COLUMN d SET DEFAULT now();",
    "ALTER TABLE t ADD COLUMN c varchar(20) DEFAULT 'x', ADD COLUMN e numeric(10, 2), ADD COLUMN v vector(768);",
    "ALTER TABLE t ADD COLUMN at timestamptz NOT NULL DEFAULT now();",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix ON t (c);",
    "CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS ux ON t USING btree (c) WHERE c IS NOT NULL;",
    "CREATE TABLE IF NOT EXISTS t (id uuid DEFAULT gen_random_uuid(), n text CHECK (length(n) < 9));",
    "CREATE SEQUENCE IF NOT EXISTS s;",
    "COMMENT ON TABLE t IS 'we never DELETE FROM t';",
    "COMMENT ON COLUMN t.c IS 'x';",
    "INSERT INTO cur (name, at) VALUES ('r', null) ON CONFLICT (name) DO NOTHING;",
    "BEGIN;\nCREATE TABLE t (id int);\nCOMMIT;",
    "\\set ON_ERROR_STOP on\nCREATE TABLE t (id int);",
    # prose never counts
    "-- this used to DROP TABLE foo and TRUNCATE bar\n/* EXECUTE 'x' */\nCREATE TABLE t (id int);",
])
def test_additive_ddl_is_allowed(sql):
    assert d.auto_apply_refusals(sql) == []


def test_corpus_calibration():
    """Of the real migrations that declare REQUIRED-BY, the plain additive ones pass and the
    drop / DO-block ones do not; the guarded posterior wipe is refused."""
    def refused(name):
        return d.auto_apply_refusals((REPO_ROOT / d.MIGRATION_SUBDIR / name).read_text(errors="replace"))
    assert refused("manual_migration_action_value_control_arm.sql")
    assert refused("manual_migration_gpu_pool_stage5_drop_legacy_tables.sql")
    assert refused("manual_migration_retire_streak_tick_v1.sql")
    assert refused("manual_migration_vision_ask_names_v1.sql")  # DO block with a backfill UPDATE
    assert refused("manual_migration_walkway_privacy_v2.sql")  # DROP CONSTRAINT
    # as new migrations against a database that does not have their tables yet
    for ok in ("manual_migration_vision_confirm_asks_v1.sql", "manual_migration_curiosity_compaction.sql",
               "manual_migration_storage_write_substrate_loop.sql", "manual_migration_temporal_self_v1.sql"):
        assert d.auto_apply_refusals((REPO_ROOT / d.MIGRATION_SUBDIR / ok).read_text(), set()) == [], ok


@pytest.mark.parametrize("sql, expect", [
    ("CREATE TABLE t (id int);", []),
    ("CREATE INDEX CONCURRENTLY IF NOT EXISTS ix ON t (c);", ["CREATE INDEX CONCURRENTLY"]),
    ("BEGIN; CREATE TABLE t (id int); COMMIT;",
     ["own transaction control (BEGIN)", "own transaction control (COMMIT)"]),
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
    assert it.action == "apply" and it.no_txn == ["CREATE INDEX CONCURRENTLY"]


@pytest.mark.parametrize("rel, text, result, action, fragment", [
    ("services/orion-sql-db/manual_migration_x_rollback.sql", HDR + "DROP TABLE x;", None,
     "operator", "rollback file"),
    (REL, "-- DESTRUCTIVE\n-- ORION-MIGRATION-REQUIRED-BY: none retire\nDROP TABLE x;",
     _fr("manual_migration_x.sql"), "operator", "DESTRUCTIVE"),
    (REL, "CREATE TABLE x (id int);", _fr("manual_migration_x.sql"), "refuse", "no -- ORION-MIGRATION-REQUIRED-BY"),
    (REL, "-- ORION-MIGRATION-REQUIRED-BY: none\nCREATE TABLE x (id int);", _fr("manual_migration_x.sql"),
     "refuse", "invalid REQUIRED-BY"),
    (REL, HDR + "CREATE TABLE x (id int); DELETE FROM y;", _fr("manual_migration_x.sql"),
     "refuse", "statement not on the allowlist: DELETE FROM y"),
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
    # The primary main checkout, synchronized with origin/main: the only place the applier writes.
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    _git(repo, "remote", "add", "origin", str(origin))
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
        _git(repo, "push", "-q", "origin", "main")
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


def test_rollback_holds_nothing_but_destructive_and_refused_files_hold_later_files(world):
    r = world.add("manual_migration_a_rollback.sql", HDR + "DROP TABLE tz;")
    ok = world.add("manual_migration_b.sql", HDR + "CREATE TABLE tb (id int);")
    assert world.run([r, ok]) == 0, world.lines
    assert len(world.called()) == 1

    world.calls.unlink()
    x = world.add("manual_migration_c.sql", "-- DESTRUCTIVE\n-- ORION-MIGRATION-REQUIRED-BY: none retire\n"
                                            "DROP TABLE tz;")
    after_x = world.add("manual_migration_d.sql", HDR + "CREATE TABLE td (id int);")
    assert world.run([x, after_x]) == 1
    assert world.called() == []
    assert any("HELD, not applied -- manual_migration_c.sql comes first" in ln for ln in world.lines)

    bad = world.add("manual_migration_e.sql", HDR + "CREATE TABLE te (id int); TRUNCATE tb;")
    later = world.add("manual_migration_f.sql", HDR + "CREATE TABLE tf (id int);")
    assert world.run([bad, later]) == 1
    assert world.called() == []
    text = "\n".join(world.lines)
    assert "NOT auto-applied -- not safe to auto-apply: statement not on the allowlist: TRUNCATE tb" in text
    assert "HELD, not applied -- manual_migration_e.sql comes first" in text
    log = world.log.read_text()
    assert "NEEDS-HUMAN services/orion-sql-db/manual_migration_c.sql" in log
    assert "REFUSED services/orion-sql-db/manual_migration_e.sql" in log
    assert "HELD services/orion-sql-db/manual_migration_f.sql" in log


def test_unverifiable_data_file_holds_later_files(world):
    data = world.add("manual_migration_a_seed.sql", "-- ORION-MIGRATION-REQUIRED-BY: none seed\n"
                                                    "INSERT INTO seed (id) VALUES (1) ON CONFLICT DO NOTHING;")
    later = world.add("manual_migration_b.sql", HDR + "CREATE TABLE tb (id int);")
    assert world.run([data, later]) == 1
    assert world.called() == []
    assert any("UNVERIFIABLE (data" in ln for ln in world.lines)


@pytest.mark.parametrize("spoil, fragment", [
    (lambda w: _git(w.repo, "switch", "-q", "-c", "feat/x"), "not main"),
    (lambda w: (w.repo / "stray.txt").write_text("x"), "uncommitted changes"),
    (lambda w: (_git(w.repo, "commit", "-q", "--allow-empty", "-m", "local only")), "is not origin/main"),
])
def test_only_the_clean_synchronized_main_checkout_writes(world, spoil, fragment):
    a = world.add("manual_migration_a.sql", HDR + "CREATE TABLE ta (id int);")
    spoil(world)
    assert world.run([a]) == 2
    assert world.called() == []
    assert fragment in world.lines[0]
    # a dry run writes nothing, so it may run anywhere
    assert world.run([a], dry_run=True) == 0


def test_a_linked_worktree_never_writes(world, tmp_path):
    a = world.add("manual_migration_a.sql", HDR + "CREATE TABLE ta (id int);")
    wt = tmp_path / "wt"
    _git(world.repo, "worktree", "add", "-q", "--detach", str(wt))
    lines = []
    rc = ap.run([a], repo=wt, env=world.env, connect_fn=lambda: _FakeConn(world.state), out=lines.append)
    assert rc == 2 and world.called() == []
    assert "linked worktree" in lines[0]


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


def test_default_prefix_passes_lock_timeout_and_writes_the_database_it_reads():
    cmd = ap.default_psql({"ORION_PG_USER": "u", "ORION_PG_DB": "otherdb"})
    assert cmd[:5] == ["docker", "exec", "-i", "-e", f"PGOPTIONS={ap.PGOPTIONS}"]
    assert cmd[-4:] == ["-U", "u", "-d", "otherdb"]
    assert ap.default_psql({})[-4:] == ["-U", "postgres", "-d", "conjourney"]
    assert "lock_timeout=10s" in ap.PGOPTIONS


def test_default_command_refuses_when_the_checker_reads_another_host(world):
    """The default command writes the local container; verification reads ORION_PG_HOST. If those
    differ, "verified" would describe a different database -- refuse instead."""
    a = world.add("manual_migration_a.sql", HDR + "CREATE TABLE ta (id int);")
    world.env.pop("ORION_MIGRATION_APPLY_PSQL")
    world.env["ORION_PG_HOST"] = "10.9.8.7"
    assert world.run([a]) == 2
    assert world.called() == []
    assert any("refusing to write one database and verify another" in x for x in world.lines)
