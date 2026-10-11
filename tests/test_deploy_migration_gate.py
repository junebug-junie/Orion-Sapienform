"""Deploy-time SQL migration gate: a service cannot be brought up while a migration it needs is
unapplied (2026-10-10/11: PRs #2594 and #2605 deployed orion-durable-runs before
manual_migration_temporal_self_event_v1.sql / manual_migration_temporal_self_v1.sql were applied;
every step failed with UndefinedTable until they were applied by hand).

Layers covered:
- orion/sql_migration_drift.py: REQUIRED-BY parsing, DESTRUCTIVE exclusion, window-free verdict.
- scripts/check_sql_migrations_applied.py --service: applied / missing (+ exact apply command) /
  DB unreachable = UNKNOWN.
- scripts/safe_docker_build.sh: refuses `up` on exit 1 and 2, honours the override, never gates
  build/config/logs.
- scripts/check_migration_required_by.py: the CI header check.

DB-free and docker-free (fake `docker` binary, fake live state).
"""
from __future__ import annotations

import importlib.util
import os
import stat
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion import sql_migration_drift as d  # noqa: E402

MIG_DIR = REPO_ROOT / "services" / "orion-sql-db"
WRAPPER = REPO_ROOT / "scripts" / "safe_docker_build.sh"
OLD = datetime(2026, 1, 1, tzinfo=timezone.utc)
APPLY_PREFIX = "docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 < "


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gate_cli = _load("check_sql_migrations_applied")
ci_check = _load("check_migration_required_by")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


def _init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q", "-b", "main")
    _git(path, "config", "user.email", "t@example.com")
    _git(path, "config", "user.name", "T")
    return path


# ------------------------------------------------------------------ parsing


def test_parse_required_by_variants():
    assert d.parse_required_by("SELECT 1;") is None
    rb = d.parse_required_by("-- ORION-MIGRATION-REQUIRED-BY: orion-durable-runs, orion-dream\nBEGIN;")
    assert rb.services == ("orion-durable-runs", "orion-dream") and not rb.errors
    rb = d.parse_required_by("-- ORION-MIGRATION-REQUIRED-BY: none -- read only by ad-hoc analysis")
    assert rb.services == () and rb.none_reason == "read only by ad-hoc analysis" and not rb.errors
    assert d.parse_required_by("-- ORION-MIGRATION-REQUIRED-BY: none").errors
    assert d.parse_required_by("-- ORION-MIGRATION-REQUIRED-BY:").errors
    assert d.parse_required_by("-- ORION-MIGRATION-REQUIRED-BY: Orion Hub").errors


def test_the_two_incident_migrations_require_durable_runs():
    files = [d.MigrationFile(p.name, p.read_text(), OLD, OLD) for p in MIG_DIR.glob("*.sql")]
    names = {f.name for f in d.required_for(files, "orion-durable-runs")}
    assert {"manual_migration_temporal_self_event_v1.sql", "manual_migration_temporal_self_v1.sql"} <= names


def test_destructive_file_is_never_required_even_if_it_says_so():
    text = "-- DESTRUCTIVE: operator only\n-- ORION-MIGRATION-REQUIRED-BY: orion-x\nDROP TABLE t;"
    assert d.is_destructive(text)
    assert d.required_for([d.MigrationFile("m.sql", text, OLD, OLD)], "orion-x") == []
    # The real retire file is DESTRUCTIVE and is no service's dependency.
    retire = (MIG_DIR / "manual_migration_retire_streak_tick_v1.sql").read_text()
    assert d.is_destructive(retire)
    all_files = [d.MigrationFile(p.name, p.read_text(), OLD, OLD) for p in MIG_DIR.glob("*.sql")]
    for svc in ("orion-attention-runtime", "orion-sql-writer"):
        assert "manual_migration_retire_streak_tick_v1.sql" not in {f.name for f in d.required_for(all_files, svc)}


def test_deploy_gate_ignores_the_recency_window():
    f = d.MigrationFile("manual_migration_a.sql",
                        "-- ORION-MIGRATION-REQUIRED-BY: orion-x\nCREATE TABLE a (id int);", OLD, OLD)
    report = d.evaluate([f], d.LiveState(set(), set(), {}, set()),
                        now=OLD + timedelta(days=400), window_days=30)
    assert not report.red  # outside the watch's window...
    gate = d.deploy_gate(report, ["manual_migration_a.sql"], "orion-x")
    assert not gate.ok  # ...but a deploy that needs it still refuses


# ------------------------------------------------------------------ CLI --service


@pytest.fixture
def mig_repo(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path / "repo")
    (repo / "services" / "orion-sql-db").mkdir(parents=True)
    (repo / "services" / "orion-sql-db" / "manual_migration_need_v1.sql").write_text(
        "-- ORION-MIGRATION-REQUIRED-BY: orion-x\nCREATE TABLE IF NOT EXISTS need_t (id int);\n")
    (repo / "services" / "orion-sql-db" / "manual_migration_retire_v1.sql").write_text(
        "-- DESTRUCTIVE: operator only\n-- ORION-MIGRATION-REQUIRED-BY: orion-y\nDROP TABLE IF EXISTS other_t;\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    monkeypatch.setattr(gate_cli, "REPO_ROOT", repo)
    return repo


class _Conn:
    def close(self):
        pass


def _live(monkeypatch, tables):
    monkeypatch.setattr(d, "load_live_state", lambda conn: d.LiveState(set(tables), set(), {}, set()))


def test_service_gate_passes_when_applied(mig_repo, monkeypatch, capsys):
    _live(monkeypatch, {"need_t"})
    assert gate_cli.service_gate("orion-x", connect_fn=_Conn) == 0
    assert "all 1 required migration(s) applied" in capsys.readouterr().out


def test_service_gate_refuses_missing_with_exact_apply_command(mig_repo, monkeypatch, capsys):
    _live(monkeypatch, set())
    assert gate_cli.service_gate("orion-x", connect_fn=_Conn) == 1
    err = capsys.readouterr().err
    assert "MISSING manual_migration_need_v1.sql" in err
    assert APPLY_PREFIX + "services/orion-sql-db/manual_migration_need_v1.sql" in err


def test_service_gate_db_down_is_unknown_not_applied(mig_repo, capsys):
    def down():
        raise gate_cli.ConnectError("could not connect to localhost:55432/conjourney: refused")
    assert gate_cli.service_gate("orion-x", connect_fn=down) == 2
    assert "UNKNOWN" in capsys.readouterr().err


def test_service_gate_never_connects_when_nothing_is_required(mig_repo, capsys):
    def boom():
        raise AssertionError("must not connect")
    # orion-y is only named by a DESTRUCTIVE file: not a dependency, so no DB needed at all.
    assert gate_cli.service_gate("orion-y", connect_fn=boom) == 0
    assert gate_cli.service_gate("orion-unrelated", connect_fn=boom) == 0


# ------------------------------------------------------------------ wrapper


def _wrapper_repo(tmp_path: Path, stub_rc: int) -> tuple[Path, Path, dict]:
    primary = _init_repo(tmp_path / "primary")
    svc = primary / "services" / "orion-x"
    svc.mkdir(parents=True)
    (svc / "docker-compose.yml").write_text("services: {}\n")
    (svc / ".env").write_text("")
    (primary / ".env").write_text("")
    (primary / "scripts").mkdir()
    marker = tmp_path / "gate-called"
    (primary / "scripts" / "check_sql_migrations_applied.py").write_text(
        "import sys, pathlib\n"
        f"pathlib.Path({str(marker)!r}).write_text(' '.join(sys.argv[1:]))\n"
        f"sys.exit({stub_rc})\n")
    _git(primary, "add", "-A")
    _git(primary, "commit", "-qm", "init")
    wt = tmp_path / "wt"
    _git(primary, "worktree", "add", "-q", str(wt), "-b", "chore/t")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker = fake_bin / "docker"
    docker.write_text('#!/bin/sh\necho fake-docker "$@"\n')
    docker.chmod(docker.stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}",
           "ORION_AGENT_BOARD_PATH": str(tmp_path / "board.jsonl")}
    env.pop("ORION_ALLOW_UNAPPLIED_MIGRATION", None)
    env.pop("CLAUDE_CODE_SESSION_ID", None)
    return wt, marker, env


def _run(wt, env, *args):
    return subprocess.run(["sh", str(WRAPPER), "orion-x", *args], cwd=wt, env=env,
                          capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize("rc,word", [(1, "not applied"), (2, "UNKNOWN")])
def test_wrapper_refuses_up(tmp_path, rc, word):
    wt, marker, env = _wrapper_repo(tmp_path, rc)
    p = _run(wt, env, "up", "-d", "--build")
    assert p.returncode == 1
    assert "fake-docker" not in p.stdout
    assert word in p.stderr and "ORION_ALLOW_UNAPPLIED_MIGRATION=1" in p.stderr
    assert marker.read_text() == "--service orion-x"


def test_wrapper_passes_up_when_gate_passes(tmp_path):
    wt, marker, env = _wrapper_repo(tmp_path, 0)
    p = _run(wt, env, "up", "-d")
    assert p.returncode == 0, p.stderr
    assert "fake-docker" in p.stdout and marker.exists()


def test_wrapper_override_skips_gate(tmp_path):
    wt, marker, env = _wrapper_repo(tmp_path, 1)
    env["ORION_ALLOW_UNAPPLIED_MIGRATION"] = "1"
    p = _run(wt, env, "up", "-d")
    assert p.returncode == 0, p.stderr
    assert "fake-docker" in p.stdout and "SKIPPING" in p.stderr
    assert not marker.exists()


@pytest.mark.parametrize("args", [("build",), ("config",), ("logs", "--tail=5"), ("ps",)])
def test_wrapper_does_not_gate_non_up(tmp_path, args):
    wt, marker, env = _wrapper_repo(tmp_path, 1)
    p = _run(wt, env, *args)
    assert p.returncode == 0, p.stderr
    assert not marker.exists()


# ------------------------------------------------------------------ CI header check


@pytest.fixture
def ci_repo(tmp_path):
    repo = _init_repo(tmp_path / "ci")
    (repo / "services" / "orion-sql-db").mkdir(parents=True)
    (repo / "services" / "orion-x").mkdir(parents=True)
    (repo / "services" / "orion-x" / "docker-compose.yml").write_text("services: {}\n")
    (repo / "services" / "orion-sql-db" / "manual_migration_old.sql").write_text("CREATE TABLE o (i int);\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    _git(repo, "branch", "base")
    return repo


def _add(repo, name, text):
    (repo / "services" / "orion-sql-db" / name).write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", f"add {name}")


def _ci(repo):
    return ci_check.main(["--repo", str(repo), "--base", "base"])


def test_ci_old_untagged_files_are_grandfathered(ci_repo):
    assert _ci(ci_repo) == 0


def test_ci_new_untagged_migration_fails(ci_repo, capsys):
    _add(ci_repo, "manual_migration_new.sql", "CREATE TABLE n (i int);\n")
    assert _ci(ci_repo) == 1
    assert "manual_migration_new.sql: new migration declares no consumer" in capsys.readouterr().err


@pytest.mark.parametrize("header,ok", [
    ("-- ORION-MIGRATION-REQUIRED-BY: orion-x", True),
    ("-- ORION-MIGRATION-REQUIRED-BY: none -- analysis-only table", True),
    ("-- ORION-MIGRATION-REQUIRED-BY: none", False),               # reason required
    ("-- ORION-MIGRATION-REQUIRED-BY: orion-nope", False),         # not a service
    ("-- DESTRUCTIVE: drop\n-- ORION-MIGRATION-REQUIRED-BY: orion-x", False),
    ("-- DESTRUCTIVE: drop\n-- ORION-MIGRATION-REQUIRED-BY: none -- retire", True),
])
def test_ci_declarations(ci_repo, header, ok):
    _add(ci_repo, "manual_migration_new.sql", header + "\nCREATE TABLE n (i int);\n")
    assert (_ci(ci_repo) == 0) is ok


def test_ci_rollback_files_are_exempt(ci_repo):
    _add(ci_repo, "manual_migration_new_rollback.sql", "DROP TABLE n;\n")
    assert _ci(ci_repo) == 0


def test_ci_unresolvable_base_is_an_error_not_a_pass(ci_repo):
    assert ci_check.main(["--repo", str(ci_repo), "--base", "no-such-ref"]) == 2


def test_ci_real_corpus_declarations_are_valid():
    assert ci_check.check_files(REPO_ROOT, set()) == []
