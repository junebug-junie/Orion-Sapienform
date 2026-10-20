"""Deploy-time SQL migration gate: a service cannot be brought up while a migration it needs is
unapplied (2026-10-10/11: PRs #2594 and #2605 deployed orion-durable-runs before
manual_migration_temporal_self_event_v1.sql / manual_migration_temporal_self_v1.sql were applied;
every step failed with UndefinedTable until they were applied by hand).

Layers covered:
- orion/sql_migration_drift.py: REQUIRED-BY parsing, DESTRUCTIVE exclusion, window-free verdict.
- scripts/check_sql_migrations_applied.py --service: applied / missing (+ exact apply command) /
  DB unreachable = UNKNOWN.
- scripts/safe_docker_build.sh: permits `up` only from clean synchronized primary main, refuses
  it on migration exit 1 and 2, honours the migration override, and never migration-gates
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
    gate = d.deploy_gate(report, [f], "orion-x")
    assert not gate.ok  # ...but a deploy that needs it still refuses


def _mf(name, text, days=0):
    return d.MigrationFile(name, text, OLD + timedelta(days=days), OLD + timedelta(days=days))


EMPTY = d.LiveState(set(), set(), {}, set())


def test_deploy_gate_blocks_when_a_later_file_redeclares_the_same_table():
    # Review finding: evaluate() blames the LAST creator, so A alone read APPLIED with t absent.
    a = _mf("manual_migration_a.sql", "-- ORION-MIGRATION-REQUIRED-BY: orion-x\nCREATE TABLE IF NOT EXISTS t (i int);")
    b = _mf("manual_migration_b.sql", "CREATE TABLE IF NOT EXISTS t (i int);", days=1)
    report = d.evaluate([a, b], EMPTY, window_days=None)
    gate = d.deploy_gate(report, [a], "orion-x")
    assert [r.name for r in gate.blocking] == ["manual_migration_a.sql"]
    full = d.LiveState({"t"}, set(), {}, set())
    assert d.deploy_gate(d.evaluate([a, b], full, window_days=None), [a], "orion-x").ok


def test_deploy_gate_follows_superseded_by_to_the_successor():
    a = _mf("manual_migration_a.sql", "-- ORION-MIGRATION-REQUIRED-BY: orion-x\n"
            "-- ORION-MIGRATION-SUPERSEDED-BY: manual_migration_b.sql\nCREATE TABLE IF NOT EXISTS t (i int);")
    b = _mf("manual_migration_b.sql", "CREATE TABLE IF NOT EXISTS t2 (i int);", days=1)
    gate = d.deploy_gate(d.evaluate([a, b], EMPTY, window_days=None), [a], "orion-x")
    assert "manual_migration_b.sql" in gate.required
    assert "manual_migration_b.sql" in [r.name for r in gate.blocking]


def test_deploy_gate_required_file_absent_from_report_blocks():
    a = _mf("manual_migration_a.sql", "-- ORION-MIGRATION-REQUIRED-BY: orion-x\nCREATE TABLE t (i int);")
    report = d.DriftReport(files=[], window_days=None)
    assert not d.deploy_gate(report, [a], "orion-x").ok


def test_none_is_not_a_prefix_match():
    assert d.parse_required_by("-- ORION-MIGRATION-REQUIRED-BY: none-svc").services == ("none-svc",)


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


# ------------------------------------------------------------------ post-pull report
# 2026-10-15: a migration-only pull recreated orion-athena-sql-db. The rebuild now skips
# migrations and prints this report instead (scripts/rebuild_services_from_git_diff.sh).

_NEED = "services/orion-sql-db/manual_migration_need_v1.sql"
_RETIRE = "services/orion-sql-db/manual_migration_retire_v1.sql"


def test_post_pull_report_missing_names_consumers_and_apply_command(mig_repo, monkeypatch):
    _live(monkeypatch, set())
    out = "\n".join(gate_cli.post_pull_report([_NEED], connect_fn=_Conn))
    assert f"  - {_NEED} (required by: orion-x)" in out
    assert "live state: NOT APPLIED -- the services above stay refused by safe_docker_build.sh" in out
    assert "apply: " + APPLY_PREFIX + _NEED in out


def test_post_pull_report_applied(mig_repo, monkeypatch):
    _live(monkeypatch, {"need_t"})
    out = "\n".join(gate_cli.post_pull_report([_NEED], connect_fn=_Conn))
    assert "live state: APPLIED" in out and "apply" not in out.split("APPLIED", 1)[1]


def test_post_pull_report_db_down_is_unknown_with_apply_command(mig_repo):
    def down():
        raise gate_cli.ConnectError("could not connect: refused")
    out = "\n".join(gate_cli.post_pull_report([_NEED], connect_fn=down))
    assert "live state: UNKNOWN (could not connect: refused; no answer is not 'applied')" in out
    assert "APPLIED" not in out.replace("NOT APPLIED", "")
    assert "apply if not yet applied: " + APPLY_PREFIX + _NEED in out


def test_post_pull_report_live_state_crash_is_unknown_not_missing(mig_repo, monkeypatch):
    def boom(conn):
        raise RuntimeError("catalog read failed")
    monkeypatch.setattr(d, "load_live_state", boom)
    out = "\n".join(gate_cli.post_pull_report([_NEED], connect_fn=_Conn))
    assert "live state: UNKNOWN (catalog read failed" in out
    assert "NOT APPLIED" not in out


def test_post_pull_report_never_offers_destructive_or_rollback(mig_repo, monkeypatch):
    rb = mig_repo / "services" / "orion-sql-db" / "manual_migration_need_v1_rollback.sql"
    rb.write_text("DROP TABLE IF EXISTS need_t;\n")
    _live(monkeypatch, set())
    rb_rel = "services/orion-sql-db/manual_migration_need_v1_rollback.sql"
    out = "\n".join(gate_cli.post_pull_report([_RETIRE, rb_rel], connect_fn=_Conn))
    assert "DESTRUCTIVE: operator-approved only; no apply command is printed" in out
    assert "rollback file: undoes another migration; never applied on deploy" in out
    assert "docker exec" not in out
    assert "stay refused" not in out


def test_post_pull_report_data_only_is_unverifiable_not_applied(mig_repo, monkeypatch):
    data = mig_repo / "services" / "orion-sql-db" / "manual_migration_backfill_v1.sql"
    data.write_text("-- ORION-MIGRATION-REQUIRED-BY: none -- backfill\nUPDATE need_t SET id = 1;\n")
    _live(monkeypatch, {"need_t"})
    out = "\n".join(gate_cli.post_pull_report(
        ["services/orion-sql-db/manual_migration_backfill_v1.sql"], connect_fn=_Conn))
    assert "live state: UNVERIFIABLE" in out
    assert "live state: APPLIED" not in out


# ------------------------------------------------------------------ wrapper


def _wrapper_repo(tmp_path: Path, stub_rc: int) -> tuple[Path, Path, Path, dict]:
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
    remote = tmp_path / "remote.git"
    remote.mkdir()
    _git(remote, "init", "--bare", "-q")
    _git(primary, "remote", "add", "origin", str(remote))
    _git(primary, "push", "-qu", "origin", "main")
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
    return primary, wt, marker, env


def _run(checkout, env, *args):
    return subprocess.run(["sh", str(WRAPPER), "orion-x", *args], cwd=checkout, env=env,
                          capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize("rc,word", [(1, "not applied"), (2, "UNKNOWN")])
def test_wrapper_refuses_up(tmp_path, rc, word):
    primary, _, marker, env = _wrapper_repo(tmp_path, rc)
    p = _run(primary, env, "up", "-d", "--build")
    assert p.returncode == 1
    assert "fake-docker" not in p.stdout
    assert word in p.stderr and "ORION_ALLOW_UNAPPLIED_MIGRATION=1" in p.stderr
    assert marker.read_text() == "--service orion-x"


def test_wrapper_passes_up_when_gate_passes(tmp_path):
    primary, _, marker, env = _wrapper_repo(tmp_path, 0)
    p = _run(primary, env, "up", "-d")
    assert p.returncode == 0, p.stderr
    assert "fake-docker" in p.stdout and marker.exists()


def test_wrapper_override_skips_gate(tmp_path):
    primary, _, marker, env = _wrapper_repo(tmp_path, 1)
    env["ORION_ALLOW_UNAPPLIED_MIGRATION"] = "1"
    p = _run(primary, env, "up", "-d")
    assert p.returncode == 0, p.stderr
    assert "fake-docker" in p.stdout and "SKIPPING" in p.stderr
    assert not marker.exists()


@pytest.mark.parametrize("args", [("build",), ("config",), ("logs", "--tail=5"), ("ps",)])
def test_wrapper_does_not_gate_non_up(tmp_path, args):
    _, wt, marker, env = _wrapper_repo(tmp_path, 1)
    p = _run(wt, env, *args)
    assert p.returncode == 0, p.stderr
    assert not marker.exists()


def test_wrapper_refuses_live_deploy_from_worktree(tmp_path):
    _, wt, marker, env = _wrapper_repo(tmp_path, 0)
    p = _run(wt, env, "up", "-d")
    assert p.returncode == 1
    assert "linked worktree" in p.stderr
    assert "fake-docker" not in p.stdout
    assert not marker.exists()


def test_wrapper_refuses_live_deploy_from_dirty_main(tmp_path):
    primary, _, marker, env = _wrapper_repo(tmp_path, 0)
    (primary / "dirty.txt").write_text("not deployed\n")
    p = _run(primary, env, "up", "-d")
    assert p.returncode == 1
    assert "dirty main" in p.stderr
    assert "fake-docker" not in p.stdout
    assert not marker.exists()


def test_wrapper_refuses_live_deploy_from_non_main_primary_branch(tmp_path):
    primary, _, marker, env = _wrapper_repo(tmp_path, 0)
    _git(primary, "branch", "-m", "feature")
    p = _run(primary, env, "up", "-d")
    assert p.returncode == 1
    assert "main is required" in p.stderr
    assert "fake-docker" not in p.stdout
    assert not marker.exists()


def test_wrapper_refuses_live_deploy_from_stale_main(tmp_path):
    primary, _, marker, env = _wrapper_repo(tmp_path, 0)
    (primary / "new.txt").write_text("ahead of origin/main\n")
    _git(primary, "add", "new.txt")
    _git(primary, "commit", "-qm", "ahead")
    p = _run(primary, env, "up", "-d")
    assert p.returncode == 1
    assert "not synchronized with origin/main" in p.stderr
    assert "fake-docker" not in p.stdout
    assert not marker.exists()


def test_wrapper_fetches_and_refuses_when_remote_main_advanced(tmp_path):
    primary, _, marker, env = _wrapper_repo(tmp_path, 0)
    remote = tmp_path / "remote.git"
    other = tmp_path / "other"
    subprocess.run(["git", "clone", "-q", "-b", "main", str(remote), str(other)], check=True)
    _git(other, "config", "user.email", "t@example.com")
    _git(other, "config", "user.name", "T")
    (other / "remote.txt").write_text("new remote main\n")
    _git(other, "add", "remote.txt")
    _git(other, "commit", "-qm", "advance remote")
    _git(other, "push", "-q", "origin", "main")

    p = _run(primary, env, "up", "-d")
    assert p.returncode == 1
    assert "not synchronized with origin/main" in p.stderr
    assert "fake-docker" not in p.stdout
    assert not marker.exists()


def _advance_remote(tmp_path, name="remote.txt"):
    remote = tmp_path / "remote.git"
    other = tmp_path / f"other-{name}"
    subprocess.run(["git", "clone", "-q", "-b", "main", str(remote), str(other)], check=True)
    _git(other, "config", "user.email", "t@example.com")
    _git(other, "config", "user.name", "T")
    (other / name).write_text("new remote main\n")
    _git(other, "add", name)
    _git(other, "commit", "-qm", "advance remote")
    _git(other, "push", "-q", "origin", "main")


def _head(repo):
    return _git(repo, "rev-parse", "HEAD").strip()


def test_pinned_run_deploys_after_remote_main_advanced_mid_run(tmp_path):
    """2026-10-20: a PR merged mid post-pull run and every remaining service was
    refused. A run pinned to the commit it started on keeps deploying it."""
    primary, _, marker, env = _wrapper_repo(tmp_path, 0)
    env["ORION_DEPLOY_PINNED_HEAD"] = _head(primary)
    first = _run(primary, env, "up", "-d")
    assert first.returncode == 0, first.stderr
    _advance_remote(tmp_path)  # a PR merges on GitHub mid-run
    p = _run(primary, env, "up", "-d")
    assert p.returncode == 0, p.stderr
    assert "fake-docker" in p.stdout and marker.exists()
    assert "origin/main advanced" in p.stderr
    # Same situation without the pin is still refused.
    env.pop("ORION_DEPLOY_PINNED_HEAD")
    assert "not synchronized" in _run(primary, env, "up", "-d").stderr


def test_pinned_run_refuses_when_main_moved_off_the_pin(tmp_path):
    primary, _, marker, env = _wrapper_repo(tmp_path, 0)
    env["ORION_DEPLOY_PINNED_HEAD"] = _head(primary)
    _advance_remote(tmp_path)
    _git(primary, "pull", "-q", "--ff-only", "origin", "main")
    p = _run(primary, env, "up", "-d")
    assert p.returncode == 1
    assert "moved during this run" in p.stderr
    assert "fake-docker" not in p.stdout and not marker.exists()


@pytest.mark.parametrize("diverge", [False, True])
def test_pinned_run_refuses_main_ahead_or_diverged(tmp_path, diverge):
    primary, _, marker, env = _wrapper_repo(tmp_path, 0)
    if diverge:
        _advance_remote(tmp_path)
    (primary / "local.txt").write_text("local only\n")
    _git(primary, "add", "local.txt")
    _git(primary, "commit", "-qm", "local commit")
    env["ORION_DEPLOY_PINNED_HEAD"] = _head(primary)
    p = _run(primary, env, "up", "-d")
    assert p.returncode == 1
    assert "not in origin/main's history" in p.stderr
    assert "fake-docker" not in p.stdout and not marker.exists()


def test_pinned_run_still_refuses_dirty_main(tmp_path):
    primary, _, marker, env = _wrapper_repo(tmp_path, 0)
    env["ORION_DEPLOY_PINNED_HEAD"] = _head(primary)
    (primary / "dirty.txt").write_text("not deployed\n")
    p = _run(primary, env, "up", "-d")
    assert p.returncode == 1 and "dirty main" in p.stderr
    assert not marker.exists()


def test_pinned_run_still_refuses_worktree(tmp_path):
    _, wt, marker, env = _wrapper_repo(tmp_path, 0)
    env["ORION_DEPLOY_PINNED_HEAD"] = _head(wt)
    p = _run(wt, env, "up", "-d")
    assert p.returncode == 1 and "linked worktree" in p.stderr
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


def test_ci_required_by_on_data_only_file_is_rejected(ci_repo, capsys):
    (ci_repo / "services" / "orion-sql-db" / "manual_migration_new.sql").write_text(
        "-- ORION-MIGRATION-REQUIRED-BY: orion-x\nUPDATE o SET i = 1;\n")
    assert _ci(ci_repo) == 1
    assert "over-promises" in capsys.readouterr().err


def test_ci_renamed_in_migration_must_declare(ci_repo):
    _git(ci_repo, "mv", "services/orion-sql-db/manual_migration_old.sql",
         "services/orion-sql-db/manual_migration_renamed.sql")
    _git(ci_repo, "commit", "-qm", "rename")
    assert _ci(ci_repo) == 1


def test_ci_rollback_files_are_exempt(ci_repo):
    _add(ci_repo, "manual_migration_new_rollback.sql", "DROP TABLE n;\n")
    assert _ci(ci_repo) == 0


def test_ci_unresolvable_base_is_an_error_not_a_pass(ci_repo):
    assert ci_check.main(["--repo", str(ci_repo), "--base", "no-such-ref"]) == 2


def test_ci_real_corpus_declarations_are_valid():
    assert ci_check.check_files(REPO_ROOT, set()) == []


def test_ci_check_flags_a_bad_or_dangling_apply_with_marker(tmp_path):
    mig = tmp_path / ci_check.MIG_DIR
    mig.mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "real.sh").write_text("#!/bin/sh\n")
    (mig / "manual_migration_ok.sql").write_text("-- ORION-MIGRATION-APPLY-WITH: scripts/real.sh\nselect 1;")
    (mig / "manual_migration_typo.sql").write_text("-- ORION-MIGRATION-APPLY-WITH: psql\nselect 1;")
    (mig / "manual_migration_gone.sql").write_text("-- ORION-MIGRATION-APPLY-WITH: scripts/gone.sh\nselect 1;")
    errors = "\n".join(ci_check.check_files(tmp_path, set()))
    assert "manual_migration_ok.sql" not in errors
    assert "manual_migration_typo.sql" in errors and "APPLY-WITH" in errors
    assert "manual_migration_gone.sql" in errors and "does not exist" in errors


def test_ci_check_real_corpus_apply_with_markers_are_valid():
    assert not [e for e in ci_check.check_files(REPO_ROOT, set()) if "APPLY-WITH" in e or "apply route" in e]
