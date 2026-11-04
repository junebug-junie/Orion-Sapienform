"""scripts/sync_fcc_sandbox_after_deploy.sh: refresh Orion's FCC sandbox after a deploy.

A fake `docker` on PATH stands in for the Hub container, so no test ever reaches
the real sandbox. The helper must print one verdict line and always exit 0.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "scripts" / "sync_fcc_sandbox_after_deploy.sh"

# `docker ps` prints $FAKE_HUB (may be empty); `docker exec` logs its args and
# answers with $FAKE_EXEC_OUT / $FAKE_EXEC_RC.
_FAKE_DOCKER = """#!/bin/sh
case "$1" in
  ps) [ -n "$FAKE_HUB" ] && printf '%s\\n' "$FAKE_HUB"; exit 0 ;;
  exec) shift; printf '%s\\n' "$*" >> "$FAKE_DOCKER_LOG"
        [ -n "$FAKE_EXEC_OUT" ] && printf '%s\\n' "$FAKE_EXEC_OUT"
        exit "${FAKE_EXEC_RC:-0}" ;;
esac
exit 99
"""


def _g(*cmd: str, cwd: Path) -> None:
    subprocess.run(["git", "-c", "user.email=t@example.com", "-c", "user.name=T", *cmd],
                   cwd=cwd, check=True, capture_output=True)


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy2(HELPER, repo / "scripts" / HELPER.name)
    _g("init", "-q", "-b", "main", cwd=repo)
    _g("add", "-A", cwd=repo)
    _g("commit", "-qm", "init", cwd=repo)
    remote = tmp_path / "remote.git"
    _g("init", "--bare", "-q", str(remote), cwd=tmp_path)
    _g("remote", "add", "origin", str(remote), cwd=repo)
    _g("push", "-qu", "origin", "main", cwd=repo)
    return repo


def _run(tmp_path: Path, repo: Path, *, hub: str = "orion-athena-hub", out: str = "",
         rc: int = 0, args: tuple[str, ...] = ()) -> tuple[subprocess.CompletedProcess, list[str]]:
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    docker = bindir / "docker"
    docker.write_text(_FAKE_DOCKER)
    docker.chmod(0o755)
    log = tmp_path / "docker-exec.log"
    env = os.environ.copy()
    env.update({
        "PATH": f"{bindir}:{env['PATH']}",
        "FAKE_HUB": hub,
        "FAKE_EXEC_OUT": out,
        "FAKE_EXEC_RC": str(rc),
        "FAKE_DOCKER_LOG": str(log),
    })
    completed = subprocess.run([str(repo / "scripts" / HELPER.name), "--wait-s", "0", *args], env=env,
                               text=True, capture_output=True, check=False)
    return completed, (log.read_text().splitlines() if log.exists() else [])


def test_ok_runs_the_sync_inside_the_hub_container(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    completed, execs = _run(tmp_path, repo, out="result=synced workspace=/mnt/orion-fcc/repo head=abc commits_advanced=3")
    assert completed.returncode == 0
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                          text=True, check=True).stdout.strip()
    assert completed.stdout.strip() == (
        "ok result=synced workspace=/mnt/orion-fcc/repo head=abc commits_advanced=3 "
        f"deploy_head={head[:12]}"
    )
    assert execs == ["orion-athena-hub python3 -m orion.fcc.sandbox_sync --wait-s 0"]


def test_declined_sync_is_reported_failed_with_exit_zero(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    completed, _ = _run(tmp_path, repo, out="result=error_fetch workspace=x", rc=1)
    assert completed.returncode == 0
    assert completed.stdout.strip() == "failed reason=sync_declined rc=1 result=error_fetch workspace=x"


def test_turn_in_flight_reads_as_deferred_not_failed(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    completed, _ = _run(tmp_path, repo, out="result=skipped_turn_in_flight_external workspace=x", rc=1)
    assert completed.stdout.strip() == (
        "skipped reason=turn_in_flight result=skipped_turn_in_flight_external workspace=x"
    )


def test_docker_client_timeout_is_named(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    completed, _ = _run(tmp_path, repo, out="", rc=124)
    assert completed.stdout.strip() == "failed reason=timeout rc=124 container=orion-athena-hub"


def test_second_pending_retry_is_skipped(tmp_path: Path) -> None:
    """Several pulls during one long turn must not stack 90-minute waiters."""
    repo = _repo(tmp_path)
    log = tmp_path / "rebuild.log"
    holder = subprocess.Popen(["flock", str(log) + ".sandbox-retry.lock", "sleep", "5"])
    try:
        for _ in range(100):
            if Path(str(log) + ".sandbox-retry.lock").exists():
                break
            __import__("time").sleep(0.05)
        __import__("time").sleep(0.2)
        completed, execs = _run(tmp_path, repo, out="result=synced", args=("--log-to", str(log)))
    finally:
        holder.kill()
        holder.wait()
    assert completed.stdout.strip() == "skipped reason=retry_already_pending"
    assert execs == []


def test_image_without_the_cli_reads_as_no_verdict(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    completed, _ = _run(tmp_path, repo, out="", rc=0)
    assert completed.returncode == 0
    assert completed.stdout.strip() == "failed reason=no_verdict rc=0 container=orion-athena-hub"


def test_no_hub_container_skips(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    completed, execs = _run(tmp_path, repo, hub="")
    assert completed.stdout.strip() == "skipped reason=hub_not_running"
    assert execs == []


def test_dirty_checkout_skips(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "dirty.txt").write_text("x")
    completed, execs = _run(tmp_path, repo)
    assert completed.stdout.strip() == "skipped reason=dirty_checkout"
    assert execs == []


def test_main_ahead_of_origin_skips(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "ahead.txt").write_text("x")
    _g("add", "ahead.txt", cwd=repo)
    _g("commit", "-qm", "ahead", cwd=repo)
    completed, execs = _run(tmp_path, repo)
    assert completed.stdout.strip() == "skipped reason=main_not_origin_main"
    assert execs == []


def test_linked_worktree_skips(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    wt = tmp_path / "wt"
    _g("worktree", "add", "-q", "--detach", str(wt), cwd=repo)
    completed, execs = _run(tmp_path, wt)
    assert completed.stdout.strip() == "skipped reason=linked_worktree"
    assert execs == []


def test_dry_run_never_execs(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    completed, execs = _run(tmp_path, repo, args=("--dry-run",))
    assert completed.stdout.strip().startswith("skipped reason=dry_run")
    assert execs == []


def test_log_to_appends_a_retry_line_to_the_rebuild_log(tmp_path: Path) -> None:
    """The detached retry has no caller to read stdout; it logs for itself."""
    repo = _repo(tmp_path)
    log = tmp_path / "rebuild.log"
    completed, _ = _run(tmp_path, repo, out="result=synced head=abc", args=("--log-to", str(log)))
    assert completed.returncode == 0
    lines = log.read_text().splitlines()
    assert len(lines) == 1
    assert " SANDBOX_SYNC retry ok result=synced head=abc deploy_head=" in lines[0]
