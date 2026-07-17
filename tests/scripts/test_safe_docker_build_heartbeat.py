"""safe_docker_build.sh must heartbeat the agent board before running a
build, so a concurrent session's `agent_board.py checkin` sees the deploy in
flight -- this is the actual mitigation for the documented incident where a
concurrent agent silently reverted another session's verified fix by running
docker compose from the shared checkout. A fake `docker` binary on PATH
stands in for the real one so this test needs neither a daemon nor a real
service compose file.
"""
from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "safe_docker_build.sh"


def _make_fake_docker(bin_dir: Path) -> None:
    fake_docker = bin_dir / "docker"
    fake_docker.write_text(
        "#!/bin/sh\necho fake-docker \"$@\"\nexit 0\n", encoding="utf-8"
    )
    fake_docker.chmod(fake_docker.stat().st_mode | stat.S_IEXEC)


def _init_worktree_repo(tmp_path: Path, service: str) -> Path:
    """A non-shared checkout (linked worktree) so the wrapper's shared-checkout
    guard doesn't refuse to run -- mirrors how this wrapper is actually used."""
    primary = tmp_path / "primary"
    primary.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=primary, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=primary, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=primary, check=True)
    service_dir = primary / "services" / service
    service_dir.mkdir(parents=True)
    (service_dir / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")
    (primary / ".env").write_text("", encoding="utf-8")
    (service_dir / ".env").write_text("", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=primary, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=primary, check=True)

    worktree = tmp_path / "worktree"
    subprocess.run(
        ["git", "worktree", "add", "-q", str(worktree), "-b", "chore/test-deploy"],
        cwd=primary,
        check=True,
    )
    return worktree


def test_heartbeats_before_running_docker_compose(tmp_path: Path) -> None:
    worktree = _init_worktree_repo(tmp_path, "orion-fake-service")

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    _make_fake_docker(fake_bin)

    board = tmp_path / "agent-board.jsonl"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
        "ORION_AGENT_BOARD_PATH": str(board),
    }

    proc = subprocess.run(
        ["sh", str(SCRIPT), "orion-fake-service", "up", "-d"],
        cwd=worktree,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert "fake-docker" in proc.stdout

    listed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "agent_board.py"), "list", "--all"],
        cwd=worktree,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert listed.returncode == 0, listed.stderr
    assert "orion-fake-service" in listed.stdout
    assert "deploy:orion-fake-service" in listed.stdout


def test_never_fails_the_build_when_board_path_is_unwritable(tmp_path: Path) -> None:
    """The heartbeat is advisory-only -- an agent-board failure must never
    block a real deploy."""
    worktree = _init_worktree_repo(tmp_path, "orion-fake-service")

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    _make_fake_docker(fake_bin)

    unwritable_dir = tmp_path / "no-perms"
    unwritable_dir.mkdir(mode=0o500)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
        "ORION_AGENT_BOARD_PATH": str(unwritable_dir / "agent-board.jsonl"),
    }

    proc = subprocess.run(
        ["sh", str(SCRIPT), "orion-fake-service", "config"],
        cwd=worktree,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert "fake-docker" in proc.stdout
