from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "install_git_safety_hooks.sh"


def _init_repo(path: Path) -> None:
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)


def _run_installer(target: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["sh", str(SCRIPT), str(target)], capture_output=True, text=True, timeout=30
    )


def test_installs_both_hooks(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    proc = _run_installer(repo)
    assert proc.returncode == 0, proc.stderr
    pre_commit = repo / ".git" / "hooks" / "pre-commit"
    post_merge = repo / ".git" / "hooks" / "post-merge"
    assert pre_commit.exists()
    assert post_merge.exists()
    assert "orion-git-safety-guard" in pre_commit.read_text(encoding="utf-8")
    assert "orion-git-safety-guard" in post_merge.read_text(encoding="utf-8")


def test_both_hooks_executable(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _run_installer(repo)
    import os

    for name in ("pre-commit", "post-merge"):
        hook = repo / ".git" / "hooks" / name
        assert os.access(hook, os.X_OK)


def test_rerun_refreshes_both_without_duplicating_marker(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _run_installer(repo)
    proc = _run_installer(repo)
    assert proc.returncode == 0
    assert "refreshed" in proc.stdout
    pre_commit = repo / ".git" / "hooks" / "pre-commit"
    post_merge = repo / ".git" / "hooks" / "post-merge"
    assert pre_commit.read_text(encoding="utf-8").count("# orion-git-safety-guard") == 1
    assert post_merge.read_text(encoding="utf-8").count("# orion-git-safety-guard") == 1


def test_preserves_existing_foreign_hook_as_backup(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    hooks_dir = repo / ".git" / "hooks"
    hooks_dir.mkdir(exist_ok=True)
    (hooks_dir / "post-merge").write_text("#!/bin/sh\necho unrelated-hook\n", encoding="utf-8")

    proc = _run_installer(repo)
    assert proc.returncode == 0
    backup = hooks_dir / "post-merge.pre-orion-safety.bak"
    assert backup.exists()
    assert "unrelated-hook" in backup.read_text(encoding="utf-8")
    assert "orion-git-safety-guard" in (hooks_dir / "post-merge").read_text(encoding="utf-8")
