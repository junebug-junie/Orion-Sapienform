from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import self_study_enrichment_hook as hook  # noqa: E402


def test_is_qualifying_path_matches_scan_surface():
    assert hook.is_qualifying_path("services/orion-hub/app/main.py")
    assert hook.is_qualifying_path("orion/bus/channels.yaml")
    assert hook.is_qualifying_path("orion/schemas/registry.py")
    assert hook.is_qualifying_path("orion/cognition/verbs/foo.yaml")


def test_is_qualifying_path_rejects_unrelated_paths():
    assert not hook.is_qualifying_path("docs/superpowers/pr-reports/foo.md")
    assert not hook.is_qualifying_path("scripts/some_unrelated_script.py")
    assert not hook.is_qualifying_path("README.md")


def test_qualifying_paths_filters_list():
    paths = [
        "services/orion-hub/app/main.py",
        "docs/foo.md",
        "orion/schemas/registry.py",
    ]
    assert hook.qualifying_paths(paths) == (
        "services/orion-hub/app/main.py",
        "orion/schemas/registry.py",
    )


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    return repo


def _commit(repo: Path, rel_path: str, content: str) -> str:
    path = repo / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", f"commit {rel_path}"], cwd=repo, check=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def test_changed_paths_real_git_repo(tmp_path):
    repo = _init_repo(tmp_path)
    sha1 = _commit(repo, "README.md", "hello")
    sha2 = _commit(repo, "services/orion-foo/app/main.py", "print(1)")
    paths = hook.changed_paths(sha1, sha2, repo)
    assert "services/orion-foo/app/main.py" in paths


def test_changed_paths_noop_when_same_sha(tmp_path):
    repo = _init_repo(tmp_path)
    sha1 = _commit(repo, "README.md", "hello")
    assert hook.changed_paths(sha1, sha1, repo) == []


def test_state_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("SELF_STUDY_ENRICHMENT_STATE_PATH", str(tmp_path / "state.json"))
    assert hook.read_last_sha(tmp_path) is None
    hook.write_last_sha(tmp_path, "abc123")
    assert hook.read_last_sha(tmp_path) == "abc123"


def test_rate_limit_ok_respects_ceiling(tmp_path, monkeypatch):
    monkeypatch.setenv("SELF_STUDY_ENRICHMENT_MAX_PER_DAY", "2")
    monkeypatch.chdir(tmp_path)
    assert hook._rate_limit_ok(tmp_path) is True
    assert hook._rate_limit_ok(tmp_path) is True
    assert hook._rate_limit_ok(tmp_path) is False


def test_rate_limit_disabled_when_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("SELF_STUDY_ENRICHMENT_MAX_PER_DAY", "0")
    assert hook._rate_limit_ok(tmp_path) is False


def test_main_subprocess_invocation_imports_orion_without_pythonpath(tmp_path):
    """Regression test for a real bug found live 2026-08-12: main()'s local
    `from orion.structural_mass.git_delta import git_churn_delta` import
    silently failed with ModuleNotFoundError on every real invocation,
    because the git hook invokes this script as `python3 <this file's
    path>` (see scripts/git_hooks/post-commit) -- which puts the script's
    OWN directory (scripts/) on sys.path[0], not the repo root, and is
    unaffected by the hook's `cd "$REPO_ROOT"` (that only changes cwd, not
    Python's script-relative sys.path[0]). The outer `except Exception` in
    __main__ swallowed it silently (main() always exits 0), so zero
    enrichment requests were ever published across 4+ hours of the
    consumer service running before this was caught.

    No existing test in this file caught it: every other test imports the
    hook module directly inside the pytest process, whose sys.path is
    already repo-root-augmented by pytest itself -- never as a real
    subprocess the way the git hook actually invokes it. This test
    deliberately runs the real script file via subprocess, with PYTHONPATH
    stripped from the environment, to reproduce the exact invocation shape
    that broke in production.

    Isolated to a disposable temp repo (not the real checkout) with a
    symlinked `orion/` package, so `git rev-parse --show-toplevel` from cwd
    resolves to a directory that genuinely contains `orion/` -- exercising
    the fix's actual assumption (repo-being-scanned == repo-containing-
    orion/, always true for the real git hook) -- without writing the real
    checkout's `.orion/` rate-limit counter file as a side effect.
    """
    real_repo_root = Path(__file__).resolve().parents[1]
    assert (real_repo_root / "orion" / "__init__.py").exists()  # sanity: found the right root

    repo = _init_repo(tmp_path)
    (repo / "orion").symlink_to(real_repo_root / "orion")
    _commit(repo, "README.md", "hello")
    _commit(repo, "services/orion-foo/app/main.py", "print(1)")  # qualifying path

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["SELF_STUDY_ENRICHMENT_STATE_PATH"] = str(tmp_path / "state.json")
    env.pop("ORION_BUS_URL", None)  # force the post-import "skip publish" branch

    result = subprocess.run(
        [sys.executable, str(real_repo_root / "scripts" / "self_study_enrichment_hook.py")],
        cwd=str(repo),
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "No module named 'orion'" not in result.stderr
    # Reaching this branch (in main(), after the fixed import line and after
    # git_churn_delta + the rate-limit check both ran) is only possible if
    # the import actually succeeded.
    assert "ORION_BUS_URL unset, skipping publish" in result.stderr
