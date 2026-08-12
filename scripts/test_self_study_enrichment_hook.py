from __future__ import annotations

import json
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
