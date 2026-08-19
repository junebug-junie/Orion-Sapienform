"""Real git repos, not mocks -- this module is fundamentally git-plumbing
(write-tree/commit-tree/update-ref), and the one real bug found while
building it (GIT_INDEX_FILE pointing at a pre-created empty file reads as a
corrupt index, not a fresh one) would never have surfaced from a mocked
subprocess.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "hooks"))
sys.path.insert(0, str(ROOT / "scripts"))

from worktree_lib import WorktreeInfo  # noqa: E402

import stop_worktree_wip_snapshot as m  # noqa: E402


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    )


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """A real, minimal git repo -- one commit, a clean working tree."""
    path = tmp_path / "repo"
    path.mkdir()
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "Test")
    (path / "README.md").write_text("hello\n")
    _git(path, "add", "README.md")
    _git(path, "commit", "-q", "-m", "initial")
    return path


def _info(path: Path, branch: str = "feat/test") -> WorktreeInfo:
    return WorktreeInfo(path=path, branch=branch, is_main=False)


# --------------------------------------------------------------------------
# _is_dirty
# --------------------------------------------------------------------------


def test_clean_repo_is_not_dirty(repo: Path) -> None:
    assert m._is_dirty(str(repo)) is False


def test_untracked_file_makes_repo_dirty(repo: Path) -> None:
    (repo / "new_file.py").write_text("x = 1\n")
    assert m._is_dirty(str(repo)) is True


def test_modified_tracked_file_makes_repo_dirty(repo: Path) -> None:
    (repo / "README.md").write_text("changed\n")
    assert m._is_dirty(str(repo)) is True


# --------------------------------------------------------------------------
# _snapshot_one -- the real end-to-end plumbing sequence
# --------------------------------------------------------------------------


def test_clean_worktree_produces_no_snapshot(repo: Path) -> None:
    assert m._snapshot_one(_info(repo)) is None
    result = subprocess.run(
        ["git", "-C", str(repo), "for-each-ref", "refs/orion-wip"],
        capture_output=True, text=True,
    )
    assert result.stdout.strip() == ""


def test_dirty_worktree_gets_a_real_snapshot_with_correct_content(repo: Path) -> None:
    (repo / "untracked.py").write_text("real content\n")

    outcome = m._snapshot_one(_info(repo, branch="feat/my-branch"))

    assert outcome is not None and "snapshotted" in outcome
    shown = _git(repo, "show", "refs/orion-wip/feat/my-branch/latest:untracked.py").stdout
    assert shown == "real content\n"


def test_snapshot_does_not_touch_real_index_or_working_tree(repo: Path) -> None:
    (repo / "untracked.py").write_text("content\n")
    before = _git(repo, "status", "--porcelain").stdout

    m._snapshot_one(_info(repo))

    after = _git(repo, "status", "--porcelain").stdout
    assert before == after
    # Nothing staged -- the real index was never touched.
    assert _git(repo, "diff", "--cached", "--stat").stdout == ""


def test_snapshot_never_creates_a_branch_or_moves_head(repo: Path) -> None:
    (repo / "untracked.py").write_text("content\n")
    head_before = _git(repo, "rev-parse", "HEAD").stdout

    m._snapshot_one(_info(repo, branch="feat/my-branch"))

    head_after = _git(repo, "rev-parse", "HEAD").stdout
    assert head_before == head_after
    branches = _git(repo, "branch", "--list").stdout
    assert "orion-wip" not in branches


def test_identical_dirty_state_deduplicates_across_calls(repo: Path) -> None:
    (repo / "untracked.py").write_text("same content\n")

    first = m._snapshot_one(_info(repo))
    assert first is not None

    second = m._snapshot_one(_info(repo))
    assert second is None  # no new snapshot -- tree is identical to the last one


def test_changed_dirty_state_produces_a_new_snapshot(repo: Path) -> None:
    (repo / "untracked.py").write_text("version 1\n")
    first = m._snapshot_one(_info(repo, branch="feat/my-branch"))
    assert first is not None
    first_latest = _git(repo, "rev-parse", "refs/orion-wip/feat/my-branch/latest").stdout

    (repo / "untracked.py").write_text("version 2\n")
    second = m._snapshot_one(_info(repo, branch="feat/my-branch"))
    assert second is not None
    second_latest = _git(repo, "rev-parse", "refs/orion-wip/feat/my-branch/latest").stdout

    assert first_latest != second_latest
    shown = _git(repo, "show", "refs/orion-wip/feat/my-branch/latest:untracked.py").stdout
    assert shown == "version 2\n"


def test_gitignored_files_are_not_captured(repo: Path) -> None:
    (repo / ".gitignore").write_text("*.secret\n")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-q", "-m", "add gitignore")
    (repo / "leaked.secret").write_text("do not capture me\n")
    (repo / "real.py").write_text("real content\n")

    m._snapshot_one(_info(repo, branch="feat/my-branch"))

    files = _git(repo, "ls-tree", "-r", "--name-only", "refs/orion-wip/feat/my-branch/latest").stdout
    assert "real.py" in files
    assert "leaked.secret" not in files


# --------------------------------------------------------------------------
# pruning
# --------------------------------------------------------------------------


def test_prune_keeps_only_the_newest_n_snapshots(repo: Path) -> None:
    ref_base = "refs/orion-wip/feat/my-branch"
    for i in range(m._MAX_SNAPSHOTS_PER_WORKTREE + 2):
        (repo / "untracked.py").write_text(f"version {i}\n")
        m._snapshot_one(_info(repo, branch="feat/my-branch"))
        time.sleep(1.1)  # ref timestamps are second-granularity via int(time.time())

    result = subprocess.run(
        ["git", "-C", str(repo), "for-each-ref", ref_base],
        capture_output=True, text=True,
    )
    timestamped = [
        line for line in result.stdout.splitlines() if not line.rstrip().endswith("/latest")
    ]
    assert len(timestamped) == m._MAX_SNAPSHOTS_PER_WORKTREE


# --------------------------------------------------------------------------
# throttle
# --------------------------------------------------------------------------


def test_should_scan_now_true_when_no_marker_exists(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(m, "_SCAN_MARKER_PATH", tmp_path / "no_such_marker")
    assert m._should_scan_now() is True


def test_should_scan_now_false_immediately_after_touch(tmp_path, monkeypatch) -> None:
    marker = tmp_path / "marker"
    monkeypatch.setattr(m, "_SCAN_MARKER_PATH", marker)
    m._touch_scan_marker()
    assert m._should_scan_now() is False


def test_should_scan_now_true_after_interval_elapses(tmp_path, monkeypatch) -> None:
    marker = tmp_path / "marker"
    monkeypatch.setattr(m, "_SCAN_MARKER_PATH", marker)
    monkeypatch.setattr(m, "_MIN_SCAN_INTERVAL_SEC", 0.05)
    m._touch_scan_marker()
    time.sleep(0.1)
    assert m._should_scan_now() is True


# --------------------------------------------------------------------------
# _ref_name
# --------------------------------------------------------------------------


def test_ref_name_replaces_spaces() -> None:
    assert m._ref_name("feat/my thing") == "refs/orion-wip/feat/my_thing"


def test_ref_name_preserves_slashes_in_branch_names() -> None:
    assert m._ref_name("feat/tension-driven-outreach") == "refs/orion-wip/feat/tension-driven-outreach"
