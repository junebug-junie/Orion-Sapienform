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
    latest_ref = m._ref_name("feat/my-branch", str(repo)) + "/latest"
    shown = _git(repo, "show", f"{latest_ref}:untracked.py").stdout
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
    latest_ref = m._ref_name("feat/my-branch", str(repo)) + "/latest"
    (repo / "untracked.py").write_text("version 1\n")
    first = m._snapshot_one(_info(repo, branch="feat/my-branch"))
    assert first is not None
    first_latest = _git(repo, "rev-parse", latest_ref).stdout

    (repo / "untracked.py").write_text("version 2\n")
    second = m._snapshot_one(_info(repo, branch="feat/my-branch"))
    assert second is not None
    second_latest = _git(repo, "rev-parse", latest_ref).stdout

    assert first_latest != second_latest
    shown = _git(repo, "show", f"{latest_ref}:untracked.py").stdout
    assert shown == "version 2\n"


def test_gitignored_files_are_not_captured(repo: Path) -> None:
    latest_ref = m._ref_name("feat/my-branch", str(repo)) + "/latest"
    (repo / ".gitignore").write_text("*.secret\n")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-q", "-m", "add gitignore")
    (repo / "leaked.secret").write_text("do not capture me\n")
    (repo / "real.py").write_text("real content\n")

    m._snapshot_one(_info(repo, branch="feat/my-branch"))

    files = _git(repo, "ls-tree", "-r", "--name-only", latest_ref).stdout
    assert "real.py" in files
    assert "leaked.secret" not in files


# --------------------------------------------------------------------------
# pruning
# --------------------------------------------------------------------------


def test_prune_keeps_only_the_newest_n_snapshots(repo: Path) -> None:
    ref_base = m._ref_name("feat/my-branch", str(repo))
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


def test_prune_drops_snapshots_older_than_max_age_regardless_of_count(repo: Path, monkeypatch) -> None:
    """A worktree that goes dirty once and is never touched again must not
    keep that one ref forever -- age-based pruning is independent of the
    count-based rule above."""
    ref_base = m._ref_name("feat/my-branch", str(repo))
    (repo / "untracked.py").write_text("only version\n")
    m._snapshot_one(_info(repo, branch="feat/my-branch"))

    monkeypatch.setattr(m, "_MAX_SNAPSHOT_AGE_SEC", 0.0)
    m._prune_old_snapshots(str(repo), ref_base)

    result = subprocess.run(
        ["git", "-C", str(repo), "for-each-ref", ref_base],
        capture_output=True, text=True,
    )
    timestamped = [
        line for line in result.stdout.splitlines() if not line.rstrip().endswith("/latest")
    ]
    assert timestamped == []


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
    ref = m._ref_name("feat/my thing", "/some/path")
    assert ref.startswith("refs/orion-wip/feat/my_thing-")


def test_ref_name_preserves_slashes_in_branch_names() -> None:
    ref = m._ref_name("feat/tension-driven-outreach", "/some/path")
    assert ref.startswith("refs/orion-wip/feat/tension-driven-outreach-")


def test_ref_name_is_deterministic_for_the_same_path() -> None:
    assert m._ref_name("feat/x", "/some/path") == m._ref_name("feat/x", "/some/path")


def test_ref_name_disambiguates_branches_that_differ_only_by_space(tmp_path) -> None:
    """Code review, 2026-08-18: two worktrees on branches differing only by
    space-vs-underscore (both legal git ref components) must not collide on
    the same ref -- the path hash, not the escaped branch name, is what
    actually guarantees uniqueness."""
    path_a = tmp_path / "worktree_a"
    path_b = tmp_path / "worktree_b"
    ref_a = m._ref_name("feat/my_thing", str(path_a))
    ref_b = m._ref_name("feat/my thing", str(path_b))
    assert ref_a != ref_b


# --------------------------------------------------------------------------
# main() -- lock + marker-only-on-successful-launch
# --------------------------------------------------------------------------


def _patch_marker_and_lock(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(m, "_SCAN_MARKER_PATH", tmp_path / "marker")
    monkeypatch.setattr(m, "_SCAN_LOCK_PATH", tmp_path / "marker.lock")


def test_main_touches_marker_when_popen_launch_succeeds(tmp_path, monkeypatch) -> None:
    _patch_marker_and_lock(monkeypatch, tmp_path)
    monkeypatch.setattr(m.subprocess, "Popen", lambda *a, **k: object())

    m.main()

    assert m._SCAN_MARKER_PATH.exists()


def test_main_does_not_touch_marker_when_popen_raises(tmp_path, monkeypatch) -> None:
    """Code review, 2026-08-18: a launch failure must not silently block the
    next legitimate scan attempt for the full throttle interval -- the
    marker is the signal that a scan actually started, not that one was
    merely attempted."""
    _patch_marker_and_lock(monkeypatch, tmp_path)

    def _raise(*a, **k):
        raise OSError("fork failed")

    monkeypatch.setattr(m.subprocess, "Popen", _raise)

    m.main()

    assert not m._SCAN_MARKER_PATH.exists()


def test_main_releases_the_lock_after_returning(tmp_path, monkeypatch) -> None:
    """A leaked flock would permanently wedge every future scan attempt --
    confirm main() can be called twice in a row without deadlocking."""
    _patch_marker_and_lock(monkeypatch, tmp_path)
    monkeypatch.setattr(m.subprocess, "Popen", lambda *a, **k: object())

    m.main()
    monkeypatch.setattr(m, "_MIN_SCAN_INTERVAL_SEC", 0.0)
    m.main()  # would hang here if the first call's flock were never released

    assert m._SCAN_MARKER_PATH.exists()


def test_main_is_a_noop_when_lock_is_already_held(tmp_path, monkeypatch) -> None:
    """Simulates a concurrent Stop event already inside its own critical
    section -- main() must observe the held lock and return without
    touching the marker or attempting to launch anything."""
    import fcntl

    _patch_marker_and_lock(monkeypatch, tmp_path)
    m._SCAN_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    held = open(m._SCAN_LOCK_PATH, "w")
    fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    launched = []
    monkeypatch.setattr(m.subprocess, "Popen", lambda *a, **k: launched.append(1))

    try:
        m.main()
    finally:
        fcntl.flock(held.fileno(), fcntl.LOCK_UN)
        held.close()

    assert launched == []
    assert not m._SCAN_MARKER_PATH.exists()


# --------------------------------------------------------------------------
# _run_worker_scan -- includes the main/shared checkout
# --------------------------------------------------------------------------


def test_worker_scan_includes_the_main_checkout(monkeypatch) -> None:
    """Code review, 2026-08-18: an earlier draft filtered `is_main` the way
    prune/status tooling correctly does for THEIR purpose, silently leaving
    the highest-risk shared checkout unprotected here. list_worktrees()'s
    own result (including any is_main=True entry) must reach parallel_map
    unfiltered."""
    fake_main = WorktreeInfo(path=Path("/repo"), branch="main", is_main=True)
    fake_linked = WorktreeInfo(path=Path("/repo-feat"), branch="feat/x", is_main=False)
    monkeypatch.setattr(m, "list_worktrees", lambda: [fake_main, fake_linked])

    seen = []
    monkeypatch.setattr(m, "parallel_map", lambda fn, items, **k: seen.extend(items))

    m._run_worker_scan()

    assert fake_main in seen
    assert fake_linked in seen
