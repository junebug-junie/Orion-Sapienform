from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from worktree_lib import (  # noqa: E402
    WorktreeLibError,
    all_open_prs,
    dir_size_bytes,
    human_size,
    list_worktrees,
    mergeable_worktrees,
    merged_branch_set,
    parallel_map,
    repo_toplevel,
)


def test_list_worktrees_from_any_worktree(
    repo_with_worktrees: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    primary, merged_wt, unmerged_wt = repo_with_worktrees
    monkeypatch.chdir(merged_wt)
    infos = list_worktrees()
    paths = {str(i.path) for i in infos}
    assert str(primary) in paths
    assert str(merged_wt) in paths
    assert str(unmerged_wt) in paths
    main_entries = [i for i in infos if i.is_main]
    assert len(main_entries) == 1
    assert str(main_entries[0].path) == str(primary)


def test_list_worktrees_raises_outside_any_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    outside = tmp_path / "not-a-repo"
    outside.mkdir()
    monkeypatch.chdir(outside)
    with pytest.raises(WorktreeLibError):
        list_worktrees()


def test_merged_branch_set(
    repo_with_worktrees: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    primary, merged_wt, unmerged_wt = repo_with_worktrees
    monkeypatch.chdir(primary)
    merged = merged_branch_set(base="origin/main")
    assert "feat/merged" in merged
    assert "feat/unmerged" not in merged


def test_merged_branch_set_raises_on_bad_base(
    repo_with_worktrees: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    primary, _, _ = repo_with_worktrees
    monkeypatch.chdir(primary)
    with pytest.raises(WorktreeLibError):
        merged_branch_set(base="origin/does-not-exist")


def test_mergeable_worktrees_excludes_main(
    repo_with_worktrees: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    primary, merged_wt, unmerged_wt = repo_with_worktrees
    monkeypatch.chdir(primary)
    worktrees, merged_set = mergeable_worktrees(base="origin/main")
    paths = {str(w.path) for w in worktrees}
    assert str(primary) not in paths
    assert str(merged_wt) in paths
    assert "feat/merged" in merged_set


def test_repo_toplevel(
    repo_with_worktrees: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    primary, merged_wt, _ = repo_with_worktrees
    monkeypatch.chdir(merged_wt)
    assert repo_toplevel() == str(merged_wt)


def test_dir_size_bytes_real_directory(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("x" * 1000, encoding="utf-8")
    size = dir_size_bytes(tmp_path)
    assert size is not None
    assert size > 0


def test_dir_size_bytes_nonexistent_path_fails_gracefully(tmp_path: Path) -> None:
    assert dir_size_bytes(tmp_path / "does-not-exist") is None


def test_parallel_map_preserves_order() -> None:
    result = parallel_map(lambda x: x * 2, [1, 2, 3, 4, 5])
    assert result == [2, 4, 6, 8, 10]


def test_parallel_map_empty_list() -> None:
    assert parallel_map(lambda x: x, []) == []


def test_all_open_prs_returns_none_when_gh_has_no_remote(
    repo_with_worktrees: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A synthetic repo with no GitHub remote is exactly the failure case
    all_open_prs() must distinguish from 'zero open PRs' -- it should
    return None (unknown), not {} (confirmed empty)."""
    primary, _, _ = repo_with_worktrees
    monkeypatch.chdir(primary)
    assert all_open_prs() is None


def test_human_size_formats() -> None:
    assert human_size(0) == "0.0B"
    assert human_size(1024) == "1.0KB"
    assert human_size(1024 * 1024) == "1.0MB"
    assert human_size(None) == "?"
