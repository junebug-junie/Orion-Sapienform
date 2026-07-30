"""Unit tests for ``orion/structural_mass/git_delta.py::git_churn_delta`` -- built
against a real, throwaway git repo (not this repo's own history) so results are
deterministic and independent of whatever commits happen to exist locally."""

from __future__ import annotations

import subprocess
from pathlib import Path

from orion.structural_mass.git_delta import git_churn_delta


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _init_repo(repo: Path) -> None:
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def test_git_churn_delta_zero_when_prev_equals_head(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "a.txt").write_text("hello\n")
    sha = _commit(repo, "initial")

    delta = git_churn_delta(sha, sha, repo_path=repo)

    assert delta.commit_count == 0
    assert delta.files_added == 0
    assert delta.files_deleted == 0
    assert delta.files_modified == 0
    assert delta.lines_added == 0
    assert delta.lines_removed == 0
    assert delta.files_changed == 0
    assert delta.lines_changed == 0


def test_git_churn_delta_counts_added_deleted_modified_files(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "keep.txt").write_text("line1\nline2\n")
    (repo / "remove.txt").write_text("gone\n")
    prev_sha = _commit(repo, "initial")

    (repo / "keep.txt").write_text("line1\nline2-changed\nline3\n")
    (repo / "remove.txt").unlink()
    (repo / "new.txt").write_text("brand new\n")
    head_sha = _commit(repo, "second")

    delta = git_churn_delta(prev_sha, head_sha, repo_path=repo)

    assert delta.commit_count == 1
    assert delta.files_added == 1  # new.txt
    assert delta.files_deleted == 1  # remove.txt
    assert delta.files_modified == 1  # keep.txt
    assert delta.files_changed == 3
    # keep.txt: -1 line2 +1 line2-changed +1 line3 => 2 added, 1 removed
    # remove.txt: -1 line
    # new.txt: +1 line
    assert delta.lines_added == 3
    assert delta.lines_removed == 2
    assert delta.lines_changed == 5


def test_git_churn_delta_counts_multiple_commits(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "a.txt").write_text("v1\n")
    prev_sha = _commit(repo, "c1")

    (repo / "a.txt").write_text("v2\n")
    _commit(repo, "c2")
    (repo / "b.txt").write_text("new\n")
    head_sha = _commit(repo, "c3")

    delta = git_churn_delta(prev_sha, head_sha, repo_path=repo)

    assert delta.commit_count == 2
    assert delta.files_added == 1
    assert delta.files_modified == 1


def test_git_churn_delta_counts_pure_rename_as_modified_not_add_plus_delete(
    tmp_path: Path,
) -> None:
    """`git diff --find-renames --name-status` emits a single `R100\told\tnew`
    line for a rename with unchanged content (two path columns, not one) --
    must count as one "modified" file, not one added + one deleted. Without
    `--find-renames` this repo's own git 2.43.0 build reports a plain rename as
    an unrelated A+D pair by default -- confirmed live -- which is exactly the
    miscount `git_churn_delta` passes `--find-renames` to avoid."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "old.txt").write_text("line1\nline2\nline3\n")
    prev_sha = _commit(repo, "initial")

    _git(repo, "mv", "old.txt", "new.txt")
    head_sha = _commit(repo, "rename")

    delta = git_churn_delta(prev_sha, head_sha, repo_path=repo)

    assert delta.files_added == 0
    assert delta.files_deleted == 0
    assert delta.files_modified == 1
    assert delta.files_changed == 1
    assert delta.lines_added == 0
    assert delta.lines_removed == 0


def test_git_churn_delta_skips_binary_files_in_line_counts(tmp_path: Path) -> None:
    """`git diff --numstat` reports `-\t-\tpath` for binary files -- must not
    raise trying to `int("-")`, and must not contribute to line counts, while
    still counting the file itself in the name-status buckets."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "placeholder.txt").write_text("unrelated\n")
    prev_sha = _commit(repo, "initial")

    (repo / "image.bin").write_bytes(b"\x00\x01\x02\xff\xfe")
    head_sha = _commit(repo, "add binary")

    delta = git_churn_delta(prev_sha, head_sha, repo_path=repo)

    assert delta.files_added == 1
    assert delta.lines_added == 0
    assert delta.lines_removed == 0


def test_git_churn_delta_skips_no_op_ticks_between_calls(tmp_path: Path) -> None:
    """Mirrors the design spec's tick-driven, self-healing-by-construction shape:
    a caller comparing the same last-seen SHA across two no-op checks should get
    an explicit zero delta both times, not an error or a stale value."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "a.txt").write_text("v1\n")
    sha = _commit(repo, "c1")

    first = git_churn_delta(sha, sha, repo_path=repo)
    second = git_churn_delta(sha, sha, repo_path=repo)

    assert first == second
    assert first.lines_changed == 0
