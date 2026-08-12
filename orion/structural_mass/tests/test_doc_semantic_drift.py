"""Unit tests for ``orion/structural_mass/doc_semantic_drift.py`` -- built
against real, throwaway git repos (not this repo's own history) so results
are deterministic and independent of whatever commits happen to exist
locally."""

from __future__ import annotations

import subprocess
from pathlib import Path

from orion.structural_mass.doc_semantic_drift import (
    DocHunkChange,
    changed_doc_files_with_status,
    conventional_commit_prefix,
    diff_hunks,
    doc_semantic_drift_changes,
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _init_repo(repo: Path) -> None:
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@test.local")
    _git(repo, "config", "user.name", "Test")


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def test_conventional_commit_prefix_matches_real_recognized_types() -> None:
    assert conventional_commit_prefix("docs: record PR link") == "docs"
    assert conventional_commit_prefix("fix(substrate-runtime): correct claim") == "fix"


def test_conventional_commit_prefix_none_for_unrecognized_subject() -> None:
    assert conventional_commit_prefix("Merge pull request #1552 from x/y") is None


def test_changed_doc_files_only_matches_markdown(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    prev = _commit(repo, "initial")

    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    (repo / "notes.txt").write_text("not markdown\n", encoding="utf-8")
    head = _commit(repo, "docs: add readme and notes")

    files = [path for _kind, path in changed_doc_files_with_status(prev, head, repo)]
    assert files == ["README.md"]


def test_changed_doc_files_excludes_deleted_docs(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "STALE.md").write_text("will be deleted\n", encoding="utf-8")
    prev = _commit(repo, "initial")

    (repo / "STALE.md").unlink()
    head = _commit(repo, "chore: remove stale doc")

    # A deleted doc has no "after" state to score drift against -- must not
    # appear in the changed-files list at all.
    assert changed_doc_files_with_status(prev, head, repo) == []


def test_diff_hunks_extracts_only_changed_lines(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "README.md").write_text("line one\nline two\nline three\n", encoding="utf-8")
    prev = _commit(repo, "initial")

    (repo / "README.md").write_text(
        "line one\nline two CHANGED\nline three\nline four NEW\n", encoding="utf-8"
    )
    head = _commit(repo, "docs: real edit")

    removed, added = diff_hunks(prev, head, "README.md", repo)
    assert removed == "line two"
    assert "line two CHANGED" in added
    assert "line four NEW" in added
    assert "line one" not in removed and "line one" not in added


def test_doc_semantic_drift_changes_covers_the_whole_range_not_just_the_last_commit(
    tmp_path: Path,
) -> None:
    """Regression guard for a real bug caught before this ever shipped:
    diff_hunks originally hardcoded `head~1` (the last commit's own parent)
    even when doc_semantic_drift_changes was asked to cover a wider
    (prev_sha, head_sha] range spanning multiple commits -- silently
    dropping any doc edit made in an earlier commit within that same poll
    window. This test's edit happens two commits before `head`, so it would
    NOT show up if the range logic regressed back to a single-commit diff."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "README.md").write_text("original content\n", encoding="utf-8")
    prev = _commit(repo, "initial")

    # Real edit lands in the *first* of two commits in this poll window.
    (repo / "README.md").write_text("original content\nreal edit here\n", encoding="utf-8")
    _commit(repo, "docs: real edit")

    # A second, unrelated commit that doesn't touch README.md -- this is
    # the commit doc_semantic_drift_changes actually receives as head_sha.
    (repo / "other.py").write_text("x = 1\n", encoding="utf-8")
    head = _commit(repo, "chore: unrelated change")

    changes = doc_semantic_drift_changes(prev, head, repo)
    assert len(changes) == 1
    change = changes[0]
    assert change.path == "README.md"
    assert "real edit here" in change.hunk_added
    assert change.sha == head  # attributed to the tick's real, current commit


def test_doc_semantic_drift_changes_carries_commit_prefix(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    prev = _commit(repo, "initial")

    (repo / "README.md").write_text("new content\n", encoding="utf-8")
    head = _commit(repo, "docs: a real change")

    changes = doc_semantic_drift_changes(prev, head, repo)
    assert len(changes) == 1
    assert changes[0].commit_prefix == "docs"
    assert isinstance(changes[0], DocHunkChange)


def test_changed_doc_files_with_status_reports_added_vs_modified(tmp_path: Path) -> None:
    """The real reason this exists: ``diff_scoped_embedding_diff=None`` has
    two structurally different causes, and only git's own status letter
    tells them apart. A newly-added doc has no "before" text (cosine
    genuinely undefined -- expected); a modified doc scoring None means a
    real embedding request failed (an alarm). Confirmed live 2026-08-12 that
    both were publishing as an indistinguishable bare None."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "existing.md").write_text("original line\n")
    prev = _commit(repo, "docs: seed")

    (repo / "existing.md").write_text("changed line\n")
    (repo / "brand_new.md").write_text("a fresh doc\n")
    head = _commit(repo, "docs: edit one, add one")

    entries = dict((path, kind) for kind, path in changed_doc_files_with_status(prev, head, repo))
    assert entries == {"existing.md": "modified", "brand_new.md": "added"}


def test_changed_doc_files_with_status_excludes_deleted_docs(tmp_path: Path) -> None:
    """--diff-filter=ACMR still excludes deletions -- a deleted doc has no
    "after" state to score drift against. The schema declares "deleted" for
    exhaustiveness, but the producer must not emit it today."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "doomed.md").write_text("content\n")
    prev = _commit(repo, "docs: seed")

    (repo / "doomed.md").unlink()
    head = _commit(repo, "docs: remove")

    assert changed_doc_files_with_status(prev, head, repo) == []


def test_doc_semantic_drift_changes_carries_change_kind(tmp_path: Path) -> None:
    """End-to-end through the real producer-facing entry point -- the kind
    has to survive all the way onto DocHunkChange, or the payload field is
    always None and the disambiguation never happens."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "existing.md").write_text("original\n")
    prev = _commit(repo, "docs: seed")

    (repo / "existing.md").write_text("updated\n")
    (repo / "fresh.md").write_text("new content\n")
    head = _commit(repo, "docs: mixed change")

    kinds = {c.path: c.change_kind for c in doc_semantic_drift_changes(prev, head, repo)}
    assert kinds == {"existing.md": "modified", "fresh.md": "added"}


def test_renamed_doc_reads_as_added_matching_its_hunk_shape(tmp_path: Path) -> None:
    """No -M/-C is passed to ``git diff``, so a rename presents as a full
    add at the new path -- and the hunk text really does look like a pure
    addition. Reporting "added" matches what actually gets scored rather
    than what git would call the operation with rename detection on."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "old_name.md").write_text("stable content here\n")
    prev = _commit(repo, "docs: seed")

    _git(repo, "mv", "old_name.md", "new_name.md")
    head = _commit(repo, "docs: rename")

    entries = dict((path, kind) for kind, path in changed_doc_files_with_status(prev, head, repo))
    assert entries == {"new_name.md": "added"}
