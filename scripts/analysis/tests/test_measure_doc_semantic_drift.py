from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.analysis.measure_doc_semantic_drift as msd  # noqa: E402
from scripts.analysis.measure_doc_semantic_drift import (  # noqa: E402
    DriftSample,
    _cosine_similarity,
    conventional_commit_prefix,
    sample_truncated,
)


def test_conventional_commit_prefix_matches_real_recognized_types() -> None:
    assert conventional_commit_prefix("docs: record PR link") == "docs"
    assert conventional_commit_prefix("fix(substrate-runtime): correct claim") == "fix"
    assert conventional_commit_prefix("feat(cocreation): ship the thing") == "feat"
    assert conventional_commit_prefix("chore(metacog): flip flags") == "chore"


def test_conventional_commit_prefix_none_for_unrecognized_subject() -> None:
    assert conventional_commit_prefix("Merge pull request #1552 from junebug-junie/feat/x") is None
    assert conventional_commit_prefix("bump version") is None


def test_cosine_similarity_identical_vectors_is_one() -> None:
    v = [1.0, 2.0, 3.0]
    assert _cosine_similarity(v, v) == 1.0


def test_cosine_similarity_orthogonal_vectors_is_zero() -> None:
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_similarity_zero_vector_returns_zero_not_nan() -> None:
    # A truncated-to-empty text (real edge case: a brand new file, or a
    # file deleted in this diff) must not raise a ZeroDivisionError.
    assert _cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0
    assert _cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0


def _sample(before_chars: int, after_chars: int, *, likely_truncated: bool) -> DriftSample:
    return DriftSample(
        sha="deadbeef",
        path="some/README.md",
        shortstat="1 insertion(+)",
        expected="real",
        commit_subject="docs: test",
        commit_prefix="docs",
        before_len_chars=before_chars,
        after_len_chars=after_chars,
        embedding_diff=0.0,
        likely_truncated=likely_truncated,
    )


def test_drift_sample_stores_the_real_measured_truncation_flag() -> None:
    # likely_truncated is a real, live-measured fact passed in from the
    # container's own tokenizer (see sample_truncated below) -- confirmed
    # live 2026-08-11 for services/orion-substrate-runtime/README.md
    # (71,795 chars before / 74,482 after, real token count over the
    # tokenizer's real 512-token limit), not a char-count approximation.
    # This test just pins that DriftSample stores what it's given.
    assert _sample(71_795, 74_482, likely_truncated=True).likely_truncated is True
    assert _sample(500, 520, likely_truncated=False).likely_truncated is False


def test_sample_truncated_true_if_either_side_is_truncated() -> None:
    # A truncated "before" that happens to share its leading content with a
    # truncated "after" produces a near-zero embedding_diff regardless of
    # what changed past the truncation point -- either side truncated must
    # flag the whole sample, not just both sides ("or", not "and").
    assert sample_truncated(True, False) is True
    assert sample_truncated(False, True) is True
    assert sample_truncated(True, True) is True
    assert sample_truncated(False, False) is False


def _run_git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_git_diff_hunks_extracts_only_changed_lines_from_a_real_commit(
    tmp_path: Path, monkeypatch
) -> None:
    """Real git repo, real commits -- not a mocked diff. Confirms
    _git_diff_hunks extracts exactly the removed/added content lines, not
    file headers or hunk markers, and not unchanged surrounding lines."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _run_git(["init", "-q"], repo)
    _run_git(["config", "user.email", "test@test.local"], repo)
    _run_git(["config", "user.name", "Test"], repo)

    doc = repo / "README.md"
    doc.write_text("line one\nline two\nline three\n", encoding="utf-8")
    _run_git(["add", "README.md"], repo)
    _run_git(["commit", "-q", "-m", "initial"], repo)

    doc.write_text("line one\nline two CHANGED\nline three\nline four NEW\n", encoding="utf-8")
    _run_git(["add", "README.md"], repo)
    _run_git(["commit", "-q", "-m", "docs: real edit"], repo)

    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()

    monkeypatch.setattr(msd, "REPO_ROOT", repo)
    removed, added = msd._git_diff_hunks(sha, "README.md")

    assert removed == "line two"
    assert "line two CHANGED" in added
    assert "line four NEW" in added
    # Unchanged context lines must not leak into either side.
    assert "line one" not in removed and "line one" not in added
    assert "line three" not in removed and "line three" not in added


def test_git_diff_hunks_pure_addition_has_empty_removed_text(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run_git(["init", "-q"], repo)
    _run_git(["config", "user.email", "test@test.local"], repo)
    _run_git(["config", "user.name", "Test"], repo)

    doc = repo / "README.md"
    doc.write_text("existing content\n", encoding="utf-8")
    _run_git(["add", "README.md"], repo)
    _run_git(["commit", "-q", "-m", "initial"], repo)

    doc.write_text("existing content\nbrand new paragraph appended\n", encoding="utf-8")
    _run_git(["add", "README.md"], repo)
    _run_git(["commit", "-q", "-m", "docs: append"], repo)

    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()

    monkeypatch.setattr(msd, "REPO_ROOT", repo)
    removed, added = msd._git_diff_hunks(sha, "README.md")

    assert removed == ""
    assert added == "brand new paragraph appended"


def test_git_diff_hunks_pure_deletion_has_empty_added_text(
    tmp_path: Path, monkeypatch
) -> None:
    """Symmetric case to the pure-addition test above -- a line removed
    with nothing added back must yield an empty added_text, not a crash or
    leftover content."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _run_git(["init", "-q"], repo)
    _run_git(["config", "user.email", "test@test.local"], repo)
    _run_git(["config", "user.name", "Test"], repo)

    doc = repo / "README.md"
    doc.write_text("existing content\nstale paragraph to remove\n", encoding="utf-8")
    _run_git(["add", "README.md"], repo)
    _run_git(["commit", "-q", "-m", "initial"], repo)

    doc.write_text("existing content\n", encoding="utf-8")
    _run_git(["add", "README.md"], repo)
    _run_git(["commit", "-q", "-m", "docs: remove stale paragraph"], repo)

    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()

    monkeypatch.setattr(msd, "REPO_ROOT", repo)
    removed, added = msd._git_diff_hunks(sha, "README.md")

    assert removed == "stale paragraph to remove"
    assert added == ""
