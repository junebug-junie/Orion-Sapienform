from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
