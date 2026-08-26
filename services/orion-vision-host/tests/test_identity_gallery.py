from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.identity_gallery import (
    classify_similarity,
    cosine_similarity,
    load_gallery_embedding,
    match_embedding,
    save_gallery_embedding,
)


def test_load_gallery_embedding_returns_none_when_unenrolled(tmp_path):
    """The common, expected state for this feature at ship time -- zero
    real photos of anyone exist yet. Must not raise."""
    assert load_gallery_embedding(str(tmp_path), "juniper") is None


def test_load_gallery_embedding_returns_none_on_corrupt_file(tmp_path):
    (tmp_path / "juniper.json").write_text("not json", encoding="utf-8")
    assert load_gallery_embedding(str(tmp_path), "juniper") is None


def test_save_then_load_round_trips(tmp_path):
    vec = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    save_gallery_embedding(str(tmp_path), "juniper", vec, sample_count=3)

    loaded = load_gallery_embedding(str(tmp_path), "juniper")

    assert loaded is not None
    np.testing.assert_allclose(loaded, vec, atol=1e-6)


def test_gallery_path_sanitizes_subject_name(tmp_path):
    """Subject name reaches the filesystem -- must not allow path traversal
    or arbitrary file writes via a crafted subject string."""
    vec = np.array([1.0, 0.0], dtype=np.float32)
    path = save_gallery_embedding(str(tmp_path), "../../etc/passwd", vec, sample_count=1)
    assert path.parent == tmp_path
    assert ".." not in path.name
    assert "/" not in path.name


def test_cosine_similarity_identical_vectors_is_one():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors_is_negative_one():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(v, -v) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector_does_not_raise_or_nan():
    result = cosine_similarity(np.zeros(3), np.array([1.0, 2.0, 3.0]))
    assert result == 0.0


def test_classify_similarity_three_bands():
    # Hand-computed: match_threshold=0.35, probable_threshold=0.55.
    assert classify_similarity(0.60, match_threshold=0.35, probable_threshold=0.55) == "probable"
    assert classify_similarity(0.55, match_threshold=0.35, probable_threshold=0.55) == "probable"  # boundary, inclusive
    assert classify_similarity(0.40, match_threshold=0.35, probable_threshold=0.55) == "possible"
    assert classify_similarity(0.35, match_threshold=0.35, probable_threshold=0.55) == "possible"  # boundary, inclusive
    assert classify_similarity(0.10, match_threshold=0.35, probable_threshold=0.55) == "unsure"
    assert classify_similarity(-0.20, match_threshold=0.35, probable_threshold=0.55) == "unsure"


def test_match_embedding_no_gallery_returns_unknown_unsure():
    result = match_embedding(
        np.array([1.0, 0.0]), None, subject="juniper", match_threshold=0.35, probable_threshold=0.55
    )
    assert result == {"subject": "unknown", "similarity": None, "state": "unsure", "reason": "not_enrolled"}


def test_match_embedding_strong_match_returns_probable_and_real_subject():
    gallery = np.array([1.0, 0.0, 0.0])
    query = np.array([1.0, 0.0, 0.0])  # identical -> similarity 1.0

    result = match_embedding(
        query, gallery, subject="juniper", match_threshold=0.35, probable_threshold=0.55
    )

    assert result["subject"] == "juniper"
    assert result["state"] == "probable"
    assert result["similarity"] == pytest.approx(1.0)


def test_match_embedding_below_threshold_returns_unknown_not_the_subject_name():
    """A low-confidence read must never leak the enrolled subject's name in
    the returned subject field -- that would defeat the whole point of the
    three-band design (a caller filtering on `subject == "juniper"` would
    otherwise see near-random unsure reads as real matches)."""
    gallery = np.array([1.0, 0.0])
    query = np.array([0.0, 1.0])  # orthogonal -> similarity 0.0

    result = match_embedding(
        query, gallery, subject="juniper", match_threshold=0.35, probable_threshold=0.55
    )

    assert result["subject"] == "unknown"
    assert result["state"] == "unsure"


def test_match_embedding_never_returns_the_raw_embedding():
    """Non-negotiable: the query embedding itself must never appear in the
    returned hypothesis, matched or not."""
    gallery = np.array([1.0, 0.0])
    query = np.array([0.9, 0.1])

    result = match_embedding(
        query, gallery, subject="juniper", match_threshold=0.35, probable_threshold=0.55
    )

    assert "embedding" not in result
    assert not any(isinstance(v, np.ndarray) for v in result.values())
