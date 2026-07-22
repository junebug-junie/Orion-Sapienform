import pytest

from orion.signals.adapters.chat_stance_scoring import (
    cosine_similarity_01,
    score_stance_confidence,
)

_CLEAN_ENFORCEMENT = {
    "enforcement": {
        "fallback_invoked": False,
        "normalized_applied": False,
        "quality_modified": False,
        "semantic_fallback": False,
    },
    "raw": {"enforcement": {"parse_error": None}},
}


def test_score_stance_confidence_clean_is_full_confidence():
    score, reasons = score_stance_confidence(_CLEAN_ENFORCEMENT)
    assert score == pytest.approx(1.0)
    assert reasons == []


def test_score_stance_confidence_penalizes_each_real_flag():
    debug = {
        "enforcement": {
            "fallback_invoked": True,
            "semantic_fallback": True,
            "quality_modified": True,
            "normalized_applied": True,
        },
        "raw": {"enforcement": {"parse_error": "boom"}},
    }
    score, reasons = score_stance_confidence(debug)
    # 1.0 - 0.40 - 0.25 - 0.15 - 0.05 - 0.20 = -0.05 -> clamped to 0.0
    assert score == pytest.approx(0.0)
    assert len(reasons) == 5


def test_score_stance_confidence_missing_enforcement_defaults_clean():
    score, reasons = score_stance_confidence({})
    assert score == pytest.approx(1.0)
    assert reasons == []


def test_cosine_similarity_01_identical_vectors_is_one():
    assert cosine_similarity_01([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_similarity_01_orthogonal_vectors_is_half():
    assert cosine_similarity_01([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.5)


def test_cosine_similarity_01_opposite_vectors_is_zero():
    assert cosine_similarity_01([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "a,b",
    [
        (None, [1.0]),
        ([1.0], None),
        ([], [1.0]),
        ([1.0, 2.0], [1.0]),
        ([0.0, 0.0], [1.0, 1.0]),
    ],
)
def test_cosine_similarity_01_returns_none_when_not_comparable(a, b):
    assert cosine_similarity_01(a, b) is None
