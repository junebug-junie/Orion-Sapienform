"""Unit tests for P2's perceptual prediction-error signal (docs/superpowers/
specs/2026-08-12-perception-frontier-design.md): ``surprise = 1 -
cos(frame_embedding, EWMA_embedding)`` per camera stream, plus the
per-stream vector-EWMA baseline it maintains.

Every non-trivial expected value below is hand-computed from simple
orthonormal-ish vectors (not just "does it run") -- see each test's own
comment for the arithmetic.
"""

from __future__ import annotations

import math

from orion.substrate.prediction_error import (
    PerceptionEmbeddingBaseline,
    perception_prediction_error,
)


# --- cold start -------------------------------------------------------------


def test_cold_start_seeds_baseline_and_reports_no_score() -> None:
    baseline = PerceptionEmbeddingBaseline()  # n=0
    result = perception_prediction_error([1.0, 0.0], baseline)
    assert result.score is None
    assert result.baseline.embedding_ewma == (1.0, 0.0)
    assert result.baseline.n == 1


def test_empty_embedding_reports_no_score_and_leaves_baseline_untouched() -> None:
    baseline = PerceptionEmbeddingBaseline(embedding_ewma=(1.0, 0.0), n=3)
    result = perception_prediction_error([], baseline)
    assert result.score is None
    assert result.baseline is baseline


# --- identical vectors: cosine 1, surprise 0 --------------------------------


def test_identical_embedding_is_zero_surprise_and_ewma_holds_steady() -> None:
    """cos([1,0], [1,0]) = dot(1)/  (norm_a=1 * norm_b=1) = 1 -> surprise
    1 - 1 = 0. EWMA of a constant input stays exactly that constant:
    alpha*1 + (1-alpha)*1 == 1, alpha*0 + (1-alpha)*0 == 0."""
    baseline = PerceptionEmbeddingBaseline(embedding_ewma=(1.0, 0.0), n=1)
    result = perception_prediction_error([1.0, 0.0], baseline)
    assert result.score == 0.0
    assert result.baseline.embedding_ewma == (1.0, 0.0)
    assert result.baseline.n == 2


# --- orthogonal vectors: cosine 0, surprise 1 -------------------------------


def test_orthogonal_embedding_is_maximal_surprise_and_updates_ewma() -> None:
    """cos([1,0], [0,1]) = dot(0) / (1*1) = 0 -> surprise 1 - 0 = 1.
    alpha=0.2 EWMA update (hand-computed): new_x = 0.2*0 + 0.8*1.0 = 0.8,
    new_y = 0.2*1 + 0.8*0.0 = 0.2."""
    baseline = PerceptionEmbeddingBaseline(embedding_ewma=(1.0, 0.0), n=1)
    result = perception_prediction_error([0.0, 1.0], baseline)
    assert result.score == 1.0
    assert result.baseline.embedding_ewma == (0.8, 0.2)
    assert result.baseline.n == 2


def test_45_degree_embedding_gives_hand_computed_partial_surprise() -> None:
    """cos([1,0], [1,1]/... ) using raw (non-unit) baseline vector [1,1]:
    dot([1,0],[1,1]) = 1. norm_a = 1. norm_b = sqrt(1^2+1^2) = sqrt(2).
    cos = 1/sqrt(2) = 0.70710678... -> surprise = 1 - 1/sqrt(2)."""
    baseline = PerceptionEmbeddingBaseline(embedding_ewma=(1.0, 1.0), n=1)
    result = perception_prediction_error([1.0, 0.0], baseline)
    expected = 1.0 - (1.0 / math.sqrt(2.0))
    assert result.score == expected


# --- dimension mismatch: honest reseed, not a crash or a fabricated score --


def test_dimension_mismatch_reseeds_baseline_and_reports_no_score() -> None:
    baseline = PerceptionEmbeddingBaseline(embedding_ewma=(1.0, 0.0), n=5)
    result = perception_prediction_error([1.0, 0.0, 0.0], baseline)
    assert result.score is None
    assert result.baseline.embedding_ewma == (1.0, 0.0, 0.0)
    assert result.baseline.n == 1  # a real reseed, not baseline.n + 1


# --- degenerate zero-norm input: no score, no baseline mutation ------------


def test_zero_vector_against_real_baseline_reports_no_score_and_no_mutation() -> None:
    baseline = PerceptionEmbeddingBaseline(embedding_ewma=(1.0, 0.0), n=4)
    result = perception_prediction_error([0.0, 0.0], baseline)
    assert result.score is None
    assert result.baseline is baseline


def test_zero_vector_at_cold_start_does_not_seed_a_degenerate_baseline() -> None:
    """Review finding: seeding the baseline from a zero vector at cold start
    would make every subsequent real observation permanently degenerate too
    (a zero-norm baseline always fails _cosine_similarity's own zero-norm
    guard) -- must stay genuinely cold (n == 0) instead."""
    baseline = PerceptionEmbeddingBaseline()  # n=0
    result = perception_prediction_error([0.0, 0.0], baseline)
    assert result.score is None
    assert result.baseline is baseline
    assert result.baseline.n == 0

    # A real observation right after must behave exactly like a fresh cold
    # start, not like a stream stuck on a degenerate baseline.
    second = perception_prediction_error([1.0, 0.0], result.baseline)
    assert second.score is None
    assert second.baseline.embedding_ewma == (1.0, 0.0)
    assert second.baseline.n == 1


# --- surprise is clamped to [0, 1] ------------------------------------------


def test_surprise_never_exceeds_one_or_drops_below_zero() -> None:
    # Opposite-direction vectors would mathematically give cos=-1, surprise=2
    # (out of the module's own [0,1] pressure convention) without the clamp.
    baseline = PerceptionEmbeddingBaseline(embedding_ewma=(1.0, 0.0), n=1)
    result = perception_prediction_error([-1.0, 0.0], baseline)
    assert result.score == 1.0


# --- (de)serialization round-trip + tolerant parsing ------------------------


def test_baseline_json_round_trip() -> None:
    baseline = PerceptionEmbeddingBaseline(embedding_ewma=(0.8, 0.2, -0.1), n=7)
    restored = PerceptionEmbeddingBaseline.from_json_dict(baseline.to_json_dict())
    assert restored == baseline


def test_baseline_from_json_dict_tolerates_malformed_payload() -> None:
    assert PerceptionEmbeddingBaseline.from_json_dict({}) == PerceptionEmbeddingBaseline()
    assert PerceptionEmbeddingBaseline.from_json_dict(
        {"embedding_ewma": "not-a-list", "n": "not-an-int"}
    ) == PerceptionEmbeddingBaseline()
    assert PerceptionEmbeddingBaseline.from_json_dict(
        {"embedding_ewma": [1.0, "bad", 3.0], "n": 2}
    ) == PerceptionEmbeddingBaseline(embedding_ewma=(), n=2)
