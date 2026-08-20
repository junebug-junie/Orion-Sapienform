"""Unit tests for P2's perceptual prediction-error signal (docs/superpowers/
specs/2026-08-12-perception-frontier-design.md): ``surprise = 1 -
cos(frame_embedding, EWMA_embedding)`` per camera stream, z-scored against a
second, scalar EWMA baseline of that raw magnitude (added 2026-08-19 --
see ``perception_prediction_error()``'s own docstring for why the raw
magnitude alone shipped, then was found numerically incomparable to every
other domain's z-scored ``prediction_error`` and migrated the same day).

Every non-trivial expected value below is hand-computed from simple
orthonormal-ish vectors (not just "does it run") -- see each test's own
comment for the arithmetic. Tests are split into three groups: the raw
cosine-distance magnitude (embedding-vector EWMA only, scalar surprise
baseline left cold on purpose so ``score`` stays ``None`` and the raw value
is inspected directly off ``baseline.surprise.ewma``), the z-score stage
(scalar surprise baseline pre-seeded so the arithmetic is legible without
chaining multiple real calls through tiny, numerically extreme first-tick
variance floors), and a genuine end-to-end multi-tick chain from a real
cold start (review finding, 2026-08-19: no test previously exercised the
real wiring between stage 1's output and stage 2's input across consecutive
real calls -- every z-score test synthetically pre-seeded the baseline
instead).
"""

from __future__ import annotations

import math

from orion.substrate.prediction_error import (
    PerceptionEmbeddingBaseline,
    _DomainEwmaBaseline,
    _PERCEPTION_PREDICTION_ERROR_MIN_VARIANCE,
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


# --- raw cosine-distance magnitude (scalar surprise baseline left cold, so
# --- ``score`` is None -- the raw value is inspected via
# --- ``baseline.surprise.ewma``, which a cold z-score seed always sets to
# --- exactly the fed-in magnitude; see ``_domain_zscore``) -----------------


def test_identical_embedding_is_zero_raw_surprise_and_ewma_holds_steady() -> None:
    """cos([1,0], [1,0]) = dot(1)/  (norm_a=1 * norm_b=1) = 1 -> surprise
    1 - 1 = 0. EWMA of a constant input stays exactly that constant:
    alpha*1 + (1-alpha)*1 == 1, alpha*0 + (1-alpha)*0 == 0. Scalar surprise
    baseline is cold (surprise.n=0 by default), so this stream's first real
    comparison reports no score yet -- the raw magnitude is still visible
    directly off the seeded scalar baseline."""
    baseline = PerceptionEmbeddingBaseline(embedding_ewma=(1.0, 0.0), n=1)
    result = perception_prediction_error([1.0, 0.0], baseline)
    assert result.score is None
    assert result.baseline.embedding_ewma == (1.0, 0.0)
    assert result.baseline.n == 2
    assert result.baseline.surprise.ewma == 0.0
    assert result.baseline.surprise.n == 1


# --- orthogonal vectors: cosine 0, raw surprise 1 ---------------------------


def test_orthogonal_embedding_is_maximal_raw_surprise_and_updates_both_ewmas() -> None:
    """cos([1,0], [0,1]) = dot(0) / (1*1) = 0 -> surprise 1 - 0 = 1.
    alpha=0.2 EWMA update (hand-computed): new_x = 0.2*0 + 0.8*1.0 = 0.8,
    new_y = 0.2*1 + 0.8*0.0 = 0.2. Scalar surprise baseline cold -> no score
    yet, but seeded to exactly this raw magnitude (1.0)."""
    baseline = PerceptionEmbeddingBaseline(embedding_ewma=(1.0, 0.0), n=1)
    result = perception_prediction_error([0.0, 1.0], baseline)
    assert result.score is None
    assert result.baseline.embedding_ewma == (0.8, 0.2)
    assert result.baseline.n == 2
    assert result.baseline.surprise.ewma == 1.0
    assert result.baseline.surprise.n == 1


def test_45_degree_embedding_gives_hand_computed_partial_raw_surprise() -> None:
    """cos([1,0], [1,1]/... ) using raw (non-unit) baseline vector [1,1]:
    dot([1,0],[1,1]) = 1. norm_a = 1. norm_b = sqrt(1^2+1^2) = sqrt(2).
    cos = 1/sqrt(2) = 0.70710678... -> surprise = 1 - 1/sqrt(2)."""
    baseline = PerceptionEmbeddingBaseline(embedding_ewma=(1.0, 1.0), n=1)
    result = perception_prediction_error([1.0, 0.0], baseline)
    expected_raw = 1.0 - (1.0 / math.sqrt(2.0))
    assert result.score is None
    assert result.baseline.surprise.ewma == expected_raw


# --- z-score stage: scalar surprise baseline pre-seeded, so the arithmetic
# --- is legible without chaining several real calls through a numerically
# --- extreme first-tick variance floor -------------------------------------


def test_second_real_comparison_z_scores_against_seeded_scalar_baseline() -> None:
    """Two identical raw-surprise observations in a row: prev scalar
    baseline (ewma=0.0, variance=0.0, n=1) after the first real comparison,
    second comparison feeds the same raw_surprise=0.0 again.
    zscore = (0.0 - 0.0) / sqrt(max(0.0, min_variance)) = 0.0 -> score 0.0."""
    warm = PerceptionEmbeddingBaseline(
        embedding_ewma=(1.0, 0.0), n=2, surprise=_DomainEwmaBaseline(ewma=0.0, variance=0.0, n=1)
    )
    result = perception_prediction_error([1.0, 0.0], warm)
    assert result.score == 0.0
    assert result.baseline.surprise.n == 2


def test_mid_range_zscore_hand_computed_against_seeded_baseline() -> None:
    """baseline=(1,0) (unit), embedding chosen so cos=0.825 exactly ->
    raw_surprise=0.175. Scalar surprise baseline pre-seeded to
    ewma=0.1, variance=0.0025 (std=0.05), n=5 -- a stream that has already
    warmed up with a calm ~0.1 average and modest spread.

    zscore = (0.175 - 0.1) / sqrt(0.0025) = 0.075 / 0.05 = 1.5
    score = min(1.0, max(0.0, 1.5) / 3.0) = 0.5 exactly.
    """
    prev_ewma, prev_var = 0.1, 0.0025
    cos = 0.825
    embedding = (cos, math.sqrt(1.0 - cos * cos))
    baseline = PerceptionEmbeddingBaseline(
        embedding_ewma=(1.0, 0.0), n=6, surprise=_DomainEwmaBaseline(ewma=prev_ewma, variance=prev_var, n=5)
    )
    result = perception_prediction_error(list(embedding), baseline)
    raw_surprise = 1.0 - cos
    expected_zscore = (raw_surprise - prev_ewma) / math.sqrt(prev_var)
    expected_score = min(1.0, max(0.0, expected_zscore) / 3.0)
    assert result.score == expected_score
    assert abs(result.score - 0.5) < 1e-9


def test_extreme_deviation_saturates_score_at_one() -> None:
    """Mirrors the real live event this migration was built for: a stream
    with a tight calm baseline (ewma=0.005, variance=2e-5, matching the
    real measured calm-state numbers this domain's own comment cites) sees
    a raw_surprise=1.0 observation (cos=0, maximally different frame --
    e.g. the camera's view changed completely). The resulting z-score is
    far above the saturation constant and must clamp to 1.0, not overflow
    past it."""
    baseline = PerceptionEmbeddingBaseline(
        embedding_ewma=(1.0, 0.0), n=11, surprise=_DomainEwmaBaseline(ewma=0.005, variance=0.00002, n=10)
    )
    result = perception_prediction_error([0.0, 1.0], baseline)
    assert result.score == 1.0


def test_below_baseline_dip_clamps_to_zero_not_negative() -> None:
    """A calmer-than-usual observation (raw_surprise below the scalar
    baseline's own mean) must clamp to 0.0, not report a negative score --
    'surprising' means more than usual, not merely different, same
    convention every other domain in this module already documents."""
    cos = 0.9  # raw_surprise = 0.1, below the seeded mean of 0.5
    embedding = (cos, math.sqrt(1.0 - cos * cos))
    baseline = PerceptionEmbeddingBaseline(
        embedding_ewma=(1.0, 0.0), n=6, surprise=_DomainEwmaBaseline(ewma=0.5, variance=0.01, n=5)
    )
    result = perception_prediction_error(list(embedding), baseline)
    assert result.score == 0.0


# --- genuine end-to-end multi-tick chain, from a real cold start (review
# --- finding, 2026-08-19: every z-score test above pre-seeds the scalar
# --- baseline synthetically -- this one threads three real consecutive
# --- calls, each baseline coming only from the prior call's own return
# --- value, to prove stage 1's output actually wires into stage 2's input
# --- correctly, not just that each stage is independently correct) --------


def test_three_real_ticks_from_cold_start_hand_computed_end_to_end() -> None:
    """Tick 1: cold start, seeds embedding_ewma=(1,0), n=1, surprise still
    cold (n=0) -- no score.
    Tick 2: embedding=(0,1) -> cos=0, raw_surprise=1.0. Embedding EWMA
    updates to (0.8, 0.2). Scalar surprise baseline cold (n=0) -> seeds to
    ewma=1.0, still no score (this stream's first real comparison).
    Tick 3: embedding=(0,1) again. cos([0,1],[0.8,0.2]) = 0.2/sqrt(0.68) --
    hand-computed below via the same cosine formula the implementation
    uses, not re-derived independently, so this is a wiring test (does
    stage 2 receive tick 2's real surprise.ewma=1.0/variance=0.0/n=1
    correctly), not a second correctness test of the cosine math itself
    (already covered by the raw-surprise tests above).
    """
    baseline = PerceptionEmbeddingBaseline()

    tick1 = perception_prediction_error([1.0, 0.0], baseline)
    assert tick1.score is None
    assert tick1.baseline.embedding_ewma == (1.0, 0.0)
    assert tick1.baseline.n == 1
    assert tick1.baseline.surprise == _DomainEwmaBaseline()

    tick2 = perception_prediction_error([0.0, 1.0], tick1.baseline)
    assert tick2.score is None
    assert tick2.baseline.embedding_ewma == (0.8, 0.2)
    assert tick2.baseline.n == 2
    assert tick2.baseline.surprise.ewma == 1.0
    assert tick2.baseline.surprise.variance == 0.0
    assert tick2.baseline.surprise.n == 1

    tick3 = perception_prediction_error([0.0, 1.0], tick2.baseline)
    norm_b = math.sqrt(0.8 * 0.8 + 0.2 * 0.2)
    cos3 = (0.0 * 0.8 + 1.0 * 0.2) / (1.0 * norm_b)
    raw_surprise3 = max(0.0, min(1.0, 1.0 - cos3))
    # Real constant, not a hand-copied literal (bug this session found: a
    # prior version of this test hardcoded `1e-8` directly, which silently
    # went stale and would have asserted the *wrong* expected score the
    # moment the real floor was recalibrated -- see
    # _PERCEPTION_PREDICTION_ERROR_MIN_VARIANCE's own comment for why 2e-6
    # replaced 1e-8).
    variance_floor = max(tick2.baseline.surprise.variance, _PERCEPTION_PREDICTION_ERROR_MIN_VARIANCE)
    expected_zscore3 = (raw_surprise3 - tick2.baseline.surprise.ewma) / math.sqrt(variance_floor)
    expected_score3 = min(1.0, max(0.0, expected_zscore3) / 3.0)
    assert tick3.score == expected_score3
    assert tick3.baseline.surprise.n == 2


# --- dimension mismatch: honest reseed, not a crash or a fabricated score --


def test_dimension_mismatch_reseeds_baseline_and_reports_no_score() -> None:
    baseline = PerceptionEmbeddingBaseline(embedding_ewma=(1.0, 0.0), n=5)
    result = perception_prediction_error([1.0, 0.0, 0.0], baseline)
    assert result.score is None
    assert result.baseline.embedding_ewma == (1.0, 0.0, 0.0)
    assert result.baseline.n == 1  # a real reseed, not baseline.n + 1


def test_dimension_mismatch_reseeds_scalar_surprise_baseline_too() -> None:
    """Review finding, 2026-08-19: the raw [0,1] surprise magnitude doesn't
    itself depend on embedding dimensionality, so an argument exists for
    carrying the scalar calibration forward across a dimension change --
    deliberately NOT done (see perception_prediction_error()'s own
    docstring for the full reasoning: a dimension change means the
    embedding *model* changed, and a different model's calm-state surprise
    distribution isn't assumed to match the old one's just because both
    happen to land in [0,1]). A warm, well-calibrated scalar baseline must
    reset to cold on a dimension-mismatch reseed, same as the vector one."""
    warm_baseline = PerceptionEmbeddingBaseline(
        embedding_ewma=(1.0, 0.0), n=500, surprise=_DomainEwmaBaseline(ewma=0.1, variance=0.002, n=499)
    )
    result = perception_prediction_error([1.0, 0.0, 0.0], warm_baseline)
    assert result.score is None
    assert result.baseline.surprise == _DomainEwmaBaseline()


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


# --- raw surprise is clamped to [0, 1] before it ever reaches the z-score
# --- stage -------------------------------------------------------------


def test_raw_surprise_never_exceeds_one_or_drops_below_zero() -> None:
    # Opposite-direction vectors would mathematically give cos=-1, surprise=2
    # (out of the module's own [0,1] pressure convention) without the clamp.
    # Scalar surprise baseline is cold here, so the clamped value is visible
    # directly off the seeded baseline rather than through `score`.
    baseline = PerceptionEmbeddingBaseline(embedding_ewma=(1.0, 0.0), n=1)
    result = perception_prediction_error([-1.0, 0.0], baseline)
    assert result.score is None
    assert result.baseline.surprise.ewma == 1.0


# --- (de)serialization round-trip + tolerant parsing ------------------------


def test_baseline_json_round_trip() -> None:
    baseline = PerceptionEmbeddingBaseline(embedding_ewma=(0.8, 0.2, -0.1), n=7)
    restored = PerceptionEmbeddingBaseline.from_json_dict(baseline.to_json_dict())
    assert restored == baseline


def test_baseline_json_round_trip_includes_scalar_surprise_fields() -> None:
    """Added 2026-08-19 with the z-score migration -- a round trip that only
    exercised the all-default (0.0/0.0/0) scalar fields would pass even if
    to_json_dict/from_json_dict silently dropped them."""
    baseline = PerceptionEmbeddingBaseline(
        embedding_ewma=(0.8, 0.2, -0.1), n=7, surprise=_DomainEwmaBaseline(ewma=0.042, variance=0.0013, n=6)
    )
    restored = PerceptionEmbeddingBaseline.from_json_dict(baseline.to_json_dict())
    assert restored == baseline


def test_baseline_from_json_dict_tolerates_missing_scalar_surprise_key() -> None:
    """A row persisted before this migration has no `surprise` key at all --
    must default to a cold scalar baseline, not raise."""
    restored = PerceptionEmbeddingBaseline.from_json_dict(
        {"embedding_ewma": [1.0, 0.0], "n": 3}
    )
    assert restored == PerceptionEmbeddingBaseline(embedding_ewma=(1.0, 0.0), n=3)


def test_baseline_from_json_dict_tolerates_malformed_payload() -> None:
    assert PerceptionEmbeddingBaseline.from_json_dict({}) == PerceptionEmbeddingBaseline()
    assert PerceptionEmbeddingBaseline.from_json_dict(
        {"embedding_ewma": "not-a-list", "n": "not-an-int"}
    ) == PerceptionEmbeddingBaseline()
    assert PerceptionEmbeddingBaseline.from_json_dict(
        {"embedding_ewma": [1.0, "bad", 3.0], "n": 2}
    ) == PerceptionEmbeddingBaseline(embedding_ewma=(), n=2)
    assert PerceptionEmbeddingBaseline.from_json_dict(
        {"surprise": {"ewma": "not-a-float", "variance": "also-bad", "n": "nope"}}
    ) == PerceptionEmbeddingBaseline()
    assert PerceptionEmbeddingBaseline.from_json_dict(
        {"surprise": "not-a-dict"}
    ) == PerceptionEmbeddingBaseline()


def test_raw_surprise_is_none_when_no_comparison_happens() -> None:
    """Cold start, degenerate input, and dimension mismatch never compute a
    real cosine comparison -- raw_surprise stays None alongside score,
    same convention."""
    cold = PerceptionEmbeddingBaseline()
    assert perception_prediction_error([1.0, 0.0], cold).raw_surprise is None
    assert perception_prediction_error([], cold).raw_surprise is None
    assert perception_prediction_error([0.0, 0.0], cold).raw_surprise is None

    warm = PerceptionEmbeddingBaseline(embedding_ewma=(1.0, 0.0, 0.0), n=3)
    assert perception_prediction_error([1.0, 0.0], warm).raw_surprise is None
    assert perception_prediction_error([0.0, 0.0], warm).raw_surprise is None


def test_raw_surprise_populated_even_while_score_still_warming() -> None:
    """A stream's very first real cosine comparison has a warm embedding
    baseline but a cold scalar surprise baseline -- score is None (no
    z-score baseline yet) but raw_surprise is a real, already-computed
    number. This is exactly the distinction the field exists to expose:
    without it, this case was indistinguishable from cold-start/degenerate
    (both reported score=None with nothing else to inspect)."""
    warm = PerceptionEmbeddingBaseline(embedding_ewma=(1.0, 0.0), n=1)
    result = perception_prediction_error([0.0, 1.0], warm)
    assert result.score is None
    assert result.raw_surprise == 1.0


# --- variance-whipsaw floor (2026-08-20): a confirmed-static camera still
# --- crossed min_error=0.55 on 10% of real ticks and saturated to 1.0 on
# --- ~4% under the old 1e-8 floor, because compute_ewma_update floors only
# --- the z-score *denominator*, never the *stored* variance it returns --
# --- a locally-quiet stretch can whipsaw the tracked variance well below
# --- this domain's true ~2e-6 steady-state variance, and the next
# --- ordinary-magnitude tick then divides by that artificially small
# --- number. These tests pin the floor's own value and prove it actually
# --- suppresses that whipsaw, not just that some floor exists. -------------


def test_min_variance_floor_raised_to_true_steady_state_variance() -> None:
    """Pins the exact constant, not just its behavior -- a regression test
    for the constant itself catches an accidental revert (e.g. a bad merge
    restoring the old 1e-8) that a purely behavioral test might not, since
    plenty of ordinary-magnitude ticks still pass either floor."""
    assert _PERCEPTION_PREDICTION_ERROR_MIN_VARIANCE == 2e-6


def test_whipsawed_low_tracked_variance_no_longer_produces_a_spurious_saturated_score() -> None:
    """Reproduces the exact failure mode found live 2026-08-20: a scalar
    surprise baseline whose *tracked* variance has drifted to 5e-7 (the real
    minimum observed on a confirmed-static camera stream over 240 ticks --
    alpha=0.2's ~5-tick effective memory is genuinely capable of estimating
    a variance this low even though the domain's true population variance
    is ~1.933e-6, ~4x higher) then sees a perfectly ordinary-magnitude tick
    (raw_surprise=0.006, inside the real measured [0.00197, 0.01003] range
    for this same confirmed-static stream -- not a real event).

    Under the old 1e-8 floor, max(5e-7, 1e-8) = 5e-7 (the whipsawed value
    wins) -> zscore = (0.006 - 0.004174) / sqrt(5e-7) = 0.001826 / 7.071e-4
    ~= 2.58 -> score = min(1.0, 2.58/3.0) ~= 0.86 -- comfortably above
    min_error=0.55 on a scene that never moved.

    Under the new 2e-6 floor, max(5e-7, 2e-6) = 2e-6 (the floor wins
    instead) -> zscore = 0.001826 / sqrt(2e-6) ~= 1.29 -> score ~= 0.43 --
    below min_error=0.55, correctly read as ordinary variation."""
    cos = 1.0 - 0.006  # raw_surprise = 0.006 exactly
    embedding = (cos, math.sqrt(1.0 - cos * cos))
    whipsawed_variance = 5e-7
    baseline = PerceptionEmbeddingBaseline(
        embedding_ewma=(1.0, 0.0),
        n=50,
        surprise=_DomainEwmaBaseline(ewma=0.004174, variance=whipsawed_variance, n=49),
    )
    result = perception_prediction_error(list(embedding), baseline)

    old_floor = 1e-8
    old_variance_floor = max(whipsawed_variance, old_floor)
    old_zscore = (0.006 - 0.004174) / math.sqrt(old_variance_floor)
    old_score = min(1.0, max(0.0, old_zscore) / 3.0)
    assert old_score > 0.55  # confirms this scenario really was a false positive before

    new_variance_floor = max(whipsawed_variance, _PERCEPTION_PREDICTION_ERROR_MIN_VARIANCE)
    expected_zscore = (0.006 - 0.004174) / math.sqrt(new_variance_floor)
    expected_score = min(1.0, max(0.0, expected_zscore) / 3.0)
    # Tolerance, not exact equality: the literal 0.006 above and the actual
    # raw_surprise the implementation computes via cos/dot/sqrt agree to
    # ~1e-15, not bit-for-bit (same reconstruction-tolerance convention
    # test_raw_surprise_matches_score_times_saturation_when_unsaturated
    # already uses below).
    assert abs(result.score - expected_score) < 1e-9
    assert result.score < 0.55


def test_real_camera_knock_event_still_saturates_under_the_raised_floor() -> None:
    """The raised floor must not blunt sensitivity to a genuine dramatic
    event -- replays the actual live-confirmed camera-knock magnitude
    (raw_surprise ~0.12, this domain's own real-event reference point, see
    _PERCEPTION_PREDICTION_ERROR_ZSCORE_SATURATION's comment) against a
    calm baseline using the *new* floor and confirms it still saturates
    at 1.0, exactly like it did under the old floor
    (test_extreme_deviation_saturates_score_at_one, above)."""
    cos = 1.0 - 0.12
    embedding = (cos, math.sqrt(1.0 - cos * cos))
    baseline = PerceptionEmbeddingBaseline(
        embedding_ewma=(1.0, 0.0),
        n=50,
        surprise=_DomainEwmaBaseline(ewma=0.004174, variance=1.933e-6, n=49),
    )
    result = perception_prediction_error(list(embedding), baseline)
    assert result.score == 1.0


def test_raw_surprise_matches_score_times_saturation_when_unsaturated() -> None:
    """Live-data cross-check: for any unsaturated tick, raw_surprise is
    independently recoverable as
    surprise.ewma + (score * SATURATION) * sqrt(surprise.variance) against
    the *prior* scalar baseline -- confirming the new field and the
    existing score math agree on the same underlying magnitude, not two
    independently-drifting numbers."""
    baseline = PerceptionEmbeddingBaseline(
        embedding_ewma=(1.0, 0.0), n=5, surprise=_DomainEwmaBaseline(ewma=0.1, variance=0.0025, n=4)
    )
    cos = 0.825
    embedding = (cos, math.sqrt(1.0 - cos * cos))
    result = perception_prediction_error(embedding, baseline)
    assert result.score is not None and result.score < 1.0
    reconstructed = 0.1 + (result.score * 3.0) * math.sqrt(0.0025)
    assert abs(reconstructed - result.raw_surprise) < 1e-9
