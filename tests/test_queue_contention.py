"""Pure EWMA-relative queue contention score (0–10, max of source subs).

Formula locked in docs/superpowers/specs/2026-09-20-hire-handoff-and-queue-
pressure-design.md and the Task 5 metric gate (all three sources retained).
"""
from __future__ import annotations

import math

import pytest

from orion.field.queue_contention import (
    SOURCE_DURABLE,
    SOURCE_GPU_POOL,
    SOURCE_SEED,
    QueueContentionReading,
    ewma_alpha,
    score_queue_contention,
)


def test_at_baseline_scores_zero() -> None:
    reading = score_queue_contention(
        {SOURCE_SEED: 100.0},
        prev_ewma={SOURCE_SEED: 100.0},
        prev_n={SOURCE_SEED: 50},
        alpha=0.01,
    )
    assert reading.score == 0.0
    assert reading.driver is None


def test_five_x_baseline_scores_ten() -> None:
    reading = score_queue_contention(
        {SOURCE_DURABLE: 10.0},
        prev_ewma={SOURCE_DURABLE: 2.0},
        prev_n={SOURCE_DURABLE: 50},
        alpha=0.0,  # freeze ewma so the assert is about ratio math, not absorb
    )
    assert reading.score == 10.0
    assert reading.driver == SOURCE_DURABLE
    assert reading.ewma[SOURCE_DURABLE] == 2.0


def test_max_not_average() -> None:
    """One hot source must dominate — averaging against calm queues would hide it."""
    reading = score_queue_contention(
        {
            SOURCE_SEED: 100.0,  # at baseline → sub 0
            SOURCE_DURABLE: 10.0,  # 5x of 2 → sub 10
            SOURCE_GPU_POOL: 0.0,  # calm → sub 0
        },
        prev_ewma={
            SOURCE_SEED: 100.0,
            SOURCE_DURABLE: 2.0,
            SOURCE_GPU_POOL: 0.0,
        },
        prev_n={SOURCE_SEED: 50, SOURCE_DURABLE: 50, SOURCE_GPU_POOL: 50},
        alpha=0.0,
    )
    assert reading.score == 10.0
    assert reading.driver == SOURCE_DURABLE
    # Average of (0, 10, 0) would be ~3.33 — max must not collapse to that.
    assert reading.score != pytest.approx(10.0 / 3.0)


def test_ratio_midpoint_is_five() -> None:
    """3x baseline → clip(10 * (3-1)/4) = 5."""
    reading = score_queue_contention(
        {SOURCE_GPU_POOL: 3.0},
        prev_ewma={SOURCE_GPU_POOL: 1.0},
        prev_n={SOURCE_GPU_POOL: 20},
        alpha=0.0,
    )
    assert reading.score == 5.0
    assert reading.driver == SOURCE_GPU_POOL


def test_floor_avoids_divide_by_near_zero() -> None:
    """EWMA near 0 still denominates at floor=1.0 — count=5 → ratio 5 → score 10."""
    reading = score_queue_contention(
        {SOURCE_GPU_POOL: 5.0},
        prev_ewma={SOURCE_GPU_POOL: 0.01},
        prev_n={SOURCE_GPU_POOL: 10},
        alpha=0.0,
        floor=1.0,
    )
    assert reading.score == 10.0


def test_first_observation_establishes_baseline_without_false_spike() -> None:
    """No prior n → cannot be 'above normal'; sub=0, ewma becomes the count."""
    reading = score_queue_contention(
        {SOURCE_SEED: 121.0},
        prev_ewma={},
        prev_n={},
        alpha=0.01,
    )
    assert reading.score == 0.0
    assert reading.driver is None
    assert reading.ewma[SOURCE_SEED] == 121.0
    assert reading.raw[SOURCE_SEED] == 121.0


def test_ewma_alpha_half_life_24h_at_2s_tick() -> None:
    alpha = ewma_alpha(dt_sec=2.0, half_life_sec=86400.0)
    expected = 1.0 - math.exp(-math.log(2) * 2.0 / 86400.0)
    assert alpha == pytest.approx(expected)
    assert 0.0 < alpha < 0.001


def test_reading_is_frozen_dataclass() -> None:
    reading = score_queue_contention(
        {SOURCE_DURABLE: 2.0},
        prev_ewma={SOURCE_DURABLE: 2.0},
        prev_n={SOURCE_DURABLE: 1},
        alpha=0.0,
    )
    assert isinstance(reading, QueueContentionReading)
    with pytest.raises(Exception):
        reading.score = 9.0  # type: ignore[misc]


def test_retired_source_baseline_is_dropped_not_carried() -> None:
    reading = score_queue_contention(
        {SOURCE_GPU_POOL: 0.0},
        prev_ewma={"gateway_waiting": 0.0, SOURCE_SEED: 3.0},
        prev_n={"gateway_waiting": 163577, SOURCE_SEED: 10},
        alpha=0.1,
    )
    assert "gateway_waiting" not in reading.ewma and "gateway_waiting" not in reading.ewma_n
    assert reading.ewma[SOURCE_SEED] == 3.0 and reading.ewma_n[SOURCE_GPU_POOL] == 1


# --- Oldest-wait component (2026-09-25) -------------------------------------

from orion.field.queue_contention import (  # noqa: E402
    DEFAULT_EXPECTED_WAIT_SEC,
    OLDEST_WAIT_SUFFIX,
    driver_source,
)

HOUR = 3600.0


def test_live_frozen_seed_queue_replay_reads_stuck() -> None:
    """Replay of 2026-09-25 live: 144 pending vs EWMA 136, oldest from 09-07 (~432h).

    Depth alone scored 0.147 (144/136 = 1.06x) and was sinking to 0 as the
    EWMA converged on the frozen count. The oldest-wait sub must see it.
    """
    reading = score_queue_contention(
        {SOURCE_SEED: 144.0},
        prev_ewma={SOURCE_SEED: 136.0},
        prev_n={SOURCE_SEED: 1176},
        alpha=0.0,
        oldest_wait_sec={SOURCE_SEED: 432 * HOUR},
    )
    assert reading.subs[SOURCE_SEED] == pytest.approx(10 * (144 / 136 - 1) / 4)
    assert reading.subs[SOURCE_SEED] < 0.2
    assert reading.score == 10.0
    assert reading.driver == SOURCE_SEED + OLDEST_WAIT_SUFFIX
    assert driver_source(reading.driver) == (SOURCE_SEED, True)


def test_stuck_queue_stays_loud_after_depth_baseline_converges() -> None:
    """Lifecycle, not arithmetic: count flat at baseline for many ticks, oldest item aging.

    The depth sub is 0 the whole time (count == EWMA); the score must still be
    nonzero and name staleness, and must not decay the way the depth sub did.
    """
    ewma = {SOURCE_SEED: 144.0}
    n = {SOURCE_SEED: 5000}
    age = 72 * HOUR
    scores = []
    for _ in range(200):
        reading = score_queue_contention(
            {SOURCE_SEED: 144.0},
            prev_ewma=ewma,
            prev_n=n,
            alpha=0.01,
            oldest_wait_sec={SOURCE_SEED: age},
        )
        assert reading.subs[SOURCE_SEED] == 0.0
        ewma, n = reading.ewma, reading.ewma_n
        scores.append(reading.score)
        age += 2.0
    assert scores[0] > 0.0
    assert reading.driver == SOURCE_SEED + OLDEST_WAIT_SUFFIX
    assert scores[-1] > scores[0]  # keeps rising while the queue stays stuck; no decay


def test_nonpositive_expected_wait_skips_source_instead_of_crashing() -> None:
    reading = score_queue_contention(
        {},
        {},
        {},
        alpha=0.0,
        oldest_wait_sec={SOURCE_GPU_POOL: 900.0, SOURCE_DURABLE: 5 * 43200.0},
        expected_wait_sec={SOURCE_GPU_POOL: 0.0},
    )
    assert SOURCE_GPU_POOL + OLDEST_WAIT_SUFFIX not in reading.subs
    assert reading.driver == SOURCE_DURABLE + OLDEST_WAIT_SUFFIX


def test_empty_queue_rest_point_is_exact_zero() -> None:
    """Empty queue: count 0 on a 0 baseline and no oldest item (age 0.0) -> 0, no driver."""
    reading = score_queue_contention(
        {SOURCE_SEED: 0.0, SOURCE_DURABLE: 0.0, SOURCE_GPU_POOL: 0.0},
        prev_ewma={SOURCE_SEED: 0.0, SOURCE_DURABLE: 0.0, SOURCE_GPU_POOL: 0.0},
        prev_n={SOURCE_SEED: 10, SOURCE_DURABLE: 10, SOURCE_GPU_POOL: 10},
        alpha=0.01,
        oldest_wait_sec={SOURCE_SEED: 0.0, SOURCE_DURABLE: 0.0, SOURCE_GPU_POOL: 0.0},
    )
    assert reading.score == 0.0
    assert reading.driver is None
    assert all(v == 0.0 for v in reading.subs.values())


def test_fresh_items_rest_point_is_exact_zero() -> None:
    """Items waiting, but every oldest item younger than its expected wait -> exactly 0."""
    ages = {k: 0.99 * v for k, v in DEFAULT_EXPECTED_WAIT_SEC.items()}
    reading = score_queue_contention(
        {SOURCE_SEED: 5.0, SOURCE_DURABLE: 1.0, SOURCE_GPU_POOL: 1.0},
        prev_ewma={SOURCE_SEED: 5.0, SOURCE_DURABLE: 1.0, SOURCE_GPU_POOL: 1.0},
        prev_n={SOURCE_SEED: 10, SOURCE_DURABLE: 10, SOURCE_GPU_POOL: 10},
        alpha=0.01,
        oldest_wait_sec=ages,
    )
    assert reading.score == 0.0
    assert reading.driver is None


def test_age_sub_one_x_is_zero_and_five_x_is_ten() -> None:
    for src, expected in DEFAULT_EXPECTED_WAIT_SEC.items():
        at_1x = score_queue_contention({}, {}, {}, alpha=0.0, oldest_wait_sec={src: expected})
        at_3x = score_queue_contention({}, {}, {}, alpha=0.0, oldest_wait_sec={src: 3 * expected})
        at_5x = score_queue_contention({}, {}, {}, alpha=0.0, oldest_wait_sec={src: 5 * expected})
        assert at_1x.score == 0.0
        assert at_3x.score == pytest.approx(5.0)
        assert at_5x.score == 10.0
        assert at_5x.driver == src + OLDEST_WAIT_SUFFIX


def test_expected_wait_override_and_negative_age_clamped() -> None:
    reading = score_queue_contention(
        {},
        {},
        {},
        alpha=0.0,
        oldest_wait_sec={SOURCE_GPU_POOL: 300.0, SOURCE_SEED: -50.0},
        expected_wait_sec={SOURCE_GPU_POOL: 100.0},
    )
    assert reading.subs[SOURCE_GPU_POOL + OLDEST_WAIT_SUFFIX] == pytest.approx(5.0)
    assert reading.oldest_wait_sec[SOURCE_SEED] == 0.0
    assert reading.subs[SOURCE_SEED + OLDEST_WAIT_SUFFIX] == 0.0


def test_depth_wins_tie_over_oldest_wait() -> None:
    reading = score_queue_contention(
        {SOURCE_DURABLE: 10.0},
        prev_ewma={SOURCE_DURABLE: 2.0},
        prev_n={SOURCE_DURABLE: 50},
        alpha=0.0,
        oldest_wait_sec={SOURCE_DURABLE: 10 * DEFAULT_EXPECTED_WAIT_SEC[SOURCE_DURABLE]},
    )
    assert reading.score == 10.0
    assert reading.driver == SOURCE_DURABLE


def test_age_only_input_scores_without_counts() -> None:
    """An age with no count still scores (count reader down, age reader up)."""
    reading = score_queue_contention(
        {}, {}, {}, alpha=0.0, oldest_wait_sec={SOURCE_GPU_POOL: 600.0}
    )
    assert reading.score == 10.0
    assert reading.raw == {}
    assert reading.ewma_n == {}


def test_driver_source_parses_both_shapes() -> None:
    assert driver_source(None) == (None, False)
    assert driver_source(SOURCE_DURABLE) == (SOURCE_DURABLE, False)
    assert driver_source(SOURCE_GPU_POOL + OLDEST_WAIT_SUFFIX) == (SOURCE_GPU_POOL, True)
