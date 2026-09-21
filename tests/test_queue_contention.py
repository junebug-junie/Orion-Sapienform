"""Pure EWMA-relative queue contention score (0–10, max of source subs).

Formula locked in docs/superpowers/specs/2026-09-20-hire-handoff-and-queue-
pressure-design.md and the Task 5 metric gate (all three sources retained).
"""
from __future__ import annotations

import math

import pytest

from orion.field.queue_contention import (
    SOURCE_DURABLE,
    SOURCE_GATEWAY,
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
            SOURCE_GATEWAY: 0.0,  # calm → sub 0
        },
        prev_ewma={
            SOURCE_SEED: 100.0,
            SOURCE_DURABLE: 2.0,
            SOURCE_GATEWAY: 0.0,
        },
        prev_n={SOURCE_SEED: 50, SOURCE_DURABLE: 50, SOURCE_GATEWAY: 50},
        alpha=0.0,
    )
    assert reading.score == 10.0
    assert reading.driver == SOURCE_DURABLE
    # Average of (0, 10, 0) would be ~3.33 — max must not collapse to that.
    assert reading.score != pytest.approx(10.0 / 3.0)


def test_ratio_midpoint_is_five() -> None:
    """3x baseline → clip(10 * (3-1)/4) = 5."""
    reading = score_queue_contention(
        {SOURCE_GATEWAY: 3.0},
        prev_ewma={SOURCE_GATEWAY: 1.0},
        prev_n={SOURCE_GATEWAY: 20},
        alpha=0.0,
    )
    assert reading.score == 5.0
    assert reading.driver == SOURCE_GATEWAY


def test_floor_avoids_divide_by_near_zero() -> None:
    """EWMA near 0 still denominates at floor=1.0 — count=5 → ratio 5 → score 10."""
    reading = score_queue_contention(
        {SOURCE_GATEWAY: 5.0},
        prev_ewma={SOURCE_GATEWAY: 0.01},
        prev_n={SOURCE_GATEWAY: 10},
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
