"""Pre-registered fail criteria for lattice vs thermometer.

docs/research/preregistration/2026-09-19-heartbeat-lattice-vs-thermometer.md
"""
from __future__ import annotations

import math

import pytest

from app.substrate.lattice_probe import (
    decide_probe,
    score_kick,
    score_shuffle,
    shuffle_site,
)
from app.substrate.routing import BOUNDARY_SITES, BULK_SITES


def test_shuffle_site_is_a_cycle_over_the_boundary_seats() -> None:
    # Pre-reg arm B: site 0 -> 1, ..., site 4 -> 0.
    assert [shuffle_site(s) for s in BOUNDARY_SITES] == [1, 2, 3, 4, 0]
    cursor = 0
    for _ in range(len(BOUNDARY_SITES)):
        cursor = shuffle_site(cursor)
    assert cursor == 0


def test_shuffle_site_leaves_bulk_sites_alone() -> None:
    for site in BULK_SITES:
        assert shuffle_site(site) == site


def test_shuffle_null_when_profiles_and_verdict_match() -> None:
    profile = [0.8] * 9
    score = score_shuffle(
        profile_a=profile,
        profile_b=list(profile),
        mean_ratio_a=0.90,
        mean_ratio_b=0.905,
        std_ratio_a=0.03,
        std_ratio_b=0.032,
        verdict_a="mixed",
        verdict_b="mixed",
    )
    assert score.null is True
    assert score.rel_shuffle < 0.05


def test_shuffle_not_null_when_profile_moves() -> None:
    a = [0.9] * 9
    b = [0.2] * 9
    score = score_shuffle(
        profile_a=a,
        profile_b=b,
        mean_ratio_a=0.90,
        mean_ratio_b=0.90,
        std_ratio_a=0.03,
        std_ratio_b=0.03,
        verdict_a="mixed",
        verdict_b="mixed",
    )
    assert score.null is False
    assert score.rel_shuffle > 0.05


def test_shuffle_not_null_when_verdict_differs_even_if_numbers_match() -> None:
    profile = [0.85] * 9
    score = score_shuffle(
        profile_a=profile,
        profile_b=list(profile),
        mean_ratio_a=0.90,
        mean_ratio_b=0.90,
        std_ratio_a=0.03,
        std_ratio_b=0.03,
        verdict_a="mixed",
        verdict_b="concentrated",
    )
    assert score.null is False


def test_kick_smear_when_far_moves_half_as_much_as_near() -> None:
    before = [0.5] * 9
    after = [0.7, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6]
    score = score_kick(profile_before=before, profile_after=after)
    assert score.near == pytest.approx(0.2)
    assert score.far == pytest.approx(0.1)
    assert score.smear == pytest.approx(0.5)
    assert score.smeared is True


def test_kick_local_when_far_barely_moves() -> None:
    before = [0.5] * 9
    after = [0.8, 0.7, 0.55, 0.5, 0.5, 0.5, 0.5, 0.51, 0.50]
    score = score_kick(profile_before=before, profile_after=after)
    assert score.smeared is False
    assert score.smear < 0.5


def test_kick_smear_if_near_is_dead() -> None:
    before = [0.5] * 9
    after = list(before)
    after[8] = 0.9
    score = score_kick(profile_before=before, profile_after=after)
    assert math.isinf(score.smear)
    assert score.smeared is True


def test_thermometer_only_when_both_fails_hold() -> None:
    shuffle = score_shuffle(
        profile_a=[0.8] * 9,
        profile_b=[0.8] * 9,
        mean_ratio_a=0.9,
        mean_ratio_b=0.9,
        std_ratio_a=0.03,
        std_ratio_b=0.03,
        verdict_a="mixed",
        verdict_b="mixed",
    )
    kick = score_kick(profile_before=[0.5] * 9, profile_after=[0.7, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6])
    decided = decide_probe(n_routed=400, shuffle=shuffle, kick=kick)
    assert decided.thermometer is True


def test_unverified_below_min_atoms() -> None:
    shuffle = score_shuffle(
        profile_a=[0.8] * 9,
        profile_b=[0.8] * 9,
        mean_ratio_a=0.9,
        mean_ratio_b=0.9,
        std_ratio_a=0.03,
        std_ratio_b=0.03,
        verdict_a="mixed",
        verdict_b="mixed",
    )
    kick = score_kick(profile_before=[0.5] * 9, profile_after=[0.7] * 9)
    decided = decide_probe(n_routed=399, shuffle=shuffle, kick=kick)
    assert decided.thermometer is False
    assert decided.reason.startswith("UNVERIFIED")


def test_lattice_when_shuffle_moves_and_kick_local() -> None:
    shuffle = score_shuffle(
        profile_a=[0.9] * 9,
        profile_b=[0.2] * 9,
        mean_ratio_a=0.9,
        mean_ratio_b=0.3,
        std_ratio_a=0.02,
        std_ratio_b=0.04,
        verdict_a="redundant",
        verdict_b="concentrated",
    )
    kick = score_kick(
        profile_before=[0.5] * 9,
        profile_after=[0.8, 0.7, 0.55, 0.5, 0.5, 0.5, 0.5, 0.51, 0.50],
    )
    decided = decide_probe(n_routed=800, shuffle=shuffle, kick=kick)
    assert decided.thermometer is False
    assert decided.reason.startswith("lattice")
