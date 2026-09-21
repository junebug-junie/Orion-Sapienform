"""Live proprioception scores: dark seats, smear, organ distinctness.

These are the tick-level facts the attention self-model and Hub should lead
with. Occupancy (who talked) is independent of the tensor profile (how far
coupling walked). Empty / untracked input stays honestly absent — never
invented as all-calm.
"""
from __future__ import annotations

import math

import pytest

from app.substrate.proprioception import (
    FIRE_WINDOW,
    SMEAR_MIN,
    OrganFireWindow,
    compute_proprioception,
    dark_seats,
    occupancy_distinctness,
    profile_smear,
)
from app.substrate.routing import ORGAN_SITE_MAP


def test_dark_seats_are_organs_with_zero_fires_in_the_window() -> None:
    counts = {
        "orion-hub": 0,
        "orion-biometrics": 12,
        "orion-cortex-exec": 0,
        "orion-bus": 3,
        "orion-cortex-orch": 1,
    }
    assert dark_seats(counts) == ["orion-cortex-exec", "orion-hub"]


def test_dark_seats_missing_organ_counts_as_dark() -> None:
    assert dark_seats({"orion-bus": 4}) == [
        "orion-biometrics",
        "orion-cortex-exec",
        "orion-cortex-orch",
        "orion-hub",
    ]


def test_dark_seats_empty_window_is_all_organs() -> None:
    assert dark_seats({}) == sorted(ORGAN_SITE_MAP)


def test_occupancy_distinctness_one_organ_talking_is_one() -> None:
    counts = {name: 0 for name in ORGAN_SITE_MAP}
    counts["orion-cortex-exec"] = 20
    assert occupancy_distinctness(counts) == pytest.approx(1.0)


def test_occupancy_distinctness_uniform_is_zero() -> None:
    counts = {name: 4 for name in ORGAN_SITE_MAP}
    assert occupancy_distinctness(counts) == pytest.approx(0.0)


def test_occupancy_distinctness_empty_is_absent() -> None:
    assert occupancy_distinctness({}) is None
    assert occupancy_distinctness({name: 0 for name in ORGAN_SITE_MAP}) is None


def test_profile_smear_far_half_of_near_is_smeared() -> None:
    profile = [1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    smear, smeared = profile_smear(profile)
    assert smear == pytest.approx(0.5)
    assert smeared is True


def test_profile_smear_localized_kick_is_not_smeared() -> None:
    profile = [1.0, 0.8, 0.4, 0.2, 0.1, 0.05, 0.04, 0.03, 0.02]
    smear, smeared = profile_smear(profile)
    assert smear < SMEAR_MIN
    assert smeared is False


def test_profile_smear_dead_near_is_absent_not_infinite() -> None:
    profile = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.4]
    smear, smeared = profile_smear(profile)
    assert smear is None
    assert smeared is None


def test_profile_smear_rejects_wrong_cut_count() -> None:
    with pytest.raises(ValueError, match="9-cut"):
        profile_smear([0.1, 0.2])


def test_compute_proprioception_untracked_fires_leaves_occupancy_absent() -> None:
    reading = compute_proprioception(
        fire_counts=None,
        mean_profile=[1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.2],
    )
    assert reading.dark_seats == []
    assert reading.organ_fire_counts == {}
    assert reading.organ_distinctness is None
    assert reading.smear is not None


def test_compute_proprioception_combines_occupancy_and_profile() -> None:
    counts = {name: 0 for name in ORGAN_SITE_MAP}
    counts["orion-hub"] = 10
    reading = compute_proprioception(
        fire_counts=counts,
        mean_profile=[1.0, 0.9, 0.3, 0.2, 0.1, 0.08, 0.05, 0.04, 0.03],
    )
    assert reading.dark_seats == [
        "orion-biometrics",
        "orion-bus",
        "orion-cortex-exec",
        "orion-cortex-orch",
    ]
    assert reading.organ_fire_counts["orion-hub"] == 10
    assert reading.organ_distinctness == pytest.approx(1.0)
    assert reading.smeared is False


def test_organ_fire_window_rolls_and_counts_only_allowlisted_organs() -> None:
    window = OrganFireWindow(maxlen=3)
    window.record("orion-hub")
    window.record("orion-hub")
    window.record("orion-bus")
    window.record("orion-cortex-exec")
    window.record("not-an-organ")
    counts = window.counts()
    assert counts["orion-hub"] == 1
    assert counts["orion-bus"] == 1
    assert counts["orion-cortex-exec"] == 1
    assert counts["orion-biometrics"] == 0
    assert sum(counts.values()) == 3
    assert FIRE_WINDOW >= 8
    assert math.isfinite(SMEAR_MIN)
