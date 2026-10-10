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
    FIRE_WINDOW_SEC,
    SMEAR_DEAD_RATIO,
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


def test_profile_smear_nearly_dead_near_is_absent_not_huge() -> None:
    # Regression, 2026-10-10: near end at 1e-5 bits cleared the old 1e-6
    # floor, so far/near read 4e4 -- live rows reached 1.46e6. A near end that
    # carries < 1/SMEAR_DEAD_RATIO of the far end is dead: absent, not huge.
    profile = [1e-5, 1e-5, 0.3, 0.5, 0.6, 0.6, 0.5, 0.4, 0.4]
    smear, smeared = profile_smear(profile)
    assert smear is None
    assert smeared is None


def test_profile_smear_dead_cut_sits_at_live_distribution_trough() -> None:
    # Alive body tops out at 4.54 live; the cut is the trough edge at 10.
    assert SMEAR_DEAD_RATIO == pytest.approx(10.0)
    alive = [0.2, 0.2, 0.3, 0.5, 0.6, 0.6, 0.5, 0.9, 0.9]  # far/near = 4.5
    smear, smeared = profile_smear(alive)
    assert smear == pytest.approx(4.5)
    assert smeared is True
    at_edge = [0.1, 0.1, 0.3, 0.5, 0.6, 0.6, 0.5, 1.0, 1.0]  # far/near = 10
    assert profile_smear(at_edge)[0] == pytest.approx(10.0)
    past_edge = [0.09, 0.09, 0.3, 0.5, 0.6, 0.6, 0.5, 1.0, 1.0]  # ~11.1
    assert profile_smear(past_edge) == (None, None)


def test_compute_proprioception_nearly_dead_near_leaves_smear_absent() -> None:
    reading = compute_proprioception(
        fire_counts={name: 3 for name in ORGAN_SITE_MAP},
        mean_profile=[1e-5, 1e-5, 0.3, 0.5, 0.6, 0.6, 0.5, 0.4, 0.4],
    )
    assert reading.smear is None
    assert reading.smeared is None
    assert reading.organ_distinctness is not None


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


def test_organ_fire_window_counts_only_allowlisted_organs() -> None:
    window = OrganFireWindow()
    window.record("orion-hub")
    window.record("orion-hub")
    window.record("orion-bus")
    window.record("orion-cortex-exec")
    window.record("not-an-organ")
    counts = window.counts()
    assert counts["orion-hub"] == 2
    assert counts["orion-bus"] == 1
    assert counts["orion-cortex-exec"] == 1
    assert counts["orion-biometrics"] == 0
    assert sum(counts.values()) == 4
    assert FIRE_WINDOW_SEC > 0
    assert math.isfinite(SMEAR_MIN)


class _Clock:
    def __init__(self) -> None:
        self.t = 1_000_000.0

    def __call__(self) -> float:
        return self.t


def test_rare_organ_not_dark_when_it_fired_inside_window() -> None:
    clock = _Clock()
    w = OrganFireWindow(window_sec=300, clock=clock, wall_clock=clock)
    clock.t += 300  # past warm-up
    w.record("orion-cortex-orch")
    for _ in range(500):  # flood from a busy organ; count window of 64 would evict orch
        w.record("orion-biometrics")
    clock.t += 120
    assert "orion-cortex-orch" not in dark_seats(w.counts())
    assert "orion-hub" in dark_seats(w.counts())


def test_organ_flagged_dark_after_window_passes_with_recency() -> None:
    clock = _Clock()
    w = OrganFireWindow(window_sec=300, clock=clock, wall_clock=clock)
    clock.t += 300  # past warm-up
    w.record("orion-cortex-orch")
    w.record("orion-bus")
    clock.t += 200
    w.record("orion-bus")
    clock.t += 150  # orch last fired 350s ago, bus 150s ago
    snap = w.snapshot()
    reading = compute_proprioception(
        fire_counts=snap.counts, mean_profile=[1.0] * 9, fire_snapshot=snap
    )
    assert "orion-cortex-orch" in reading.dark_seats
    assert "orion-bus" not in reading.dark_seats
    assert reading.organ_seconds_since_last_fire["orion-cortex-orch"] == pytest.approx(350)
    assert reading.organ_seconds_since_last_fire["orion-hub"] is None
    assert reading.organ_last_fired_at["orion-cortex-orch"].startswith("1970-01-12")
    assert reading.fire_window_sec == 300


def test_empty_window_is_unknown_not_all_dark() -> None:
    w = OrganFireWindow(window_sec=300, clock=_Clock(), wall_clock=_Clock())
    snap = w.snapshot()
    reading = compute_proprioception(
        fire_counts=snap.counts, mean_profile=[1.0] * 9, fire_snapshot=snap
    )
    assert reading.dark_seats == []
    assert reading.organ_fire_counts == {}
    assert reading.organ_distinctness is None


def test_wall_clock_expiry_with_no_incoming_events() -> None:
    clock = _Clock()
    w = OrganFireWindow(window_sec=60, clock=clock, wall_clock=clock)
    w.record("orion-hub")
    assert w.counts()["orion-hub"] == 1
    clock.t += 61  # no record() calls at all
    assert w.counts()["orion-hub"] == 0
    snap = w.snapshot()
    assert sum(snap.counts.values()) == 0
    reading = compute_proprioception(
        fire_counts=snap.counts, mean_profile=[1.0] * 9, fire_snapshot=snap
    )
    assert reading.dark_seats == []  # whole window empty -> unknown
    assert reading.organ_distinctness is None
    # recency survives pruning AND is reported on the empty-window reading
    assert snap.seconds_since_last_fire["orion-hub"] == pytest.approx(61)
    assert reading.organ_seconds_since_last_fire["orion-hub"] == pytest.approx(61)
    assert reading.organ_last_fired_at["orion-hub"] is not None


def test_window_memory_is_bounded() -> None:
    w = OrganFireWindow(window_sec=300, clock=_Clock(), wall_clock=_Clock(), max_events_per_organ=10)
    for _ in range(1000):
        w.record("orion-bus")
    assert w.counts()["orion-bus"] == 10


def test_warmup_after_restart_reads_unknown_not_dark() -> None:
    clock = _Clock()
    w = OrganFireWindow(window_sec=300, clock=clock, wall_clock=clock)
    clock.t += 10
    w.record("orion-biometrics")  # only one organ has spoken since boot
    snap = w.snapshot()
    assert snap.warm is False
    reading = compute_proprioception(
        fire_counts=snap.counts, mean_profile=[1.0] * 9, fire_snapshot=snap
    )
    assert reading.dark_seats == []
    assert reading.organ_distinctness is None
    clock.t += 300
    w.record("orion-biometrics")
    snap = w.snapshot()
    assert snap.warm is True
    reading = compute_proprioception(
        fire_counts=snap.counts, mean_profile=[1.0] * 9, fire_snapshot=snap
    )
    assert "orion-hub" in reading.dark_seats


def test_h1_result_carries_recency_fields() -> None:
    from dataclasses import asdict

    from app.substrate.ensemble import EnsembleConfig, EnsembleSubstrate
    from app.substrate.reconstruction import compute_h1_ensemble

    clock = _Clock()
    w = OrganFireWindow(window_sec=300, clock=clock, wall_clock=clock)
    clock.t += 300
    w.record("orion-bus")
    snap = w.snapshot()
    ens = EnsembleSubstrate(config=EnsembleConfig(n_trajectories=2), base_seed=9)
    d = asdict(compute_h1_ensemble(ens, fire_counts=snap.counts, fire_snapshot=snap))
    assert d["fire_window_sec"] == 300
    assert d["organ_seconds_since_last_fire"]["orion-bus"] == pytest.approx(0)
    assert d["organ_last_fired_at"]["orion-hub"] is None
