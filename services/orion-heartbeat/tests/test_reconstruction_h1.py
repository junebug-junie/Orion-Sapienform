from __future__ import annotations

import math
import statistics

import pytest

from app.substrate.mps_state import HeartbeatSubstrate
from app.substrate.reconstruction import compute_h1
from app.substrate.routing import BOND_DIM, BOUNDARY_BULK_CUT, N_SITES, route_atom


def test_compute_h1_on_fresh_substrate_is_not_degenerate() -> None:
    # CLAUDE.md's no-empty-shell-cognition rule, applied to this harness
    # itself: a freshly-initialized random MPS should already show some real
    # entanglement (random tensors at bond_dim=4 are generically entangled),
    # not a silent zero -- if this test ever sees ratio==0.0 exactly, that's
    # a bug in the harness, not a legitimate "no structure" finding.
    sub = HeartbeatSubstrate(seed=7)
    result = compute_h1(sub)

    assert len(result.entropy_profile) == N_SITES - 1
    assert result.boundary_bulk_entropy > 0.0
    assert result.max_possible_entropy == pytest.approx(math.log2(BOND_DIM))
    assert 0.0 < result.ratio <= 1.0
    assert result.verdict in ("redundant", "concentrated", "mixed")
    assert result.tick_count == 0


def test_compute_h1_boundary_bulk_entropy_matches_profile_index() -> None:
    sub = HeartbeatSubstrate(seed=3)
    result = compute_h1(sub)
    assert result.boundary_bulk_entropy == result.entropy_profile[BOUNDARY_BULK_CUT - 1]


def test_compute_h1_tick_count_reflects_absorptions() -> None:
    sub = HeartbeatSubstrate(seed=5)
    for _ in range(4):
        assignment = route_atom(
            source_service="orion-bus", atom_type="signal", confidence=0.7, salience=0.6, uncertainty=0.3
        )
        sub.absorb(assignment)

    result = compute_h1(sub)
    assert result.tick_count == 4


def test_verdict_thresholds_are_internally_consistent() -> None:
    sub = HeartbeatSubstrate(seed=11)
    result = compute_h1(sub)
    if result.ratio >= 0.6:
        assert result.verdict == "redundant"
    elif result.ratio <= 0.2:
        assert result.verdict == "concentrated"
    else:
        assert result.verdict == "mixed"


def test_boundary_subprofile_length() -> None:
    sub = HeartbeatSubstrate(seed=13)
    result = compute_h1(sub)
    # boundary block has 5 sites -> 4 internal cuts
    assert len(result.boundary_subprofile) == 4


def test_compute_h1_ensemble_reports_every_trajectory_ratio() -> None:
    """2026-07-30: the per-trajectory ratios behind mean/std are reported, not
    discarded. At small N, mean+std alone cannot distinguish "all trajectories
    agree" from "two clusters that average out" -- which is the entire reason
    this substrate runs an ensemble rather than one trajectory, so the samples
    have to be inspectable, not just their summary."""
    from app.substrate.ensemble import EnsembleConfig, EnsembleSubstrate
    from app.substrate.reconstruction import compute_h1_ensemble

    ensemble = EnsembleSubstrate(config=EnsembleConfig(n_trajectories=4), base_seed=100)
    result = compute_h1_ensemble(ensemble)

    assert len(result.ratios) == 4
    assert len(result.ratios) == len(result.seeds)
    assert all(0.0 <= r <= 1.0 for r in result.ratios)
    # The reported mean/std must actually summarize the reported samples --
    # not be computed from a second, independently-recomputed sample set.
    assert result.mean_ratio == pytest.approx(sum(result.ratios) / len(result.ratios))
    assert result.std_ratio == pytest.approx(statistics.pstdev(result.ratios))


def test_compute_h1_ensemble_reports_bulk_penetration_depth() -> None:
    from app.substrate.ensemble import EnsembleConfig, EnsembleSubstrate
    from app.substrate.reconstruction import bulk_penetration_depth, compute_h1_ensemble

    ensemble = EnsembleSubstrate(config=EnsembleConfig(n_trajectories=4), base_seed=300)
    result = compute_h1_ensemble(ensemble)

    profiles = [traj.entropy_profile() for traj in ensemble.trajectories]
    n_cuts = len(profiles[0])
    mean_profile = [
        float(sum(p[i] for p in profiles) / len(profiles))
        for i in range(n_cuts)
    ]
    assert result.bulk_penetration_depth == pytest.approx(
        bulk_penetration_depth(mean_profile)
    )
    assert 0.0 <= result.bulk_penetration_depth <= 1.0


def test_verdict_thresholds_helper_matches_live_classification() -> None:
    """The helper a read-only surface draws its bands from must be the same
    numbers compute_h1_ensemble actually classifies with -- the point of
    exposing them is precisely so they cannot drift apart when retuned
    (design doc "Recommended next patch" step 4 anticipates retuning)."""
    from app.substrate.ensemble import EnsembleConfig, EnsembleSubstrate
    from app.substrate.reconstruction import compute_h1_ensemble, verdict_thresholds

    bands = verdict_thresholds()
    assert bands["low_ratio"] < bands["high_ratio"]

    ensemble = EnsembleSubstrate(config=EnsembleConfig(n_trajectories=3), base_seed=200)
    result = compute_h1_ensemble(ensemble)
    if result.mean_ratio >= bands["high_ratio"]:
        assert result.verdict == "redundant"
    elif result.mean_ratio <= bands["low_ratio"]:
        assert result.verdict == "concentrated"
    else:
        assert result.verdict == "mixed"
