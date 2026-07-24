from __future__ import annotations

import math

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
