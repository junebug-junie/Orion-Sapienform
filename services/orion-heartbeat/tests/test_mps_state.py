from __future__ import annotations

import pytest

from app.substrate.mps_state import HeartbeatSubstrate
from app.substrate.routing import BOND_DIM, N_SITES, SiteAssignment, route_atom


def _sample_assignment(site_organ: str = "orion-hub", **kwargs):
    defaults = dict(confidence=0.8, salience=0.6, uncertainty=0.3)
    defaults.update(kwargs)
    return route_atom(source_service=site_organ, atom_type="observation", **defaults)


def test_initial_state_is_normalized() -> None:
    sub = HeartbeatSubstrate(seed=1)
    assert sub.norm() == pytest.approx(1.0, abs=1e-8)
    assert sub.tick_count == 0


def test_absorb_keeps_state_normalized_and_bond_capped() -> None:
    sub = HeartbeatSubstrate(seed=1)
    organs = ["orion-hub", "orion-biometrics", "orion-cortex-exec", "orion-bus", "orion-cortex-orch"]
    for i in range(20):
        assignment = _sample_assignment(organs[i % len(organs)], confidence=0.7, salience=0.5, uncertainty=0.2)
        sub.absorb(assignment)

    assert sub.tick_count == 20
    assert sub.norm() == pytest.approx(1.0, abs=1e-6)
    assert sub.max_bond() <= BOND_DIM


def test_entropy_profile_shape_and_bounds() -> None:
    sub = HeartbeatSubstrate(seed=2)
    for _ in range(10):
        sub.absorb(_sample_assignment("orion-cortex-orch"))

    profile = sub.entropy_profile()
    assert len(profile) == N_SITES - 1
    for value in profile:
        assert value >= -1e-9  # entropy is non-negative (small float slop allowed)


def test_absorb_is_deterministic_given_same_seed_and_events() -> None:
    events = [
        ("orion-hub", "observation", 0.9, 0.5, 0.1),
        ("orion-biometrics", "signal", 0.6, 0.7, 0.4),
        ("orion-cortex-exec", "reasoning_step", 0.8, 0.3, 0.2),
    ]

    def run() -> list[float]:
        sub = HeartbeatSubstrate(seed=99)
        for organ, atom_type, conf, sal, unc in events:
            assignment = route_atom(
                source_service=organ, atom_type=atom_type, confidence=conf, salience=sal, uncertainty=unc
            )
            sub.absorb(assignment)
        return sub.entropy_profile()

    profile_a = run()
    profile_b = run()
    assert profile_a == pytest.approx(profile_b, abs=1e-10)


def test_different_seeds_produce_different_states() -> None:
    def run(seed: int) -> list[float]:
        sub = HeartbeatSubstrate(seed=seed)
        sub.absorb(_sample_assignment("orion-hub"))
        return sub.entropy_profile()

    profile_1 = run(1)
    profile_2 = run(2)
    assert profile_1 != pytest.approx(profile_2, abs=1e-6)


def test_absorb_guards_against_a_site_with_no_right_neighbor() -> None:
    # Unreachable through routing.route_atom() given the current
    # ORGAN_SITE_MAP (0-4, always < N_SITES-1) -- exercised directly here so
    # the defensive guard itself has real test coverage, not just a comment
    # claiming it does.
    sub = HeartbeatSubstrate(seed=1)
    bad_assignment = SiteAssignment(
        site_index=N_SITES - 1, operator_kind="amplitude", confidence=1.0, salience=1.0, uncertainty=0.0
    )
    with pytest.raises(ValueError):
        sub.absorb(bad_assignment)


def test_absorb_actually_changes_the_state() -> None:
    # Review gap: nothing previously proved absorb() does real dynamical
    # work rather than being a no-op that only increments tick_count -- a
    # stubbed-out absorb() would have passed every other test in this file.
    seed = 21
    sub_untouched = HeartbeatSubstrate(seed=seed)
    profile_before = sub_untouched.entropy_profile()

    sub_touched = HeartbeatSubstrate(seed=seed)
    sub_touched.absorb(_sample_assignment("orion-hub", confidence=0.95, salience=0.9, uncertainty=0.05))
    profile_after = sub_touched.entropy_profile()

    assert profile_after != pytest.approx(profile_before, abs=1e-9)


def test_absorb_reaches_every_bulk_site_not_just_the_nearest_one() -> None:
    # Regression test for the critical review finding (2026-07-24): the
    # original absorb() only gated (site, site+1), one hop, which left
    # cuts 6/7/8 (entirely within the declared 5-site bulk block) frozen at
    # their random-initialization values forever regardless of how much
    # traffic was absorbed. Confirmed live by an independent review pass
    # (froze within ~50 absorptions, stayed frozen through 800). This test
    # would have failed against the pre-fix version of absorb().
    sub = HeartbeatSubstrate(seed=17)
    organs = ["orion-hub", "orion-biometrics", "orion-cortex-exec", "orion-bus", "orion-cortex-orch"]

    profile_initial = sub.entropy_profile()
    # Cuts 6, 7, 8 (profile indices 5, 6, 7) are the ones entirely within
    # the bulk block that the original bug left permanently untouched.
    bulk_internal_indices = (5, 6, 7)

    for i in range(120):
        assignment = _sample_assignment(
            organs[i % len(organs)],
            confidence=0.6 + 0.3 * ((i * 7) % 5) / 5.0,
            salience=0.4 + 0.4 * ((i * 3) % 5) / 5.0,
            uncertainty=0.1 + 0.3 * ((i * 11) % 5) / 5.0,
        )
        sub.absorb(assignment)

    profile_final = sub.entropy_profile()
    for idx in bulk_internal_indices:
        assert profile_final[idx] != pytest.approx(profile_initial[idx], abs=1e-9), (
            f"cut index {idx} never moved from its initial value across 120 absorptions "
            "-- this is exactly the frozen-bulk-site bug found by review"
        )
