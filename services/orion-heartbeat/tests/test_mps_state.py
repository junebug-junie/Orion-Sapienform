from __future__ import annotations

import pytest

from app.substrate.mps_state import HeartbeatSubstrate
from app.substrate.routing import BOND_DIM, N_SITES, ORGAN_SITE_MAP, SiteAssignment


def _sample_assignment(site_organ: str = "orion-hub", level: float = 0.6, operator_kind: str = "amplitude"):
    return SiteAssignment(
        site_index=ORGAN_SITE_MAP[site_organ], operator_kind=operator_kind, level=level, input_kind="measured"
    )


def test_initial_state_is_normalized() -> None:
    sub = HeartbeatSubstrate(seed=1)
    assert sub.norm() == pytest.approx(1.0, abs=1e-8)
    assert sub.tick_count == 0


def test_absorb_keeps_state_normalized_and_bond_capped() -> None:
    sub = HeartbeatSubstrate(seed=1)
    organs = ["orion-hub", "orion-biometrics", "orion-cortex-exec", "orion-bus", "orion-cortex-orch"]
    for i in range(20):
        assignment = _sample_assignment(organs[i % len(organs)], level=0.5)
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
        ("orion-hub", "amplitude", 0.5),
        ("orion-biometrics", "amplitude", 0.7),
        ("orion-cortex-exec", "rotation", 0.3),
    ]

    def run() -> list[float]:
        sub = HeartbeatSubstrate(seed=99)
        for organ, kind, level in events:
            sub.absorb(_sample_assignment(organ, level=level, operator_kind=kind))
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
        site_index=N_SITES - 1, operator_kind="amplitude", level=1.0, input_kind="measured"
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
    sub_touched.absorb(_sample_assignment("orion-hub", level=0.9))
    profile_after = sub_touched.entropy_profile()

    assert profile_after != pytest.approx(profile_before, abs=1e-9)


def test_decay_tick_raises_before_relaxation_targets_computed() -> None:
    # compute_relaxation_targets() must be called once before any
    # decay_tick() -- documented guard, previously untested anywhere.
    sub = HeartbeatSubstrate(seed=1)
    with pytest.raises(RuntimeError, match="compute_relaxation_targets"):
        sub.decay_tick(decay_prob=0.1)


def test_decay_tick_does_not_change_tick_count() -> None:
    sub = HeartbeatSubstrate(seed=1)
    sub.compute_relaxation_targets(gamma=0.2)
    sub.absorb(_sample_assignment("orion-hub"))
    assert sub.tick_count == 1

    for _ in range(10):
        sub.decay_tick(decay_prob=0.3)

    assert sub.tick_count == 1  # dissipation is not organ traffic


def test_decay_tick_keeps_state_normalized() -> None:
    sub = HeartbeatSubstrate(seed=1)
    sub.compute_relaxation_targets(gamma=0.2)
    sub.absorb(_sample_assignment("orion-hub"))

    for _ in range(20):
        sub.decay_tick(decay_prob=0.3)

    assert sub.norm() == pytest.approx(1.0, abs=1e-6)
    assert sub.max_bond() <= BOND_DIM


def test_decay_tick_is_deterministic_given_same_seed() -> None:
    # Same discipline as test_absorb_is_deterministic_given_same_seed_and_
    # events -- decay_tick's own stochastic draws must come from this
    # trajectory's own seeded RNG, not global np.random state.
    def run() -> list[float]:
        sub = HeartbeatSubstrate(seed=55)
        sub.compute_relaxation_targets(gamma=0.2)
        sub.absorb(_sample_assignment("orion-hub"))
        for _ in range(15):
            sub.decay_tick(decay_prob=0.3)
        return sub.entropy_profile()

    assert run() == pytest.approx(run(), abs=1e-10)


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
        assignment = _sample_assignment(organs[i % len(organs)], level=0.3 + 0.6 * ((i * 7) % 5) / 5.0)
        sub.absorb(assignment)

    profile_final = sub.entropy_profile()
    for idx in bulk_internal_indices:
        assert profile_final[idx] != pytest.approx(profile_initial[idx], abs=1e-9), (
            f"cut index {idx} never moved from its initial value across 120 absorptions "
            "-- this is exactly the frozen-bulk-site bug found by review"
        )


def test_absorb_at_level_zero_applies_no_gate_but_counts_the_atom() -> None:
    # #3838: a measured input at rest (level 0) must not perturb the state.
    sub = HeartbeatSubstrate(seed=5)
    before = sub.entropy_profile()
    for _ in range(25):
        sub.absorb(_sample_assignment("orion-biometrics", level=0.0))
    assert sub.tick_count == 25
    assert sub.entropy_profile() == pytest.approx(before, abs=1e-12)


def test_absorb_push_grows_with_level() -> None:
    # A larger measured level moves the state further than a smaller one.
    import numpy as np

    def moved(level: float) -> float:
        sub = HeartbeatSubstrate(seed=8)
        before = np.array(sub.entropy_profile())
        sub.absorb(_sample_assignment("orion-hub", level=level))
        return float(np.abs(np.array(sub.entropy_profile()) - before).sum())

    assert moved(0.1) < moved(0.9)


def test_decay_tick_relaxes_a_busy_state_to_rest() -> None:
    # The 07-28 thermalization check, at trajectory level, with no reheat:
    # silence alone must bring the seam entanglement down. Each site is
    # damped toward its own dominant eigenvector, so the rest state is a
    # near-product state (seam entropy ~0), not the random initial state.
    sub = HeartbeatSubstrate(seed=13)
    sub.compute_relaxation_targets(gamma=0.2)
    for i in range(200):
        sub.absorb(_sample_assignment(["orion-hub", "orion-bus", "orion-cortex-exec"][i % 3], level=0.9))
    busy = sub.entropy_profile()[4]
    assert busy > 0.5
    for _ in range(300):
        sub.decay_tick(decay_prob=0.15)
    assert sub.entropy_profile()[4] < 0.05 * busy
