"""Unit tests for orion/curiosity/value.py: the spend log's scoring and the
value-ordered offer. Pure, no I/O."""

from __future__ import annotations

import math

import pytest

from orion.curiosity.value import (
    ARM_UNCERTAINTY_ORDER,
    ARM_VALUE_ORDER,
    COLD_POOL_YIELD,
    KIND_FORMED,
    KIND_MOVED_UNTESTED,
    KIND_TESTED,
    UNKNOWN_NO_END_SNAPSHOT,
    UNKNOWN_NO_RUN_ID,
    UNKNOWN_NO_SCORABLE_TEST,
    UNKNOWN_NO_START_SNAPSHOT,
    UNKNOWN_START_STAMPED,
    PriorState,
    PriorTestRecord,
    build_yield_model,
    diff_snapshots,
    entropy_nats,
    index_states,
    kl_nats,
    offer_arm,
    prior_tests_from_rows,
    raw_yield,
    revision_agreement,
    valid_confidence,
)

RUN = "abc123def456"
OTHER = "fedcba654321"


def _state(pid, conf, tested, *, last_run_id="", run_id="", status="open"):
    return PriorState(pid, conf, tested, status, last_run_id, run_id)


# --- the unit -------------------------------------------------------------------


def test_entropy_is_ln2_at_half_and_for_unknown_confidence() -> None:
    assert entropy_nats(0.5) == pytest.approx(math.log(2))
    assert entropy_nats(None) == pytest.approx(math.log(2))
    assert entropy_nats(0.9) == pytest.approx(0.3251, abs=1e-4)
    assert entropy_nats(0.95) == pytest.approx(0.1985, abs=1e-4)


def test_entropy_is_symmetric_and_clamped_at_the_edges() -> None:
    assert entropy_nats(0.2) == pytest.approx(entropy_nats(0.8))
    assert entropy_nats(0.0) == entropy_nats(0.01) > 0.0
    assert entropy_nats(1.0) == entropy_nats(0.99) > 0.0


@pytest.mark.parametrize("broken", [1.7, -0.2, float("nan"), float("inf"), "x", True])
def test_entropy_reads_a_broken_confidence_as_unknown_not_near_certain(broken) -> None:
    # Review finding: entropy_nats(1.7) clamped to 0.99 and read as nearly
    # certain, while Prior.uncertainty read it as maximally uncertain -- so
    # cold-start identity depended on every caller remembering to validate.
    assert entropy_nats(broken) == entropy_nats(None) == pytest.approx(math.log(2))


def test_kl_rests_at_exactly_zero_when_nothing_moved() -> None:
    assert kl_nats(0.55, 0.55) == 0.0
    assert kl_nats(1.0, 0.99) == 0.0  # equal after the clamp


def test_kl_known_values() -> None:
    assert kl_nats(0.92, 0.95) == pytest.approx(0.0081, abs=1e-4)
    assert kl_nats(0.8, 0.5) == pytest.approx(0.1927, abs=1e-4)
    assert kl_nats(0.05, 0.95) == pytest.approx(2.65, abs=1e-2)
    assert kl_nats(0.3, 0.6) > 0.0


@pytest.mark.parametrize(
    "raw,expected",
    [(0.55, 0.55), ("0.55", 0.55), (1, 1.0), (0, 0.0), (None, None), ("x", None),
     (1.7, None), (-0.1, None), (float("nan"), None), (True, None)],
)
def test_valid_confidence(raw, expected) -> None:
    assert valid_confidence(raw) == expected


def test_prior_state_reads_falkordb_string_numbers() -> None:
    state = PriorState.from_json(
        {"prior_id": "p1", "confidence": "0.55", "times_tested": "3", "last_run_id": RUN}
    )
    assert state == PriorState("p1", 0.55, 3, "", RUN, "")
    assert PriorState.from_json({"confidence": 0.5}) is None


def test_index_states_keeps_the_most_tested_fork() -> None:
    idx = index_states([_state("p1", 0.6, 1), _state("p1", 0.7, 6), _state("p2", 0.5, 0)])
    assert idx["p1"].times_tested == 6
    assert set(idx) == {"p1", "p2"}


@pytest.mark.parametrize("raw,expected", [("2.0", 2), (2.0, 2), ("3", 3), ("inf", 0), ("nan", 0), ("x", 0), (None, 0)])
def test_prior_state_reads_tested_counts_the_way_the_offer_does(raw, expected) -> None:
    # Review finding: int("2.0") raised and read as 0 here while the offer's
    # `_as_int` read it as 2, so one prior had two tested counts.
    assert PriorState.from_json({"prior_id": "p1", "times_tested": raw}).times_tested == expected


def test_index_states_breaks_fork_ties_on_values_not_read_order() -> None:
    # Two copies with the same tested count: the start and end snapshots of one
    # run must pick the same one whatever order FalkorDB returns them in.
    a, b = _state("p1", 0.6, 2), _state("p1", 0.8, 2)
    assert index_states([a, b])["p1"] == index_states([b, a])["p1"] == b


# --- what a run bought ----------------------------------------------------------


def test_a_tested_prior_that_moved_scores_its_kl() -> None:
    before = {"p1": _state("p1", 0.95, 2)}
    after = {"p1": _state("p1", 0.92, 3, last_run_id=RUN)}
    out = diff_snapshots(before, after, run_id=RUN)
    assert out.realized_nats == pytest.approx(kl_nats(0.92, 0.95))
    assert (out.n_tested, out.n_moved) == (1, 1)
    assert out.per_prior[0].kind == KIND_TESTED


def test_a_tested_prior_that_did_not_move_scores_zero_not_unknown() -> None:
    # Inconclusive is a real answer: times_tested bumps, confidence stays.
    before = {"p1": _state("p1", 0.55, 0)}
    after = {"p1": _state("p1", 0.55, 1, last_run_id=RUN)}
    out = diff_snapshots(before, after, run_id=RUN)
    assert out.realized_nats == 0.0
    assert (out.n_tested, out.n_moved) == (1, 0)
    assert out.per_prior[0].nats == 0.0


def test_a_formed_prior_is_counted_not_scored() -> None:
    out = diff_snapshots({}, {"p9": _state("p9", 0.55, 0, run_id=RUN)}, run_id=RUN)
    assert out.n_formed == 1
    assert out.realized_nats == 0.0
    assert out.per_prior[0].kind == KIND_FORMED


def test_changes_stamped_by_another_run_are_unattributed_not_scored() -> None:
    before = {"p1": _state("p1", 0.5, 0), "p2": _state("p2", 0.5, 0)}
    after = {
        "p1": _state("p1", 0.9, 1, last_run_id=OTHER),  # another run's test
        "p2": _state("p2", 0.8, 1),  # never stamped at all
        "p3": _state("p3", 0.55, 0, run_id=OTHER),  # formed elsewhere
    }
    out = diff_snapshots(before, after, run_id=RUN)
    assert out.n_unattributed == 3
    assert out.n_tested == 0
    assert out.realized_nats == 0.0


def test_a_move_without_a_test_count_is_protocol_drift_not_summed() -> None:
    before = {"p1": _state("p1", 0.5, 1)}
    after = {"p1": _state("p1", 0.8, 1, last_run_id=RUN)}
    out = diff_snapshots(before, after, run_id=RUN)
    assert out.n_moved_untested == 1
    assert out.realized_nats == 0.0
    assert out.per_prior[0].kind == KIND_MOVED_UNTESTED
    assert out.per_prior[0].nats == pytest.approx(kl_nats(0.8, 0.5))


def test_an_unreadable_snapshot_is_unknown_never_zero() -> None:
    start = diff_snapshots(None, {}, run_id=RUN)
    end = diff_snapshots({}, None, run_id=RUN)
    assert (start.realized_nats, start.unknown_reason) == (None, UNKNOWN_NO_START_SNAPSHOT)
    assert (end.realized_nats, end.unknown_reason) == (None, UNKNOWN_NO_END_SNAPSHOT)


def test_a_known_result_carries_no_unknown_reason() -> None:
    out = diff_snapshots({"p1": _state("p1", 0.55, 0)}, {"p1": _state("p1", 0.55, 1, last_run_id=RUN)}, run_id=RUN)
    assert (out.realized_nats, out.unknown_reason) == (0.0, None)


def test_a_start_already_stamped_by_this_run_is_unknown_not_partial() -> None:
    # An earlier attempt of this run moved p1 0.5 -> 0.7 before this "start"
    # was taken; the retry moved it on to 0.75. Scoring 0.7 -> 0.75 would
    # report part of the run as all of it.
    before = {"p1": _state("p1", 0.7, 1, last_run_id=RUN)}
    after = {"p1": _state("p1", 0.75, 2, last_run_id=RUN)}
    out = diff_snapshots(before, after, run_id=RUN)
    assert (out.realized_nats, out.unknown_reason) == (None, UNKNOWN_START_STAMPED)
    formed = {"p9": _state("p9", 0.55, 0, run_id=RUN)}
    assert diff_snapshots(formed, formed, run_id=RUN).unknown_reason == UNKNOWN_START_STAMPED


def test_no_run_id_attributes_nothing() -> None:
    # Unstamped priors carry last_run_id "": an empty run id would claim them.
    before = {"p1": _state("p1", 0.5, 0)}
    after = {"p1": _state("p1", 0.9, 1)}
    out = diff_snapshots(before, after, run_id="")
    assert (out.realized_nats, out.unknown_reason) == (None, UNKNOWN_NO_RUN_ID)


def test_tested_but_nothing_scorable_is_unknown_not_zero() -> None:
    # Review finding: a run whose only test wrote an invalid confidence summed
    # nothing and recorded 0.0 -- "tested, nothing moved", which it was not.
    before = {"p1": _state("p1", 0.5, 0)}
    after = {"p1": _state("p1", None, 1, last_run_id=RUN)}
    out = diff_snapshots(before, after, run_id=RUN)
    assert (out.n_tested, out.n_invalid_confidence) == (1, 1)
    assert (out.realized_nats, out.unknown_reason) == (None, UNKNOWN_NO_SCORABLE_TEST)
    # Repairing a broken confidence is unscorable too (unusable BEFORE the test).
    repaired = diff_snapshots(
        {"p1": _state("p1", None, 0)}, {"p1": _state("p1", 0.8, 1, last_run_id=RUN)}, run_id=RUN
    )
    assert (repaired.realized_nats, repaired.unknown_reason) == (None, UNKNOWN_NO_SCORABLE_TEST)


def test_an_invalid_confidence_is_flagged_and_left_out_of_the_sum() -> None:
    before = {"p1": _state("p1", None, 0), "p2": _state("p2", 0.5, 0)}
    after = {
        "p1": _state("p1", 0.7, 1, last_run_id=RUN),
        "p2": _state("p2", 0.8, 1, last_run_id=RUN),
    }
    out = diff_snapshots(before, after, run_id=RUN)
    assert out.n_invalid_confidence == 1
    assert out.n_tested == 2
    assert out.realized_nats == pytest.approx(kl_nats(0.8, 0.5))


def test_revision_agreement() -> None:
    before = {"p1": _state("p1", 0.5, 0), "p2": _state("p2", 0.6, 0)}
    after = {
        "p1": _state("p1", 0.8, 1, last_run_id=RUN),
        "p2": _state("p2", 0.4, 1, last_run_id=RUN),
    }
    out = diff_snapshots(before, after, run_id=RUN)
    assert revision_agreement(out, {"p1": (0.5, 0.8), "p2": (0.6, 0.4)}) == 1.0
    assert revision_agreement(out, {"p1": (0.5, 0.8)}) == 0.5
    assert revision_agreement(out, {}) == 0.0
    unmoved = diff_snapshots({"p1": _state("p1", 0.5, 0)}, {"p1": _state("p1", 0.5, 1, last_run_id=RUN)}, run_id=RUN)
    assert revision_agreement(unmoved, {}) is None


# --- what the next run is expected to buy ---------------------------------------


def _tests(pid, trajectory):
    return [PriorTestRecord(pid, a, b) for a, b in zip(trajectory, trajectory[1:])]


def test_cold_start_yield_is_one_so_expected_value_is_entropy() -> None:
    model = build_yield_model([], window=3, pseudo_tests=2.0)
    assert model.pool_yield == COLD_POOL_YIELD
    assert model.yield_for("anything") == 1.0
    assert model.expected_nats("anything", 0.7) == pytest.approx(entropy_nats(0.7))


def test_net_progress_not_surprise_a_flip_flopping_prior_loses_value() -> None:
    oscillating = _tests("osc", [0.55, 0.7, 0.55, 0.7])
    steady = _tests("steady", [0.55, 0.7, 0.85, 0.95])
    osc_gross = sum(kl_nats(t.after, t.before) for t in oscillating)
    steady_gross = sum(kl_nats(t.after, t.before) for t in steady)
    assert osc_gross > 0.1  # every flip is surprising...
    # ...and teaches ~nothing: net KL 0.047 over 1.99 nats offered, x 1/3 straightness.
    assert raw_yield(oscillating) == pytest.approx(0.0079, abs=1e-3)
    assert raw_yield(steady) > 20 * raw_yield(oscillating)
    assert steady_gross > 0  # sanity


def test_straightness_discounts_back_and_forth_whatever_the_window_parity() -> None:
    from orion.curiosity.value import straightness

    assert straightness(_tests("s", [0.55, 0.7, 0.85])) == pytest.approx(1.0)
    assert straightness(_tests("f", [0.7, 0.3, 0.7, 0.3])) == pytest.approx(1 / 3)
    assert straightness(_tests("u", [0.5, 0.5, 0.5])) == 1.0
    # A 0.3 <-> 0.7 flip-flop nets a full swing over an odd window; without the
    # straightness discount it would out-yield a belief that is really learning.
    odd_flip = _tests("f", [0.7, 0.3, 0.7, 0.3])
    learner = _tests("l", [0.55, 0.775, 0.8875, 0.94375])
    assert raw_yield(learner) > 3 * raw_yield(odd_flip)


def test_yield_counts_only_moves_a_recorded_test_made() -> None:
    # Review finding: net displacement was last.after - first.before, so a
    # jump BETWEEN two scored tests (another run, an unstamped edit, a run
    # scored unknown) read as this prior's progress: [0.5->0.5, 0.9->0.9]
    # yielded 0.361 and [0.5->0.6, 0.9->0.9] had straightness 4.0.
    from orion.curiosity.value import straightness

    gap_only = [PriorTestRecord("p", 0.5, 0.5), PriorTestRecord("p", 0.9, 0.9)]
    assert raw_yield(gap_only) == 0.0
    small_then_gap = [PriorTestRecord("p", 0.5, 0.6), PriorTestRecord("p", 0.9, 0.9)]
    assert straightness(small_then_gap) == pytest.approx(1.0)
    # Progress is the 0.5 -> 0.6 test alone; both tests still offered their
    # own uncertainty (H at 0.5 and at 0.9).
    assert raw_yield(small_then_gap) == pytest.approx(
        kl_nats(0.6, 0.5) / (entropy_nats(0.5) + entropy_nats(0.9))
    )
    # One such prior no longer lifts the whole pool.
    stuck = _tests("a", [0.6, 0.6, 0.6, 0.6])
    assert build_yield_model(stuck + gap_only, window=3, pseudo_tests=2.0).pool_yield == 0.0


def test_progress_that_was_undone_between_tests_is_not_credited() -> None:
    # Review finding (second round): summing only recorded moves credited a
    # belief the tests keep pushing 0.5 -> 0.7 while something off the record
    # knocks it back each time (a self-inquiry turn, an unscored run): raw
    # yield 0.306 -- 4x a genuine learner -- and the pool lifted 23x. Only
    # progress a test made AND that stuck is credited now.
    from orion.curiosity.value import straightness

    undone = [PriorTestRecord("p", 0.5, 0.7)] * 3
    assert straightness(undone) == pytest.approx(0.2)  # the gaps are path, not progress
    assert raw_yield(undone) == pytest.approx(kl_nats(0.7, 0.5) * 0.2 / (3 * math.log(2)))
    learner = _tests("l", [0.55, 0.775, 0.8875, 0.94375])
    assert raw_yield(learner) > 30 * raw_yield(undone)
    stuck = [PriorTestRecord(f"s{i}", 0.6, 0.6) for i in range(5) for _ in range(3)]
    assert build_yield_model(stuck + undone, window=3, pseudo_tests=2.0).pool_yield < 0.002
    # Tests pushed up, something else pushed it further down: nothing credited.
    assert raw_yield([PriorTestRecord("p", 0.5, 0.6), PriorTestRecord("p", 0.3, 0.35)]) == 0.0


def test_credited_progress_stays_inside_where_the_belief_actually_went() -> None:
    # Property check over random windows with random jumps between tests: the
    # credited end point lies between the first tested value and the last
    # observed one, never past what either the tests or the belief did, and
    # straightness stays in [0, 1].
    import random

    from orion.curiosity.value import _credited_net, straightness

    rng = random.Random(20260926)
    for _ in range(5000):
        tests, value = [], rng.random()
        for _ in range(rng.randint(1, 4)):
            before = value if rng.random() < 0.5 else rng.random()  # sometimes a jump
            value = rng.random()
            tests.append(PriorTestRecord("p", before, value))
        recorded = sum(t.after - t.before for t in tests)
        observed = tests[-1].after - tests[0].before
        credited = _credited_net(tests)
        assert abs(credited) <= abs(recorded) + 1e-12 and abs(credited) <= abs(observed) + 1e-12
        assert credited == 0.0 or (credited > 0) == (observed > 0) == (recorded > 0)
        end = tests[0].before + credited
        lo, hi = sorted((tests[0].before, tests[-1].after))
        assert lo - 1e-12 <= end <= hi + 1e-12
        assert 0.0 <= straightness(tests) <= 1.0


def test_shrinkage_pulls_thin_evidence_toward_the_pool() -> None:
    history = _tests("a", [0.5, 0.9]) + _tests("b", [0.5, 0.5])
    model = build_yield_model(history, window=3, pseudo_tests=2.0)
    raw_a = raw_yield(_tests("a", [0.5, 0.9]))
    assert model.yield_for("a") == pytest.approx((1 * raw_a + 2 * model.pool_yield) / 3)
    assert model.yield_for("never_tested") == model.pool_yield
    no_shrink = build_yield_model(history, window=3, pseudo_tests=0.0)
    assert no_shrink.yield_for("a") == pytest.approx(raw_a)


def test_pool_yield_is_a_ratio_of_sums() -> None:
    history = _tests("a", [0.5, 0.9]) + _tests("b", [0.5, 0.5])
    model = build_yield_model(history, window=3, pseudo_tests=2.0)
    expected = kl_nats(0.9, 0.5) / (entropy_nats(0.5) + entropy_nats(0.5))
    assert model.pool_yield == pytest.approx(expected)


def test_only_the_last_window_tests_count() -> None:
    early_progress = _tests("p", [0.5, 0.95]) + _tests("p", [0.95, 0.95]) * 3
    model = build_yield_model(early_progress, window=3, pseudo_tests=0.0)
    assert model.yield_for("p") == 0.0  # learned: the last 3 tests moved nothing


def test_prior_tests_from_rows_keeps_only_scorable_tests() -> None:
    rows = [
        {"prior_id": "p1", "kind": "tested", "before": 0.5, "after": 0.7},
        {"prior_id": "p2", "kind": "formed", "before": None, "after": 0.55},
        {"prior_id": "p3", "kind": "tested", "before": None, "after": 0.7},
        {"prior_id": "p4", "kind": "moved_untested", "before": 0.5, "after": 0.6},
    ]
    assert prior_tests_from_rows(rows) == [PriorTestRecord("p1", 0.5, 0.7)]


# --- the arm ---------------------------------------------------------------------


def test_offer_arm_is_off_by_default_shape() -> None:
    assert offer_arm(RUN, enabled=False, propensity=0.5) == (ARM_UNCERTAINTY_ORDER, 0.0)
    assert offer_arm(RUN, enabled=True, propensity=0.0) == (ARM_UNCERTAINTY_ORDER, 0.0)
    assert offer_arm(RUN, enabled=True, propensity=1.0) == (ARM_VALUE_ORDER, 1.0)


def test_offer_arm_is_deterministic_per_run_and_roughly_fair() -> None:
    assert offer_arm(RUN, enabled=True, propensity=0.5) == offer_arm(RUN, enabled=True, propensity=0.5)
    ids = [f"{i:012x}" for i in range(2000)]
    share = sum(offer_arm(i, enabled=True, propensity=0.5)[0] == ARM_VALUE_ORDER for i in ids) / len(ids)
    assert 0.45 < share < 0.55


def test_offer_arm_at_propensity_one_is_always_the_value_arm() -> None:
    # The draw is in [0, 1) -- its largest value is 0xffffffff / 2**32 -- so
    # a recorded propensity of 1.0 can never have assigned the other arm.
    assert 0xFFFFFFFF / float(2**32) < 1.0
    ids = [f"{i:012x}" for i in range(5000)]
    assert all(offer_arm(i, enabled=True, propensity=1.0) == (ARM_VALUE_ORDER, 1.0) for i in ids)
