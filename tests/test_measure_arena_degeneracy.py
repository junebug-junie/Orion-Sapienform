"""Deterministic unit tests for measure_arena_degeneracy.py.

No DB, no network. Same sibling-module-by-file-path loading pattern as
test_measure_society_of_mind_magnitude_probe.py.

Lives in top-level `tests/`, not `scripts/analysis/tests/` -- `scripts` is in
`pyproject.toml`'s `norecursedirs`, so a bare `pytest` run from repo root never
discovers anything placed there.

Every fixture below is hand-computed: the expected value is derived by hand in
the test's own comment/name, never by running the function and pasting what it
said. The point of this instrument is to be trusted about a degeneracy claim, so
its tests must be able to fail.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "measure_arena_degeneracy.py"
)
_spec = importlib.util.spec_from_file_location("measure_arena_degeneracy", _MODULE_PATH)
mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules["measure_arena_degeneracy"] = mod
_spec.loader.exec_module(mod)


# ---------------------------------------------------------------------------
# template_key_from_proposal_id
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        # Real live proposal id shape, copied from a stored frame.
        (
            "proposal:inspect_bus_channel_catalog:tick_5b0822b7faf3:attention.frame:tick_x:p.v1",
            "inspect_bus_channel_catalog",
        ),
        # Real live dispatch id shape: the template sits one field later.
        (
            "dispatch:proposal:prune_build_cache:tick_b6174dd04993:attention.frame:tick_x:e.v1",
            "prune_build_cache",
        ),
        # Real live policy decision id shape: two prefixes before the marker.
        (
            "policy.decision:proposal:watch_reliability:tick_abc:attention.frame:tick_x:s.v1",
            "watch_reliability",
        ),
        # Real live result_ref shape: three prefixes.
        (
            "result:dispatch:proposal:summarize_loaded_state:tick_abc:attention.frame:x:e.v1",
            "summarize_loaded_state",
        ),
    ],
)
def test_template_key_parses_every_live_id_shape(raw, expected):
    assert mod.template_key_from_proposal_id(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        None,
        123,
        "",
        "nothing_here",
        # `proposal` present but nothing after it -- must not IndexError.
        "dispatch:proposal",
        # `proposal` present, but the next field is empty.
        "proposal::tick_abc",
    ],
)
def test_template_key_refuses_to_guess(raw):
    """An unrecognised id must be reported, never attributed to some template.

    A silent misattribution here would corrupt section D's starvation table --
    which is the table the whole arc's conclusions rest on.
    """
    assert mod.template_key_from_proposal_id(raw) == mod.UNPARSEABLE_TEMPLATE


def test_template_key_uses_the_last_proposal_marker():
    """A template literally named `proposal` must not shadow the real marker.

    Hand-derived: fields are [dispatch, proposal, proposal, tick_a]. The LAST
    `proposal` is at index 2, so the template key is index 3 = 'tick_a'. If the
    parser used the FIRST marker it would return 'proposal' instead.
    """
    assert mod.template_key_from_proposal_id("dispatch:proposal:proposal:tick_a") == "tick_a"


# ---------------------------------------------------------------------------
# pearson
# ---------------------------------------------------------------------------


def test_pearson_perfect_positive_is_exactly_one():
    """y = 2x + 1 is a perfect linear relation; r must be 1.0 to float precision.

    This is the exact case section B exists to detect (corr == 1.0 means two
    'competing' signals are one signal), so it must be exact, not approximate.
    """
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [3.0, 5.0, 7.0, 9.0, 11.0]
    assert mod.pearson(xs, ys) == pytest.approx(1.0, abs=1e-12)


def test_pearson_perfect_negative():
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [4.0, 3.0, 2.0, 1.0]
    assert mod.pearson(xs, ys) == pytest.approx(-1.0, abs=1e-12)


def test_pearson_hand_computed_partial_correlation():
    """Hand-computed, not machine-derived.

    xs = [1,2,3,4], ys = [1,3,2,4].  mean_x = mean_y = 2.5.
    dx = [-1.5,-0.5,0.5,1.5]; dy = [-1.5,0.5,-0.5,1.5]
    cov  = 2.25 - 0.25 - 0.25 + 2.25 = 4.0
    var_x = var_y = 2.25+0.25+0.25+2.25 = 5.0
    r = 4.0 / sqrt(5.0*5.0) = 4.0/5.0 = 0.8
    """
    assert mod.pearson([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(0.8, abs=1e-12)


def test_pearson_returns_none_for_a_constant_series():
    """A constant series has an UNDEFINED correlation, not a zero one.

    Reporting 0.0 would read as 'measured, and they are unrelated' when the
    truth is 'one of these never moved' -- the exact confusion this instrument
    exists to eliminate.
    """
    assert mod.pearson([1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]) is None
    assert mod.pearson([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]) is None


def test_pearson_returns_none_on_degenerate_input():
    assert mod.pearson([], []) is None
    assert mod.pearson([1.0], [1.0]) is None
    assert mod.pearson([1.0, 2.0], [1.0]) is None


# ---------------------------------------------------------------------------
# run_lengths
# ---------------------------------------------------------------------------


def test_run_lengths_hand_computed():
    """['a','a','b','a','a','a'] -> runs of 2, 1, 3. Sum must equal input length."""
    assert mod.run_lengths(["a", "a", "b", "a", "a", "a"]) == [2, 1, 3]


def test_run_lengths_all_distinct_is_all_ones():
    assert mod.run_lengths(["a", "b", "c"]) == [1, 1, 1]


def test_run_lengths_all_identical_is_one_run():
    assert mod.run_lengths(["a"] * 7) == [7]


def test_run_lengths_empty():
    assert mod.run_lengths([]) == []


# ---------------------------------------------------------------------------
# measure_urgency_resolution -- section A
# ---------------------------------------------------------------------------


def _cand(template: str, urgency: float, priority: float = 0.5) -> dict:
    return {
        "proposal_id": f"proposal:{template}:tick_abc:attention.frame:tick_abc:p.v1",
        "urgency_score": urgency,
        "priority_score": priority,
    }


def test_urgency_resolution_detects_a_fully_degenerate_frame():
    """4 candidates, all the same urgency -> 1 distinct value, 4 tied at max.

    This is the shipped-live shape the instrument was written to expose.
    """
    frames = [[_cand("a", 0.9), _cand("b", 0.9), _cand("c", 0.9), _cand("d", 0.9)]]
    result = mod.measure_urgency_resolution(frames)
    assert result.frames == 1
    assert result.candidates == 4
    assert result.distinct_urgency_histogram == {1: 1}
    assert result.tied_at_max == [4]


def test_urgency_resolution_detects_a_healthy_frame():
    """4 candidates, 4 distinct urgencies -> 4 distinct values, 1 tied at max.

    The inverse case. Without this, a bug that always reported degeneracy would
    pass the test above.
    """
    frames = [[_cand("a", 0.9), _cand("b", 0.7), _cand("c", 0.5), _cand("d", 0.3)]]
    result = mod.measure_urgency_resolution(frames)
    assert result.distinct_urgency_histogram == {4: 1}
    assert result.tied_at_max == [1]


def test_urgency_resolution_counts_partial_ties_at_max():
    """[0.9, 0.9, 0.4] -> 2 distinct values, 2 tied at the max of 0.9."""
    frames = [[_cand("a", 0.9), _cand("b", 0.9), _cand("c", 0.4)]]
    result = mod.measure_urgency_resolution(frames)
    assert result.distinct_urgency_histogram == {2: 1}
    assert result.tied_at_max == [2]


def test_urgency_resolution_rounds_float_noise_to_the_same_value():
    """Values equal to 9dp must count as ONE distinct value, not two.

    A JSON round-trip can perturb the last bit of two numbers that were computed
    identically; counting those as distinct would understate the degeneracy.
    """
    frames = [[_cand("a", 0.9), _cand("b", 0.9 + 1e-13)]]
    result = mod.measure_urgency_resolution(frames)
    assert result.distinct_urgency_histogram == {1: 1}


def test_urgency_resolution_skips_frames_with_no_readable_urgency():
    frames = [[], [{"proposal_id": "proposal:a:tick:x:y"}], [_cand("a", 0.5)]]
    result = mod.measure_urgency_resolution(frames)
    assert result.frames == 1
    assert result.frames_missing_urgency == 2


# ---------------------------------------------------------------------------
# measure_cross_template_correlation -- section B
# ---------------------------------------------------------------------------


def test_cross_template_correlation_flags_an_identical_signal():
    """Two templates carrying the same number every frame must report corr 1.0."""
    frames = [
        [_cand("alpha", 0.1 * i, 0.2 * i), _cand("beta", 0.1 * i, 0.3 * i)] for i in range(1, 21)
    ]
    pairs, templates = mod.measure_cross_template_correlation(frames, top_n=8, min_pairs=10)
    assert set(templates) == {"alpha", "beta"}
    assert len(pairs) == 1
    assert pairs[0].corr_urgency == pytest.approx(1.0, abs=1e-12)
    assert pairs[0].n == 20


def test_cross_template_correlation_respects_min_pairs():
    """Under the threshold, report nothing rather than an artifact.

    Directly encodes the short-window-distribution-stats lesson: a correlation
    over a handful of samples has repeatedly been wrong in this repo.
    """
    frames = [[_cand("alpha", 0.1 * i), _cand("beta", 0.2 * i)] for i in range(1, 6)]
    pairs, _ = mod.measure_cross_template_correlation(frames, top_n=8, min_pairs=100)
    assert pairs == []


def test_cross_template_correlation_reports_undefined_for_a_pinned_template():
    """A template whose urgency never moves must yield None, not 0.0."""
    frames = [[_cand("alpha", 0.1 * i), _cand("pinned", 0.5)] for i in range(1, 21)]
    pairs, _ = mod.measure_cross_template_correlation(frames, top_n=8, min_pairs=10)
    assert len(pairs) == 1
    assert pairs[0].corr_urgency is None


def test_cross_template_correlation_ignores_unparseable_ids():
    frames = [
        [_cand("alpha", 0.1 * i), {"proposal_id": "garbage", "urgency_score": 0.5}]
        for i in range(1, 21)
    ]
    _, templates = mod.measure_cross_template_correlation(frames, top_n=8, min_pairs=10)
    assert templates == ["alpha"]


# ---------------------------------------------------------------------------
# measure_decision_redundancy -- section C
# ---------------------------------------------------------------------------


def _decision(template: str, decision: str) -> dict:
    return {
        "proposal_id": f"proposal:{template}:tick_abc:attention.frame:tick_abc:p.v1",
        "decision": decision,
    }


def test_decision_redundancy_counts_identical_runs():
    """3 identical frames then 2 different ones -> 2 runs over 5 frames.

    Hand-derived: fingerprints are [X, X, X, Y, Y] -> runs [3, 2], so
    total_frames - n_runs = 5 - 2 = 3 frames carried no new decision.
    """
    same = [_decision("a", "approved_read_only"), _decision("b", "review_required")]
    other = [_decision("a", "review_required"), _decision("b", "review_required")]
    result = mod.measure_decision_redundancy([same, same, same, other, other])
    assert result.total_frames == 5
    assert result.runs == [3, 2]
    assert result.distinct_fingerprints == 2
    assert result.empty_frames == 0


def test_decision_redundancy_is_order_insensitive_within_a_frame():
    """The same decisions in a different order are the SAME decision set.

    Dispatch consumes a set of approvals, not an ordered list, so two frames
    differing only in ordering carry no new information.
    """
    a = [_decision("a", "approved_read_only"), _decision("b", "review_required")]
    b = [_decision("b", "review_required"), _decision("a", "approved_read_only")]
    result = mod.measure_decision_redundancy([a, b])
    assert result.runs == [2]
    assert result.distinct_fingerprints == 1


def test_decision_redundancy_distinguishes_a_changed_decision():
    """Same templates, one flipped decision -> a genuinely new frame.

    The inverse of the test above: without it, a fingerprint that ignored the
    decision entirely would still pass.
    """
    a = [_decision("a", "approved_read_only")]
    b = [_decision("a", "rejected")]
    result = mod.measure_decision_redundancy([a, b])
    assert result.runs == [1, 1]
    assert result.distinct_fingerprints == 2


def test_decision_redundancy_counts_empty_frames_as_a_run():
    """An unbroken run of empty frames is exactly as redundant as identical ones."""
    result = mod.measure_decision_redundancy([[], [], [_decision("a", "approved_read_only")]])
    assert result.empty_frames == 2
    assert result.runs == [2, 1]
    assert result.distinct_fingerprints == 2


# ---------------------------------------------------------------------------
# measure_dispatch_reachability -- section D
# ---------------------------------------------------------------------------


def test_dispatch_reachability_separates_dispatched_from_blocked():
    """Proposed 2, dispatched 1, blocked 1 -- from the two separate live lists.

    Reading a single `candidates` list with a `status` field (which the frame
    does not have) produced an all-zero table that was indistinguishable from
    total starvation. This test locks the real shape in.
    """
    proposals = [[_cand("alpha", 0.5), _cand("beta", 0.5)]]
    dispatched = [{"dispatch_id": "dispatch:proposal:alpha:tick_a:x:y"}]
    blocked = [
        {
            "dispatch_id": "dispatch:proposal:beta:tick_a:x:y",
            "reasons": ["max_dispatch_candidates_exceeded"],
        }
    ]
    reach = mod.measure_dispatch_reachability(proposals, dispatched, blocked)
    assert reach["alpha"].proposed == 1
    assert reach["alpha"].dispatched == 1
    assert reach["alpha"].blocked == 0
    assert reach["beta"].dispatched == 0
    assert reach["beta"].blocked == 1
    assert reach["beta"].blocked_reasons["max_dispatch_candidates_exceeded"] == 1


def test_dispatch_reachability_falls_back_to_result_ref():
    """A candidate with no dispatch_id must still be attributed via result_ref."""
    dispatched = [{"result_ref": "result:dispatch:proposal:gamma:tick_a:x:y"}]
    reach = mod.measure_dispatch_reachability([], dispatched, [])
    assert reach["gamma"].dispatched == 1


def test_dispatch_reachability_records_a_starved_template():
    """Proposed but never dispatched -- the shape the whole arc is about."""
    proposals = [[_cand("starved", 0.5)] for _ in range(50)]
    reach = mod.measure_dispatch_reachability(proposals, [], [])
    assert reach["starved"].proposed == 50
    assert reach["starved"].dispatched == 0


# ---------------------------------------------------------------------------
# percentile
# ---------------------------------------------------------------------------


def test_percentile_nearest_rank_hand_computed():
    """[1..10], q=0.5 -> ceil(0.5*10)=5th value (1-indexed) = 5.

    q=0.95 -> ceil(9.5)=10th = 10. q=0.0 clamps to the first = 1.
    """
    values = [float(i) for i in range(1, 11)]
    assert mod.percentile(values, 0.5) == 5.0
    assert mod.percentile(values, 0.95) == 10.0
    assert mod.percentile(values, 0.0) == 1.0


def test_percentile_empty_is_none():
    assert mod.percentile([], 0.5) is None
