"""Personality rev 4 Move 2: care inside a family, the holdback arm, the runaway guard, the trace."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from orion.physiology import personality_values as pv
from orion.physiology.appetite import FamilyAppetite, rank_value

DIM = 8


def _vec(*idx_vals):
    v = [0.0] * DIM
    for i, x in idx_vals:
        v[i] = x
    return v


def _tendency(tid="t1", axis=0, weight=10.0, cap=1.0):
    return pv.Tendency(tid, f"label {tid}", pv.unit(_vec((axis, 1.0))), weight, cap)


def test_holdback_arm_is_deterministic_and_near_twenty_percent():
    arms = [pv.holdback_arm(f"want-{i}") for i in range(20000)]
    share = arms.count(pv.ARM_HOLDBACK) / len(arms)
    assert 0.185 < share < 0.215
    assert all(pv.holdback_arm(f"want-{i}") == arms[i] for i in range(200))
    assert pv.holdback_arm("x", 0.0) == pv.ARM_CARE and pv.holdback_arm("x", 1.0) == pv.ARM_HOLDBACK


def test_care_rests_at_exactly_zero():
    t = [_tendency()]
    assert pv.care_for(None, t) == pv.CareResult(0.0, pv.REASON_NO_TEXT)
    assert pv.care_for(_vec((0, 1.0)), []).reason == pv.REASON_NO_TENDENCY
    orth = pv.care_for(_vec((1, 1.0)), t)
    assert orth.care == 0.0 and orth.reason == pv.REASON_NO_MATCH
    opposed = pv.care_for(_vec((0, -1.0)), t)
    assert opposed.care == 0.0 and opposed.reason == pv.REASON_NO_MATCH


def test_care_is_cosine_times_weight_strength_never_above_one():
    t = [_tendency(weight=10.0)]
    r = pv.care_for(_vec((0, 3.0)), t)               # not unit on input: normalized first
    assert r.care == pytest.approx(0.5) and r.tendency_id == "t1" and r.cosine == pytest.approx(1.0)
    half = pv.care_for(_vec((0, 1.0), (1, 1.0)), t)
    assert half.care == pytest.approx(0.5 * math.sqrt(0.5))
    assert pv.care_for(_vec((0, 1.0)), [_tendency(weight=1e9)]).care <= 1.0
    assert pv.strength(0.0) == 0.0 and pv.strength(10.0) == 0.5


def test_the_best_tendency_wins_and_a_cap_scales_it():
    ts = [_tendency("a", 0, 10.0), _tendency("b", 1, 30.0)]
    v = _vec((0, 1.0), (1, 1.0))
    assert pv.care_for(v, ts).tendency_id == "b"
    capped = pv.apply_caps(ts, {"b": pv.Cap("b", 0.9, 0.6 / 0.9)})
    r = pv.care_for(v, capped)
    assert r.care == pytest.approx(max(math.sqrt(0.5) * 0.5, math.sqrt(0.5) * 0.75 * (0.6 / 0.9)))


def _app(values=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)):
    return FamilyAppetite(family="f", tau_sec=3600.0, tau_basis="x", last_close_at=None, n_closes=0, n_graded=0,
                          graded=False, mean_outcome=None, values=tuple(values))


def test_care_enters_value_rank_before_the_percentile():
    app = _app()
    base = rank_value(app, 0.45, None)
    assert rank_value(app, 0.45, None, care=0.0) == base
    lifted = rank_value(app, 0.45, None, care=1.0)        # 0.45 x 1.5 = 0.675
    assert lifted[0] > base[0] and lifted[1] == base[1]
    # No history: rank = intensity x multiplier, clamped to 1.
    assert rank_value(_app(()), 0.8, None, care=1.0)[0] == 1.0
    assert rank_value(_app(()), 0.4, None, care=1.0)[0] == pytest.approx(0.6)


def _shares(days_share, today):
    return {today - timedelta(days=i + 1): {"t": s} for i, s in enumerate(days_share)}


def test_runaway_cap_needs_every_one_of_seven_days_above_sixty_percent():
    today = date(2026, 11, 20)
    caps = pv.runaway_caps(_shares([0.8] * 7, today), today)
    assert caps["t"].factor == pytest.approx(0.75) and caps["t"].mean_share == pytest.approx(0.8)
    assert pv.runaway_caps(_shares([0.8] * 6 + [0.5], today), today) == {}
    assert pv.runaway_caps(_shares([0.8] * 6, today), today) == {}          # a day with nothing shown
    assert pv.runaway_caps(_shares([0.6] * 7, today), today) == {}          # at the line is not above it


def test_runaway_cap_survives_midnight_before_the_days_first_choice():
    today = date(2026, 11, 20)
    shares = _shares([0.8] * 7, today)
    assert today not in shares and "t" in pv.runaway_caps(shares, today)
    shares[today] = {"t": 0.1}                      # today's partial day never lifts or sets a cap
    assert "t" in pv.runaway_caps(shares, today)


def test_daily_shown_shares_counts_unmatched_options_in_the_total():
    d = date(2026, 11, 20)
    shares = pv.daily_shown_shares([(d, "t"), (d, "t"), (d, None), (d, "u")])
    assert shares[d] == {"t": 0.5, "u": 0.25}


def test_care_promoted_keys_are_in_the_top_k_only_with_care():
    with_care = {"a": 0.9, "b": 0.8, "c": 0.7}
    without = {"a": 0.9, "b": 0.5, "c": 0.7}
    assert pv.care_promoted_keys(with_care, without, 2) == {"b"}
    assert pv.care_promoted_keys(with_care, with_care, 2) == set()
    assert pv.care_promoted_keys(with_care, without, 0) == set()


def test_trace_round_trips_and_absent_trace_reads_as_the_net():
    term = pv.PersonalityTerm(arm=pv.ARM_CARE, care=0.25, tendency_id="t1", cosine=0.5, endorsement_rate=0.7,
                              endorsement_n=12)
    w = pv.trace_warnings(term, rank=0.6, rank_without_care=0.5, net=0.3, net_without_care=0.2)
    assert pv.trace_arm(w) == "care" and pv.trace_care(w) == pytest.approx(0.25)
    assert pv.net_without_care(w, 0.3) == pytest.approx(0.2)
    assert "personality:endorsement=0.7000:n=12" in w
    hold = pv.trace_warnings(pv.PersonalityTerm(arm=pv.ARM_HOLDBACK, reason=pv.REASON_HOLDBACK,
                                                endorsement_excluded="holdback"),
                             rank=0.5, rank_without_care=0.5, net=0.1, net_without_care=0.1)
    assert "personality:care=0:reason=holdback" in hold and "personality:endorsement=excluded:holdback" in hold
    assert pv.trace_care(hold) == 0.0
    assert pv.net_without_care([], 0.4) == 0.4 and pv.trace_arm([]) is None


def test_object_text_is_collapsed_and_cut():
    assert pv.object_text("  a \n b ") == "a b"
    assert pv.object_text("") is None and pv.object_text(None) is None
    assert len(pv.object_text("x" * 1000)) == pv.OBJECT_TEXT_MAX


def test_unit_rejects_degenerate_vectors():
    assert pv.unit([]) is None and pv.unit([0.0, 0.0]) is None and pv.unit([float("nan")]) is None
    assert pv.unit([3.0, 4.0]) == pytest.approx((0.6, 0.8))
