"""The three-arm readout (scripts/analysis/curiosity_three_arm_readout.py): who is in the test, the stop rule, the
pre-registered permutation test, the harm look and the fallback report (design 2026-11-04, patch 5). No DB."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import curiosity_three_arm_readout as ro  # noqa: E402

START = datetime(2026, 11, 5, 7, 0, tzinfo=timezone.utc)
THREE = {"value_order": 0.3333, "replica_order": 0.3333, "uncertainty_order": 0.3334}
TWO = {"value_order": 0.5, "replica_order": 0.0, "uncertainty_order": 0.5}
REP, VAL, UNC = "replica_order", "value_order", "uncertainty_order"


def run(i, arm, nats, *, hours=0.0, props=THREE, turn_ok=True, n_tested=2, plan="ok"):
    return {"run_id": f"r{i}", "decided_at": START + timedelta(hours=hours), "arm": arm, "arm_propensities": props,
            "turn_ok": turn_ok, "realized_nats": nats, "n_tested": n_tested,
            "n_moved": 1 if (nats or 0) > 0 else 0, "plan_status": plan if arm == REP else None}


def arms_of(n, replica, value, uncertainty):
    """n completed runs per arm; each arm's nats cycle through its list (mostly zeros, long tail)."""
    rows, i = [], 0
    for arm, vals in ((REP, replica), (VAL, value), (UNC, uncertainty)):
        for k in range(n):
            rows.append(run(i, arm, vals[k % len(vals)], hours=i * 0.1))
            i += 1
    return rows


def test_rows_from_before_the_replica_had_a_real_chance_are_left_out():
    rows = [run(1, UNC, 0.0, hours=-30, props=TWO), run(2, VAL, 0.5, hours=-29, props=TWO),
            run(3, VAL, 0.1, hours=-28, props=None), run(4, REP, 0.2), run(5, UNC, 0.0, hours=1, props=dict(THREE))]
    live = ro.three_arm_rows(rows)
    assert [r["run_id"] for r in live] == ["r4", "r5"]
    assert ro.replica_propensity({"arm_propensities": '{"replica_order": 0.3333}'}) == 0.3333


def test_too_few_runs_give_no_verdict_however_the_numbers_look():
    rows = arms_of(16, [0.5, 0.0], [0.0], [0.0])
    out = ro.build_report(rows, [], now=START + timedelta(days=6), permutations=200, unblind=True)
    assert not out["primary"]["read"] and out["primary"]["p"] is None
    assert out["primary"]["verdict"].startswith("RUNNING")
    assert out["primary"]["stop"]["completed_per_arm"] == {REP: 16, VAL: 16, UNC: 16}
    assert out["arms"][REP]["moved_share"] == 0.5 and out["arms"][REP]["median_nats"] == 0.25


def test_a_clear_winner_at_250_per_arm_wins_both_tests():
    rows = arms_of(250, [0.0, 0.0, 0.0, 0.6], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1])
    out = ro.build_report(rows, [], now=START + timedelta(days=40), permutations=500)
    prim = out["primary"]
    assert prim["read"] and prim["stop"]["reason"].startswith("every arm")
    assert prim["replica_wins"] and all(p < 0.05 for p in prim["p"].values())


def test_beating_one_rule_arm_is_a_loss_and_the_clock_stops_the_test():
    rows = arms_of(30, [0.0, 0.3], [0.0, 0.3], [0.0])
    out = ro.build_report(rows, [], now=START + timedelta(days=181), permutations=500)
    prim = out["primary"]
    assert prim["read"] and "180 days" in prim["stop"]["reason"]
    assert prim["p"][UNC] < 0.05 and prim["p"][VAL] > 0.05
    assert prim["replica_wins"] is False and prim["verdict"].startswith("REPLICA LOSES")


def test_permutation_p_is_one_sided_and_needs_two_values_per_arm():
    assert ro.permutation_p_greater([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], permutations=200) > 0.9
    assert ro.permutation_p_greater([1.0], [0.0, 0.0]) is None


def test_fisher_one_sided_matches_hand_value():
    # 3 of 3 bad vs 0 of 3: P = 1 / C(6,3) = 0.05
    assert abs(ro.fisher_p_greater(3, 0, 0, 3) - 0.05) < 1e-12
    assert ro.fisher_p_greater(0, 3, 3, 0) == 1.0


def test_harm_look_pauses_an_arm_that_keeps_failing():
    rows = [run(i, REP, None, hours=i, turn_ok=False) for i in range(12)]
    rows += [run(100 + i, VAL if i % 2 else UNC, 0.1, hours=i) for i in range(40)]
    out = ro.build_report(rows, [], now=START + timedelta(days=31), permutations=100)
    assert out["harm"]["pause"] and out["harm"]["failed_turns"]["replica"] == "12/12"
    assert out["harm"]["next_scheduled_look"] == START + timedelta(days=60)
    assert out["needs_juniper"] == ["harm"]
    calm = ro.build_report(arms_of(20, [0.1], [0.1], [0.1]), [], now=START + timedelta(days=5), permutations=100)
    assert not calm["harm"]["pause"] and calm["needs_juniper"] == []


def test_fallbacks_count_by_reason_and_flag_unfunded_over_30_percent():
    rows = [run(1, REP, 0.1, hours=1, plan="ok"), run(2, REP, 0.0, hours=2, plan="unfunded"),
            run(3, REP, 0.0, hours=3, plan="unfunded"), run(4, REP, 0.0, hours=4, plan=None),
            run(5, REP, 0.0, hours=5, plan="invalid:not_a_candidate:3"), run(6, UNC, 0.0, hours=6)]
    out = ro.build_report(rows, [], now=START + timedelta(days=2), permutations=100)
    f = out["fallbacks"]
    assert f["by_status"] == {"invalid:not_a_candidate:3": 1, "no_plan_row": 1, "ok": 1, "unfunded": 2}
    assert f["fell_back"] == 4 and f["unfunded_share"] == 0.4 and f["alarm"]
    assert "unfunded" in out["needs_juniper"]
    assert out["as_treated"]["plan_used"]["completed"] == 1 and out["as_treated"]["fell_back"]["completed"] == 4
    old = ro.fallback_report(ro.three_arm_rows(rows), now=START + timedelta(days=20), days=14)
    assert old["replica_runs"] == 0 and old["verdict"].startswith("NO DATA")


def test_render_speaks_plainly_and_names_unknown_answers():
    out = ro.build_report(arms_of(3, [0.1], [0.0], [0.0]), None, now=START + timedelta(days=1), permutations=50)
    text = ro.render(out)
    assert "replica plan" in text and "RUNNING" in text and "Answers to Prime: UNKNOWN" in text
    assert ro.render(ro.build_report([], [], now=START, permutations=10)).count("has not started") == 1


def test_before_the_stop_the_outcome_numbers_are_hidden():
    out = ro.build_report(arms_of(16, [0.5, 0.0], [0.0], [0.0]), [], now=START + timedelta(days=6), permutations=50)
    assert out["blinded"] and "mean_nats" not in out["arms"][REP] and "moved_share" not in out["arms"][REP]
    assert "mean_nats" not in out["as_treated"]["plan_used"]
    text = ro.render(out)
    assert "nats," not in text and "As treated" not in text and "hidden until the stop" in text


def test_runs_drawn_after_the_stop_never_change_the_verdict():
    rows = arms_of(250, [0.0, 0.0, 0.0, 0.6], [0.0] * 7 + [0.1], [0.0] * 7 + [0.1])
    first = ro.build_report(rows, [], now=START + timedelta(days=40), permutations=300)
    last_draw = rows[-1]["decided_at"]
    later = rows + [run(9000 + i, REP, 0.0, hours=(last_draw - START).total_seconds() / 3600 + 1 + i)
                    for i in range(400)]
    second = ro.build_report(later, [], now=START + timedelta(days=60), permutations=300)
    assert second["primary"]["stop"]["stop_at"] == first["primary"]["stop"]["stop_at"] == last_draw
    assert second["primary"]["p"] == first["primary"]["p"] and second["primary"]["replica_wins"]
    assert second["primary"]["stop"]["completed_per_arm"][REP] == 250


def test_the_harm_look_only_moves_at_a_scheduled_look_and_lost_runs_count_as_failed():
    early = [run(i, REP, None, hours=i, turn_ok=None) for i in range(12)]
    early += [run(100 + i, VAL if i % 2 else UNC, 0.1, hours=i) for i in range(40)]
    assert ro.harm_check(ro.three_arm_rows(early), now=START + timedelta(days=10))["verdict"] == "no look yet"
    look = ro.harm_check(ro.three_arm_rows(early), now=START + timedelta(days=31))
    assert look["pause"] and look["failed_turns"]["replica"] == "12/12"
    # Runs drawn after day 30 cannot move the day-30 look, however bad they are.
    late = early + [run(500 + i, UNC, None, hours=24 * 31 + i, turn_ok=False) for i in range(50)]
    again = ro.harm_check(ro.three_arm_rows(late), now=START + timedelta(days=45))
    assert again["failed_turns"] == look["failed_turns"] and again["look"] == START + timedelta(days=30)


def test_too_few_scored_runs_at_the_clock_stop_is_inconclusive_not_a_loss():
    rows = [run(1, REP, 0.2), run(2, VAL, 0.0, hours=1), run(3, VAL, 0.1, hours=2),
            run(4, UNC, 0.0, hours=3), run(5, UNC, 0.0, hours=4)]
    prim = ro.build_report(rows, [], now=START + timedelta(days=200), permutations=50)["primary"]
    assert prim["read"] and prim["replica_wins"] is None and prim["verdict"].startswith("INCONCLUSIVE")


def test_unreadable_answers_go_to_juniper_and_bad_propensity_text_is_not_a_crash():
    out = ro.build_report(arms_of(2, [0.1], [0.1], [0.1]), None, now=START + timedelta(days=1), permutations=10)
    assert out["needs_juniper"] == ["answers_unknown"]
    assert ro.replica_propensity({"arm_propensities": "{not json"}) == 0.0
    assert ro.replica_propensity({"arm_propensities": "[1, 2]"}) == 0.0
