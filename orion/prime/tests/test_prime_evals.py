"""The 2A readout's arithmetic, and the frame eval on its live fixture."""

from __future__ import annotations

import pytest

from orion.prime.evals import run_prime_frame_eval, run_prime_report as rpt


def test_fisher_exact_matches_known_values():
    # Lady tasting tea: [[3, 1], [1, 3]] -> two-sided p = 0.4857.
    assert rpt.fisher_exact_two_sided(3, 1, 1, 3) == pytest.approx(0.4857, abs=1e-4)
    assert rpt.fisher_exact_two_sided(10, 0, 0, 10) == pytest.approx(1.08e-5, rel=0.01)


def _ask(arm, value):
    return {"verb": "ask", "test_before": f"holdback:{arm}", "outcome_value": value}


def test_narrator_needs_enough_closed_per_arm_and_a_real_difference():
    few = [_ask("treated", 1.0)] * 30 + [_ask("control", 0.0)] * 5
    assert rpt.narrator(few)["verdict"] == "too_few"
    same = [_ask("treated", 1.0)] * 10 + [_ask("treated", 0.0)] * 10 + [_ask("control", 1.0)] * 5 + [_ask("control", 0.0)] * 5
    assert rpt.narrator(same)["verdict"] == "no_difference"
    cause = [_ask("treated", 1.0)] * 18 + [_ask("treated", 0.0)] * 2 + [_ask("control", 1.0)] * 3 + [_ask("control", 0.0)] * 9
    out = rpt.narrator(cause + [_ask("treated", None)])
    assert out["verdict"] == "cause_up" and out["treated"]["open"] == 1 and out["p"] < 0.05


def test_thoughts_and_cadence_buckets():
    acts = [{"verb": "infer", "outcome_value": 1.0}, {"verb": "infer", "outcome_value": 0.0, "outcome_note": "preempted: x"},
            {"verb": "infer", "outcome_value": 0.0, "outcome_note": "failed: no_json_object"}, {"verb": "infer"}]
    assert rpt.thoughts(acts) == {"opened": 4, "produced": 1, "preempted": 1, "failed": 1, "open": 1}
    bids = [{"outcome": "funded", "decision": {"prices_seen": {"heat": 0.0, "lane:chat": 0.0}}},
            {"outcome": "deferred", "decision": {"prices_seen": {"heat": 0.4}}},
            {"outcome": "funded", "decision": {"prices_seen": {"lane:chat": 0.2}}}]
    out = rpt.cadence(bids)
    assert out["unpriced"] == {"funded": 1, "bids": 1, "funded_share": 1.0}
    assert out["priced"]["bids"] == 2 and out["priced"]["funded_share"] == 0.5


def test_frame_eval_passes_on_the_live_fixture():
    results = run_prime_frame_eval.run()
    assert all(ok for _, ok, _ in results), [r for r in results if not r[1]]


def test_concerns_tally_reads_each_end_from_the_close():
    acts = [{"verb": "predict", "outcome_value": 1.0, "outcome_note": "settled: question act x"},
            {"verb": "predict", "outcome_value": 0.0, "outcome_note": "dropped unresolved: moved on"},
            {"verb": "predict", "outcome_value": 0.0, "outcome_note": "faded: not kept for a day"},
            {"verb": "predict", "outcome_value": 0.0, "outcome_note": "lapsed: no question"},
            {"verb": "predict", "outcome_value": 0.0, "outcome_source": "expired", "outcome_note": ""},
            {"verb": "predict"}, {"verb": "infer", "outcome_value": 1.0}]
    assert rpt.concerns(acts) == {"opened": 6, "open": 1, "settled": 1, "dropped": 1, "faded": 1, "lapsed": 1,
                                  "expired": 1, "other": 0}


def test_regulation_counts_changes_and_starts_per_signal_hour_per_arm():
    from datetime import datetime, timedelta, timezone

    t0 = datetime(2026, 10, 30, 6, 0, tzinfo=timezone.utc)
    fam = "durable.curiosity_investigate"

    def reg(arm, value, start_h, hours=2.0):
        return {"verb": "regulate", "test_before": f"holdback:{arm}", "outcome_value": value,
                "object_ref": f"physiology_family:{fam}", "horizon_sec": hours * 3600 + rpt.REGULATE_GRACE_SEC,
                "opened_at": t0 + timedelta(hours=start_h)}

    acts = [reg("treated", 1.0, 0), reg("treated", 0.0, 10), reg("control", 1.0, 20), reg("control", None, 30)]
    starts = [{"actor": fam, "opened_at": t0 + timedelta(hours=h)} for h in (0.5, 21, 21.5, 23)]
    out = rpt.regulation(acts, starts)
    assert out["treated"] | {} == {"signals": 2, "open": 0, "closed": 2, "changed": 1, "signal_hours": 4.0, "starts": 1,
                                   "changed_share": 0.5, "starts_per_hour": 0.25}
    assert out["control"]["starts"] == 2 and out["control"]["open"] == 1 and out["verdict"] == "too_few"
