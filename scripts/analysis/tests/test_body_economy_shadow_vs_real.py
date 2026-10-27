"""Patch C3a: the shadow-vs-real disagreement report's join (scripts/analysis/body_economy_shadow_vs_real.py)."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ANALYSIS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ANALYSIS))
sys.path.insert(0, str(_ANALYSIS.parents[1]))

import body_economy_shadow_vs_real as svr  # noqa: E402

T0 = datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc)
FAM = "durable.self_inquiry"


def dec(i, outcome, reason="net_positive", *, corr=None, producer="orion-durable-runs.admission", family=FAM,
        at=T0, hours=6, binding=None):
    return svr.Decision(f"d{i}", family, producer, at, outcome, reason, binding, corr or f"c{i}",
                        f"durable_run:r{i}", at + timedelta(hours=hours))


def act(i, opened, *, family=FAM, by="corr"):
    return svr.Act(family, opened, f"c{i}" if by == "corr" else None, f"durable_run:r{i}" if by == "ref" else None)


def test_the_four_cells_and_the_window():
    acts = [act(0, T0 - timedelta(days=1)),       # the producer was already running
            act(1, T0 + timedelta(hours=1)), act(3, T0 + timedelta(minutes=5), by="ref"),
            act(4, T0 + timedelta(hours=7))]              # opened after the 6 h window: not "ran"
    rows = svr.classify([dec(1, "funded"), dec(2, "funded"), dec(3, "deferred", "capacity_reserved", binding="lane:agent"),
                         dec(4, "deferred", "price", binding="lane:agent")], acts)
    assert [r["cell"] for r in rows] == ["funded_ran", "funded_not_run", "deferred_ran", "deferred_not_run"]
    assert rows[0]["wait_sec"] == 3600.0 and rows[2]["ran"] is True       # joined by object_ref
    s = svr.summarize(rows)
    x = s["families"][FAM]
    assert (x["funded_ran"], x["funded_not_run"], x["deferred_ran"], x["deferred_not_run"]) == (1, 1, 1, 1)
    assert x["agreement"] == 0.5 and x["deferred_ran_by_reason"] == {"capacity_reserved": 1}
    assert x["top_binding"] == [("lane:agent", 2)] and s["total"]["in_table"] == 4


def test_no_market_probes_no_producer_and_pre_producer_rows_stay_out_of_the_table():
    acts = [act(9, T0 + timedelta(days=1))]           # the family's producer starts a day after T0
    rows = svr.classify([
        dec(1, "deferred", "physiology_unavailable", at=T0 + timedelta(days=2)),
        dec(2, "funded", producer="deploy-verification", at=T0 + timedelta(days=2)),
        dec(3, "funded", family="concept_induction", at=T0 + timedelta(days=2)),
        dec(4, "funded", at=T0),                       # before the producer existed
    ], acts)
    assert [r["excluded"] for r in rows] == ["no_market", "synthetic_probe", "no_act_producer", "before_act_producer"]
    assert all(r["cell"] is None for r in rows)
    assert rows[2]["ran"] is None and rows[3]["ran"] is None   # unknown, never "not run"
    t = svr.summarize(rows)["total"]
    assert (t["in_table"], t["no_market"], t["synthetic_probe"], t["no_act_producer"], t["before_act_producer"]) == (0, 1, 1, 1, 1)
    assert t["agreement"] is None


def test_an_act_of_another_family_or_before_the_bid_does_not_count():
    rows = svr.classify([dec(1, "funded")], [act(0, T0 - timedelta(days=1)),
                                             act(1, T0 + timedelta(hours=1), family="durable.curiosity_investigate"),
                                             act(1, T0 - timedelta(hours=1)), act(5, T0)])
    assert rows[0]["cell"] == "funded_not_run"


def test_window_end_prefers_latest_then_the_family_window():
    assert svr.window_end(T0, T0 + timedelta(hours=1), None, 21600) == T0 + timedelta(hours=1)
    assert svr.window_end(T0, None, T0 - timedelta(hours=1), 3600) == T0       # never before the decision
    assert svr.window_end(T0, None, None, 7200) == T0 + timedelta(hours=2)


def test_parse_vetoes_maps_hub_log_lines_to_families():
    lines = ["2026-10-25T12:00:00+0000 athena orion-athena-hub[1]: INFO curiosity_investigation_blocked reason=daily_cap",
             "2026-10-25T12:01:00+0000 athena orion-athena-hub[1]: curiosity_self_inquiry_blocked reason=cooldown",
             "2026-10-25T12:02:00+0000 athena orion-athena-hub[1]: curiosity_self_inquiry_blocked reason=cooldown",
             "unrelated line"]
    assert svr.parse_vetoes(lines) == {"durable.curiosity_investigate": {"daily_cap": 1},
                                       "durable.self_inquiry": {"cooldown": 2}}
