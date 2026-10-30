"""Patch D: the binding replay's pure parts (scripts/analysis/body_economy_binding_replay.py)."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ANALYSIS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ANALYSIS))
sys.path.insert(0, str(_ANALYSIS.parents[1]))

import body_economy_binding_replay as rep  # noqa: E402

T0 = datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc)
FAM = frozenset({"durable.curiosity_investigate"})


def _state(t, agent_level):
    return {"tick_id": f"t{t.isoformat()}", "generated_at": t.isoformat(), "warnings": [],
            "constraints": [{"constraint": "lane:agent", "level": agent_level, "budget": 3600.0},
                            {"constraint": "heat", "level": 30.0}]}


def _act(actor, opened, closed, slot_s):
    return {"actor": actor, "opened_at": opened.isoformat(), "closed_at": closed.isoformat(),
            "cost_json": {"gpu_slot_s_by_lane": {"agent": slot_s}}}


def test_own_use_is_removed_pro_rata_from_the_trailing_window_only():
    # A 2 h curiosity act holding the agent lane fully (7200 slot-s): 3600 slot-s fall in any trailing
    # hour inside it, half of that in the hour that straddles its start.
    acts = [_act("durable.curiosity_investigate", T0, T0 + timedelta(hours=2), 7200.0),
            _act("durable.self_inquiry", T0, T0 + timedelta(hours=2), 7200.0)]       # a sibling: kept
    states = [_state(T0 + timedelta(minutes=30), 3600.0), _state(T0 + timedelta(hours=1, minutes=30), 3600.0),
              _state(T0 + timedelta(hours=4), 900.0)]
    out = rep.without_own_use(states, acts, FAM)
    levels = [next(c["level"] for c in s["constraints"] if c["constraint"] == "lane:agent") for s in out]
    assert levels == [1800.0, 0.0, 900.0]
    assert out[0]["constraints"][1] == {"constraint": "heat", "level": 30.0}             # non-lanes untouched
    assert states[0]["constraints"][0]["level"] == 3600.0                                # input not mutated


def test_summary_counts_holds_and_extra_delay():
    rows = [
        {"verdict": "queued_at_pool", "n_bids": 1, "held_sec": 0.0, "extra_delay_sec": 0.0,
         "bids": [{"outcome": "deferred", "reason": "capacity_reserved", "binding": "lane:agent"}]},
        {"verdict": "window_end", "n_bids": 3, "held_sec": 21600.0, "extra_delay_sec": 20000.0,
         "bids": [{"outcome": "deferred", "reason": "price", "binding": "lane:agent"}] * 3},
        {"verdict": "funded", "n_bids": 1, "held_sec": 0.0, "extra_delay_sec": None,
         "bids": [{"outcome": "funded", "reason": "net_positive", "binding": None}]},
    ]
    s = rep.summarize(rows)
    assert s["runs"] == 3 and s["queued_at_pool"] == 1 and s["admitted_at_window_end"] == 1
    assert s["funded_first_bid"] == 1 and s["held_runs"] == 1 and s["runs_delayed_past_real_open"] == 1
    assert s["deferral_reasons_all_bids"] == {"capacity_reserved": 1, "price": 3}
