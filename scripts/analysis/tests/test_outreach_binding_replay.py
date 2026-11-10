"""Patch D for hub.outreach: the outreach binding replay's pure parts (scripts/analysis/outreach_binding_replay.py)."""
from __future__ import annotations

import sys
from pathlib import Path

_ANALYSIS = Path(__file__).resolve().parents[1]
_ROOT = _ANALYSIS.parents[1]
sys.path.insert(0, str(_ANALYSIS))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "services" / "orion-hub" / "evals"))

import outreach_binding_replay as rep  # noqa: E402


def test_fixture_eval_passes():
    import run_outreach_binding_replay as ev

    assert ev.fixture() == []


def test_every_binding_door_is_cap_counted_now():
    """One attention budget (2026-11-10): no door is cap-exempt any more; the Hub's count reads the one
    shared exempt list (the strain report only), so every replayed attempt is cap-counted."""
    text = (_ROOT / "services/orion-hub/scripts/endogenous_outreach_decisions.py").read_text()
    assert "CAP_EXEMPT_SOURCES" in text
    from orion.physiology.attention_pacing import CAP_EXEMPT_SOURCES

    assert not set(rep.DOOR_SOURCES) & set(CAP_EXEMPT_SOURCES)
    assert rep.door_of(None) == rep.TICK and rep.door_of("curiosity_outreach") is None
    assert rep.door_of("strain_report") is None
    got = rep.attempts_from_rows([{"decision_id": "r", "decided_at": "2026-11-10T12:00:00+00:00", "reason": "sent",
                                   "forced": False, "source": "reading_finding_outreach"}], {})
    assert got and got[0]["cap_counted"] is True


def test_summary_counts_holds_per_door():
    rows = [
        {"door": "endogenous_tick", "real_reason": "sent", "verdict": "funded", "n_bids": 1, "held_sec": 0.0,
         "bids": [{"outcome": "funded", "reason": "net_positive"}]},
        {"door": "endogenous_tick", "real_reason": "sent", "verdict": "window_end", "n_bids": 4, "held_sec": 3600.0,
         "bids": [{"outcome": "deferred", "reason": "price"}] * 3 + [{"outcome": None, "reason": None}]},
        {"door": "office_reach_out", "real_reason": "orion_passed", "verdict": "funded", "n_bids": 2, "held_sec": 600.0,
         "bids": [{"outcome": "deferred", "reason": "sated"}, {"outcome": "funded", "reason": "net_positive"}]},
    ]
    s = rep.summarize(rows)
    assert s["all"]["attempts"] == 3 and s["all"]["let_through_at_first_bid"] == 1
    assert s["all"]["held_then_admitted"] == 2 and s["all"]["held_sends"] == 1
    assert s["endogenous_tick"]["held_to_window_end"] == 1 and s["all"]["held_over_window"] == 0
    assert s["all"]["deferral_reasons_all_bids"] == {"price": 3, "sated": 1}
