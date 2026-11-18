"""prediction.experiment (Family C, 2026-11-16): a tier-2, lane-less family whose bids bind.

Pins: the family is tier 2 with no lanes; its producer is a binding producer (so Prime's charge can apply
later); the real market funds a decision want whose hunger runs from the previous decision, and defers (sated)
one opened at the bid itself -- which is why the runner sends ``opened_at`` = the previous decision.
Also: the run-cap override's pure judge (orion/physiology/run_cap_override.py).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from orion.physiology.binding import BINDING_PRODUCERS
from orion.physiology.choice import ReservationBook, build_context, decide
from orion.physiology.families import load_families
from orion.physiology.run_cap_override import OBEYED, judge, load_bounds, parse_bounds
from orion.physiology.tests.test_family_lanes import _live_body
from orion.schemas.physiology_run_cap import RunCapOverrideV1, run_cap_key
from orion.schemas.physiology_want import WantV1, WantWindowV1

REPO = Path(__file__).resolve().parents[3]
FAMILIES = REPO / "config" / "physiology" / "families.yaml"
NOW = datetime(2026, 11, 13, 12, 0, tzinfo=timezone.utc)


def test_the_family_is_tier_2_lane_less_and_binding():
    fams, err = load_families(FAMILIES)
    assert err is None
    cfg = fams.families["prediction.experiment"]
    assert cfg.tier == 2 and list(cfg.lanes) == [] and cfg.deferrable_window_s == 2400
    assert "orion-prediction.experiment.binding" in BINDING_PRODUCERS


def _decide(opened_at):
    fams, _ = load_families(FAMILIES)
    st = _live_body()
    ctx = build_context(fams, [], [], st, NOW, weights={})
    w = WantV1(want_id="exp:0", family="prediction.experiment", producer="orion-prediction.experiment.binding",
               verb="experiment", tier=2, object_ref="run_cap:aitown.session#0", opened_at=opened_at,
               window=WantWindowV1(latest=NOW + timedelta(minutes=40)))
    return decide(w, st, ctx, ReservationBook(), NOW)


def test_the_market_funds_a_decision_hungry_since_the_last_one():
    d = _decide(NOW - timedelta(minutes=40))
    assert d.outcome == "funded", d.why


def test_a_want_opened_at_the_bid_is_sated_which_is_why_the_runner_sends_the_last_decision():
    d = _decide(NOW)
    assert d.outcome == "deferred" and d.reason == "sated", d.why


# --- the override's pure judge ---------------------------------------------------------------------

BOUNDS = load_bounds(REPO / "config" / "physiology" / "run_cap_overrides.yaml")


def _raw(**over):
    base = dict(family="aitown.session", knob="hold_starts", value=True, experiment_id="e", block_index=0,
                set_at=NOW, expires_at=NOW + timedelta(seconds=90))
    base.update(over)
    return RunCapOverrideV1(**base).model_dump_json()


@pytest.mark.parametrize("raw,ttl,now,reason", [
    (_raw(), 90, NOW + timedelta(seconds=30), OBEYED),
    (None, None, NOW, "absent"),
    (_raw(), -1, NOW, "no_ttl"),
    (_raw(value=False), 90, NOW, "not_allowed"),
    (_raw(knob="max_open_sessions"), 90, NOW, "knob_mismatch"),
    (_raw(family="hub.outreach"), 90, NOW, "family_mismatch"),
    (_raw(), 90, NOW + timedelta(seconds=2701), "stale_set_at"),
    (_raw(set_at=NOW + timedelta(minutes=5)), 90, NOW, "future_set_at"),
    ("{", 90, NOW, "malformed"),
])
def test_judge(raw, ttl, now, reason):
    v = judge(raw, ttl, now, "aitown.session", BOUNDS["aitown.session"])
    assert v.reason == reason and v.active == (reason == OBEYED)
    if v.active:
        assert v.retry_after_sec == 90


def test_an_unlisted_family_is_never_obeyed():
    v = judge(_raw(), 90, NOW, "aitown.session", None)
    assert not v.active and v.reason == "not_configured"


def test_a_bad_bounds_file_is_refused_and_a_missing_one_holds_nothing(tmp_path):
    with pytest.raises(ValueError):
        parse_bounds({"families": {"x": {"knob": "k", "allowed": [], "max_hold_s": 10}}})
    assert load_bounds(tmp_path / "nope.yaml") == {}


def test_the_key_name():
    assert run_cap_key("aitown.session") == "orion:physiology:run_cap:aitown.session"
