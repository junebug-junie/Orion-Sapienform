"""Organ 3 Appetite (Patch C1): hunger grows with time unfunded, resets on close, ranks within family."""

from __future__ import annotations

import math
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pytest

from orion.physiology.appetite import (
    ActRecord,
    build_family_appetite,
    estimate_cost,
    hunger,
    intensity,
    lane_hunger_seed,
    mid_rank,
    rank_value,
)
from orion.physiology.families import FamilyConfig, load_families, parse_families

NOW = datetime(2026, 10, 27, 6, 0, tzinfo=timezone.utc)
CFG = FamilyConfig(family="thought.painting", tier=2, deferrable_window_s=21600, lanes=("diffusion",))


def act(i, *, hours_ago, outcome=None, source="producer", family="thought.painting", cost=None, dur_s=60):
    closed = NOW - timedelta(hours=hours_ago)
    return ActRecord(act_id=f"a{i}", family=family, opened_at=closed - timedelta(seconds=dur_s), closed_at=closed,
                     outcome_value=outcome, outcome_source=source, cost=cost or {})


def test_intensity_rests_at_zero_and_rises_toward_one():
    assert intensity(0, 3600) == 0.0 and intensity(None, 3600) == 0.0
    assert intensity(3600, 3600) == pytest.approx(1 - math.exp(-1))
    assert intensity(10 * 3600, 3600) > 0.99


def test_lifecycle_close_resets_hunger_then_it_regrows():
    acts = [act(i, hours_ago=h) for i, h in enumerate([10, 8, 6, 4, 2, 1])]
    app = build_family_appetite(CFG, acts, NOW)
    assert app.tau_basis.startswith("rhythm:") and app.last_close_at == NOW - timedelta(hours=1)
    before = hunger(app, NOW).intensity
    # A new act closes now: the drive discharges to 0.
    app2 = build_family_appetite(CFG, acts + [act(9, hours_ago=0)], NOW)
    assert hunger(app2, NOW).intensity == 0.0 < before
    later = NOW + timedelta(hours=3)
    assert hunger(app2, later).intensity > before


def test_lane_seed_is_a_floor_from_b1_waiting_self_work():
    app = build_family_appetite(CFG, [act(1, hours_ago=0.1)], NOW)
    seed, lane = lane_hunger_seed({"diffusion": {"deferrable_queued": 1, "deferrable_waiting_sec": 7200.0},
                                   "agent": {"deferrable_queued": 0, "deferrable_waiting_sec": 99999.0}},
                                  ["diffusion", "agent"])
    assert (seed, lane) == (7200.0, "diffusion")
    h = hunger(app, NOW, seed_sec=seed, seed_lane=lane)
    assert h.basis == "lane_seed:diffusion" and h.unfunded_sec == 7200.0
    # Nothing queued, or unread: no seed (unknown is not starving).
    assert lane_hunger_seed({"diffusion": {"deferrable_no_reading": "stale"}}, ["diffusion"]) == (None, None)


def test_a_family_that_never_closed_counts_from_the_want_and_unknown_is_zero():
    app = build_family_appetite(CFG, [], NOW)
    assert app.last_close_at is None and app.tau_basis == "deferrable_window"
    assert hunger(app, NOW).intensity == 0.0 and hunger(app, NOW).basis == "unknown"
    assert hunger(app, NOW, opened_at=NOW - timedelta(hours=6)).basis == "since_opened"


def test_ungraded_family_ranks_on_intensity_alone_and_is_labelled():
    acts = [act(i, hours_ago=2 * i + 1, outcome=0.0) for i in range(10)]
    app = build_family_appetite(CFG, acts, NOW)
    assert not app.graded and app.n_graded == 10
    rank, basis, out = rank_value(app, 0.99, outcome_value=0.0)
    assert basis == "ungraded" and out == 1.0 and rank > 0.9


def test_graded_family_needs_30_graded_closes_and_expired_is_not_a_grade():
    acts = [act(i, hours_ago=i + 1, outcome=1.0 if i % 2 else 0.0) for i in range(29)]
    acts += [act(100 + i, hours_ago=40 + i, outcome=0.0, source="expired") for i in range(5)]
    assert not build_family_appetite(CFG, acts, NOW).graded
    app = build_family_appetite(CFG, acts + [act(99, hours_ago=0.5, outcome=1.0)], NOW)
    assert app.graded and app.n_graded == 30 and app.mean_outcome == pytest.approx(15 / 30)
    hi, basis, _ = rank_value(app, 0.9, outcome_value=1.0)
    lo, _, _ = rank_value(app, 0.9, outcome_value=0.0)
    assert basis == "graded" and hi > lo


def test_mid_rank_counts_ties_half():
    assert mid_rank(0.5, [0.1, 0.5, 0.9]) == pytest.approx(0.5)
    assert mid_rank(1.0, [0.1, 0.5]) == 1.0


def test_cost_estimate_want_then_fixed_then_p90_then_unpriced_never_zero():
    acts = [act(i, hours_ago=i + 1, cost={"seconds": 50.0 + i, "gpu_slot_s_by_lane": {"diffusion": 40.0 + i}})
            for i in range(10)]
    app = build_family_appetite(CFG, acts, NOW)
    c = estimate_cost(None, CFG, app)
    assert c.basis["seconds"] == "p90" and c.scalars["seconds"] == pytest.approx(58.1)
    assert c.one_of_slot_s == pytest.approx(48.1) and "juniper_msgs" in c.unpriced
    c2 = estimate_cost({"seconds": 5.0}, CFG, app)
    assert c2.scalars["seconds"] == 5.0 and c2.basis["seconds"] == "want"
    out = FamilyConfig(family="hub.outreach", tier=2, deferrable_window_s=3600, lanes=("chat",), placement="fixed",
                       fixed_cost={"juniper_msgs": 1})
    c3 = estimate_cost(None, out, build_family_appetite(out, [], NOW))
    assert c3.scalars["juniper_msgs"] == 1.0 and c3.basis["juniper_msgs"] == "fixed"
    assert "gpu_slot_s_by_lane" in c3.unpriced and "juniper_msgs" not in c3.unpriced


def test_live_families_config_parses_and_keeps_the_tier_decision():
    fams, err = load_families(Path(__file__).resolve().parents[3] / "config/physiology/families.yaml")
    assert err is None
    assert fams.get("hub.chat_reply").tier == 0 and fams.get("vision.council").tier == 0
    assert not fams.get("hub.chat_reply").bids
    assert fams.get("actions.journal").tier == 1 and fams.get("topic_foundry.run").tier == 2
    for f, c in fams.families.items():
        if c.bids:
            assert c.deferrable_window_s > 0, f


def test_families_config_rejects_lane_prefix_and_bad_tier():
    with pytest.raises(ValueError):
        parse_families({"version": 1, "families": {"x": {"tier": 3}}})
    with pytest.raises(ValueError):
        parse_families({"version": 1, "families": {"x": {"tier": 2, "deferrable_window_s": 10, "lanes": ["lane:agent"]}}})
    with pytest.raises(ValueError):
        parse_families({"version": 1, "families": {"x": {"tier": 2}}})   # a bidder needs a hunger clock
