"""Attention patch 3b: a want whose cost states an EMPTY gpu_slot_s_by_lane holds no GPU slot. It is
placed on no lane and never capacity-deferred on one. Reading's after-the-read look is the first user:
since patch 3a declared reading's lanes, each look was charged a whole read and 37 of 40 deferred on
lane:agent / lane:agent-gpu2 (live, 2026-11-10 08:42Z .. 11-11 01:14Z)."""
from __future__ import annotations

from datetime import timedelta

from orion.physiology.appetite import estimate_cost
from orion.physiology.choice import ReservationBook, decide
from orion.physiology.event_wants import event_want
from orion.physiology.tests.test_choice import FAMS, NOW, body, ctx_for, history, lane
from orion.schemas.grammar import ActCostV2

READ = "hub.world_pulse_read"
READS = history(READ, slot={"agent-gpu2": 2500.0})          # reads cost ~0.7 of a lane hour each
FULL = body(lane("agent", level=3600.0), lane("agent-gpu2", slots=2, level=7000.0))


def look(cost):
    return event_want(source="reading", key="seed-1", family=READ, producer="orion-hub.reading_events", verb="look",
                      tier=2, event_ref="world_pulse_read_seed:seed-1", salience=0.9,
                      opened_at=NOW - timedelta(minutes=5), cost=cost)


def test_without_a_stated_cost_a_look_is_charged_a_whole_read_and_deferred_on_a_full_lane():
    d = decide(look(None), FULL, ctx_for(READS, FULL), ReservationBook(), NOW)
    assert d.outcome == "deferred" and d.reason == "capacity_reserved" and d.binding_constraint.startswith("lane:agent")


def test_a_look_that_holds_no_slot_is_not_placed_on_a_lane_and_is_funded_on_its_value():
    d = decide(look(ActCostV2(gpu_slot_s_by_lane={})), FULL, ctx_for(READS, FULL), ReservationBook(), NOW)
    assert d.outcome == "funded" and d.reason == "net_positive" and d.net > 0
    assert d.lane is None and not any(k.startswith("lane:") for k in d.cost_frac) and d.reserved == {}
    assert not any("lane_cost_unmeasured" in w or "no_slot_history" in w for w in d.warnings)


def test_estimate_cost_reads_an_empty_map_as_stated_not_unknown():
    cfg = FAMS.get(READ)
    from orion.physiology.choice import build_context
    app = build_context(FAMS, READS, [], FULL, NOW).appetite[READ]
    c = estimate_cost({"gpu_slot_s_by_lane": {}}, cfg, app)
    assert c.lane_slot_s == {} and c.one_of_slot_s is None and c.basis["gpu_slot_s_by_lane"] == "want"
    assert "gpu_slot_s_by_lane" not in c.unpriced
    unset = estimate_cost({"gpu_slot_s_by_lane": None}, cfg, app)    # None still means "estimate it"
    assert unset.one_of_slot_s is not None and unset.basis["gpu_slot_s_by_lane"] == "p90"


def test_the_hub_reading_producer_states_the_no_slot_cost():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "services/orion-hub/scripts"))
    import reading_event_wants as rew
    assert rew.LOOK_COST.gpu_slot_s_by_lane == {}
    # It survives the wire (Hub posts model_dump_json; the market validates WantV1).
    from orion.schemas.physiology_want import WantV1
    w = WantV1.model_validate_json(look(rew.LOOK_COST).model_dump_json())
    assert w.cost is not None and w.cost.gpu_slot_s_by_lane == {}
