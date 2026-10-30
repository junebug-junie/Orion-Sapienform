"""Orion Prime step 3: Prime's regulate signal as a bounded, temporary charge in Choice.

The charge (slow) or credit (quicken) is magnitude x family_weight, magnitude <= 0.5; a want ranked above
the middle of its family's history still beats it; the exploration floor still applies; the decision's
``why`` names the charge either way; what it did to a bid is classified the same for both arms.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from pydantic import ValidationError

from orion.physiology import prime_charge as pc
from orion.physiology.appetite import ActRecord
from orion.physiology.choice import ReservationBook, decide
from orion.physiology.families import parse_families
from orion.physiology.tests.test_choice import FAMS, NOW, STARVED, body, ctx_for, history, lane, seeded_book, want
from orion.physiology.weights import Evidence, FamilyWeight

C = "durable.curiosity_investigate"


def spread_history(last_h=10.0):
    """Curiosity closes at uneven gaps, so its value history spreads and a mid-hunger want ranks mid-way."""
    out, t = [], NOW - timedelta(hours=last_h)
    for i, g in enumerate([0.2, 0.5, 1.0, 2.0, 3.0, 0.3, 0.7, 1.5, 4.0, 0.4, 2.5, 0.8]):
        out.append(ActRecord(act_id=f"c{i}", family=C, opened_at=t - timedelta(seconds=60), closed_at=t,
                             outcome_value=1.0, outcome_source="producer",
                             cost={"seconds": 1000.0, "gpu_slot_s_by_lane": {"agent": 1800.0}}))
        t -= timedelta(hours=g)
    return out


ACTS = spread_history()


def sig(direction="slow", m=0.5, arm="treated", family=C, **kw):
    return pc.PrimeRegulationV1(signal_id="s1", act_id="act_x", family=family, direction=direction, arm=arm,
                                magnitude=m, why="curiosity ran five times this morning",
                                opened_at=kw.get("opened_at", NOW - timedelta(hours=1)),
                                expires_at=kw.get("expires_at", NOW + timedelta(hours=2)))


def cwant(i=0, **kw):
    return want(C, verb="infer", i=i, **kw)


def st():
    return body(lane("agent"), lane("agent-gpu2", slots=2))


def test_only_curiosity_investigate_declares_receptors_and_the_parser_refuses_bad_ones():
    assert FAMS.get(C).receptors == ("slow", "quicken")
    assert [f for f, c in FAMS.families.items() if c.receptors] == [C]
    base = {"version": 1, "families": {"x": {"tier": 2, "deferrable_window_s": 60, "receptors": ["calm"]}}}
    with pytest.raises(ValueError, match="receptors"):
        parse_families(base)
    with pytest.raises(ValueError, match="Tier 0"):
        parse_families({"version": 1, "families": {"x": {"tier": 0, "receptors": ["slow"]}}})


def test_the_charge_is_bounded_by_half_the_family_weight_and_signed_by_direction():
    assert pc.amount(sig("slow", 0.5), 1.08) == pytest.approx(0.54)
    assert pc.amount(sig("quicken", 0.3), 1.0) == pytest.approx(-0.3)
    assert pc.amount(sig("slow"), float("nan")) == 0.0
    with pytest.raises(ValidationError):
        sig("slow", 0.9)          # a charge above the bound cannot even be written
    with pytest.raises(ValidationError):
        sig("faster")


@pytest.mark.parametrize("intensity", [0.25, 0.4])
def test_a_slow_charge_holds_a_low_value_want_and_names_itself_as_the_binding_constraint(intensity):
    plain = decide(cwant(intensity=intensity), st(), ctx_for(ACTS, st()), ReservationBook(), NOW)
    d = decide(cwant(intensity=intensity), st(), ctx_for(ACTS, st()), ReservationBook(), NOW, prime=sig())
    assert plain.outcome == "funded" and plain.net > 0
    assert d.outcome == "deferred" and d.reason == "price" and d.binding_constraint == "prime:slow"
    # No price to decay: the default re-bid re-tests the want's growing hunger.
    assert d.retry_after_sec is None
    assert "did not beat Prime's slow charge 0.50" in d.why and "curiosity ran five times" in d.why
    assert pc.effect_of("slow", net_with=d.net, net_without=d.net + 0.5, floor_funded=False) == "held"


def test_a_hungry_valuable_want_outbids_the_largest_slow_charge():
    d = decide(cwant(), st(), ctx_for(ACTS, st()), ReservationBook(), NOW, prime=sig())
    assert d.outcome == "funded" and d.net > 0
    assert "value 1.00 - lane charge 0.00 beat Prime's slow charge 0.50" in d.why
    assert pc.effect_of("slow", net_with=d.net, net_without=d.net + 0.5, floor_funded=False) == "outbid"


@pytest.mark.parametrize("weight", [0.2, 0.6, 1.0, 1.5])
def test_never_a_veto_any_want_ranked_above_the_middle_beats_a_max_slow_charge(weight):
    w = {C: FamilyWeight(C, 2, weight, weight, "evidence", Evidence(1.0, 40, None, 0))}
    d = decide(cwant(), st(), ctx_for(ACTS, st(), weights=w), ReservationBook(), NOW, prime=sig(m=pc.MAX_MAGNITUDE))
    assert d.value_rank > 0.5 and d.outcome == "funded"


def test_a_quicken_credit_releases_a_want_the_market_held_for_no_hunger():
    fresh = history(C, slot={"agent": 1800.0}, last_h=0.0)
    plain = decide(cwant(), st(), ctx_for(fresh, st()), ReservationBook(), NOW)
    d = decide(cwant(), st(), ctx_for(fresh, st()), ReservationBook(), NOW, prime=sig("quicken"))
    assert plain.reason == "sated" and plain.net == 0.0
    assert d.outcome == "funded" and d.net == pytest.approx(0.5)
    assert "Prime's quicken credit 0.50" in d.why
    assert pc.effect_of("quicken", net_with=d.net, net_without=0.0, floor_funded=False) == "released"


def test_a_slow_charge_on_a_want_that_would_wait_anyway_changes_nothing():
    fresh = history(C, slot={"agent": 1800.0}, last_h=0.0)
    d = decide(cwant(), st(), ctx_for(fresh, st()), ReservationBook(), NOW, prime=sig())
    assert d.outcome == "deferred" and d.binding_constraint == "prime:slow"
    assert pc.effect_of("slow", net_with=d.net, net_without=d.net + 0.5, floor_funded=False) == "unchanged"


def test_the_exploration_floor_still_funds_a_starved_family_through_a_slow_charge():
    starved = {C: 1, "durable.self_inquiry": 99}
    d = decide(cwant(intensity=0.25), st(), ctx_for(ACTS, st()), seeded_book(starved), NOW, prime=sig())
    assert d.outcome == "funded" and d.reason == "exploration_floor" and d.net <= 0
    assert pc.effect_of("slow", net_with=d.net, net_without=d.net + 0.5, floor_funded=True) == "floor"
    assert STARVED  # the shared fixture still exists (other family starved there)


def test_a_slow_charge_on_a_full_lane_turns_a_worth_it_capacity_deferral_into_a_net_le_zero_one():
    full = body(lane("agent", level=3600.0), lane("agent-gpu2", slots=2, level=7200.0))
    plain = decide(cwant(intensity=0.25), full, ctx_for(ACTS, full), ReservationBook(), NOW)
    d = decide(cwant(intensity=0.25), full, ctx_for(ACTS, full), ReservationBook(), NOW, prime=sig())
    assert plain.reason == "capacity_reserved" and plain.net > 0          # queues at the pool (Patch D)
    assert d.reason == "capacity_reserved" and d.net <= 0                 # held: the charge says not now
    assert "Prime's slow charge" in d.why


def test_effects_merge_keeps_the_strongest_and_tally_counts_wants_not_bids():
    e = pc.merge(None, "w1", "held", NOW, "held why")
    e = pc.merge(e, "w1", "outbid", NOW + timedelta(minutes=10), "outbid why")
    assert (e.effect, e.last, e.bids, e.why) == ("held", "outbid", 2, "outbid why")
    o = pc.merge(None, "w2", "outbid", NOW, "x")
    assert pc.tally([e, o]) == {"wants": 2, "held": 1, "released": 0, "outbid": 1, "unchanged": 0, "floor": 0}
    assert pc.changed_any([e, o]) and not pc.changed_any([o])
    fields = {b"signal": pc.dumps(sig()).encode(), b"w:w2": pc.dumps(o).encode(), "w:w1": pc.dumps(e), "w:bad": "{"}
    assert sorted(x.want_id for x in pc.parse_effects(fields)) == ["w1", "w2"]
    assert pc.parse_signal(fields[b"signal"]).signal_id == "s1" and pc.parse_signal("nope") is None


def test_a_signal_is_active_only_inside_its_window():
    s = sig()
    assert pc.active(s, NOW) and not pc.active(s, NOW + timedelta(hours=2)) and not pc.active(s, NOW - timedelta(hours=2))
