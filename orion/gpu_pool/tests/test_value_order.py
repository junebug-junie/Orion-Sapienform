"""Attention patch 3b: the background queue served by value (spec 2026-11-07 section 1, acceptance 2).

The pure rule (orion/gpu_pool/value_order.py) and the scheduler that uses it."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from orion.gpu_pool.config import load_pool_config
from orion.gpu_pool.scheduler import Grant, LeaseView, Reordered, _order, queue_position, schedule
from orion.gpu_pool.tests.test_scheduler import cards, live
from orion.gpu_pool.value_order import ValueOrderConfig, pick, sequence

T0 = datetime(2026, 11, 11, 6, 0, tzinfo=timezone.utc)
CFG = load_pool_config()
ON = ValueOrderConfig()


@dataclass
class W:
    lid: str
    family: Optional[str]
    net: Optional[float]
    at: datetime


def w(lid, family, net, mins):
    return W(lid, family, net, T0 + timedelta(minutes=mins))


def first(items, now, **kw):
    return pick(items, now, net=lambda x: x.net, family=lambda x: x.family, arrived=lambda x: x.at, **kw).lid


def test_a_later_want_worth_the_margin_more_is_served_first():
    assert first([w("a", "reading", 0.5, 0), w("b", "curiosity", 0.8, 1)], T0 + timedelta(minutes=2)) == "b"


def test_a_gap_under_the_margin_keeps_arrival_order():
    assert first([w("a", "reading", 0.70, 0), w("b", "curiosity", 0.79, 1)], T0 + timedelta(minutes=2)) == "a"


def test_the_same_family_keeps_arrival_order_even_with_another_family_between():
    q = [w("a", "reading", 0.1, 0), w("b", "curiosity", 0.5, 1), w("c", "reading(stage2)", 0.9, 2)]
    assert first(q, T0 + timedelta(minutes=3)) == "b"


def test_upkeep_keeps_its_place_and_jumps_nobody():
    assert first([w("u", None, None, 0), w("b", "curiosity", 0.9, 1)], T0 + timedelta(minutes=2)) == "u"
    assert first([w("a", "reading", 0.2, 0), w("u", None, None, 1), w("b", "curiosity", 0.9, 2)],
                 T0 + timedelta(minutes=3)) == "a"


def test_aging_lets_a_hold_03_lower_win_after_3_hours_and_not_before():
    old, new = w("a", "reading", 0.5, 0), w("b", "curiosity", 0.8, 180)
    assert first([old, new], T0 + timedelta(minutes=181)) == "a"
    old, new = w("a", "reading", 0.5, 0), w("b", "curiosity", 0.8, 60)
    assert first([old, new], T0 + timedelta(minutes=61)) == "b"


def test_sequence_is_pick_repeated_and_keeps_every_item():
    q = [w("a", "reading", 0.2, 0), w("b", "curiosity", 0.9, 1), w("c", "self_inquiry", 0.6, 2)]
    out = sequence(q, T0 + timedelta(minutes=3), net=lambda x: x.net, family=lambda x: x.family,
                   arrived=lambda x: x.at)
    assert [x.lid for x in out] == ["b", "c", "a"]


# --- the scheduler ------------------------------------------------------------------------------

def hold(lid, net, family, age_s, priority="background", kind="hold"):
    return LeaseView(lease_id=lid, work_class="agent", priority=priority, status="queued", kind=kind,
                     created_at=T0 - timedelta(seconds=age_s), want_net=net, want_family=family)


def one_agent_seat():
    # one free agent slot and nothing else the agent class could borrow
    return live(**{"agent-gpu2": live()["agent-gpu2"].__class__("agent-gpu2", False, 0),
                   "agent-deep": live()["agent-deep"].__class__("agent-deep", False, 0)})


def granted(decisions):
    return [d.lease_id for d in decisions if isinstance(d, Grant)]


def test_value_off_keeps_todays_arrival_order_exactly():
    leases = [hold("early", 0.2, "reading", 600), hold("late", 0.9, "curiosity", 60)]
    off = schedule(CFG, one_agent_seat(), cards(), leases, T0)
    disabled = schedule(CFG, one_agent_seat(), cards(), leases, T0, value=ValueOrderConfig(enabled=False))
    assert granted(off) == granted(disabled) == ["early"]
    assert not [d for d in off + disabled if isinstance(d, Reordered)]
    assert [l.lease_id for l in _order(CFG, leases)] == [l.lease_id for l in _order(CFG, leases, T0, None)]


def test_value_on_serves_the_higher_want_first_and_says_so():
    leases = [hold("early", 0.2, "reading", 600), hold("late", 0.9, "curiosity", 60)]
    out = schedule(CFG, one_agent_seat(), cards(), leases, T0, value=ON)
    assert granted(out) == ["late"]
    (r,) = [d for d in out if isinstance(d, Reordered)]
    assert (r.lease_id, r.jumped_lease_id, r.net, r.jumped_net) == ("late", "early", 0.9, 0.2)
    assert (r.family, r.jumped_family, r.role) == ("curiosity", "reading", "agent")


def test_no_reorder_event_when_arrival_order_wins():
    leases = [hold("early", 0.85, "reading", 600), hold("late", 0.9, "curiosity", 60)]
    out = schedule(CFG, one_agent_seat(), cards(), leases, T0, value=ON)
    assert granted(out) == ["early"]
    assert not [d for d in out if isinstance(d, Reordered)]


def test_upkeep_lease_keeps_its_place_in_the_scheduler():
    leases = [hold("upkeep", None, None, 600), hold("late", 0.9, "curiosity", 60)]
    out = schedule(CFG, one_agent_seat(), cards(), leases, T0, value=ON)
    assert granted(out) == ["upkeep"]


def test_only_the_background_class_is_reordered():
    # A system-priority lease with a low net is still ahead of a background one with a high net, and
    # two system leases keep arrival order whatever their nets.
    sys_early = hold("sys_early", 0.1, "reading", 600, priority="system", kind="request")
    sys_late = hold("sys_late", 0.9, "curiosity", 300, priority="system", kind="request")
    bg = hold("bg", 0.95, "self_inquiry", 60)
    order = [l.lease_id for l in _order(CFG, [bg, sys_late, sys_early], T0, ON)]
    assert order == ["sys_early", "sys_late", "bg"]
    inter = hold("chat", None, None, 10, priority="interactive", kind="request")
    assert _order(CFG, [bg, inter], T0, ON)[0].lease_id == "chat"


def test_queue_position_follows_the_grant_order():
    leases = [hold("early", 0.2, "reading", 600), hold("late", 0.9, "curiosity", 60)]
    assert queue_position(CFG, leases, "late", T0, ON) == 1
    assert queue_position(CFG, leases, "early", T0, ON) == 2
    assert queue_position(CFG, leases, "late", T0, None) == 2
