"""Urgent scheduling rules (U1, U3, cap, rollback in orion/gpu_pool/scheduler.py).
Spec: docs/superpowers/specs/2026-09-28-urgent-curiosity-and-hardware-watch-design.md Part 1."""
from __future__ import annotations

from datetime import timedelta

import pytest

from orion.gpu_pool.scheduler import (
    Abort, CardLive, LeaseView, Recall, RoleLive, SwapBlocked, SwapLoad, schedule,
)
from orion.gpu_pool.tests.test_scheduler import CFG, T0, cards, grants, lease, live, of, run

CLEAR = {"thermal": None, "visual_baseline": None}
GRACE = timedelta(seconds=CFG.defaults.urgent_preempt_grace_sec)
LENT = cards(gpu0=CardLive("gpu0", lent=True))


def hold(status="queued", role=None, priority="background", **kw):
    return lease("agent", status, role, priority=priority, kind="hold", retryable=True, **kw)


def urgent(**kw):
    return hold(priority="urgent", **kw)


def agent_slots(n):
    return live(agent=RoleLive("agent", True, n, 131072, False))


def preempts(decisions):
    return [r.lease_id for r in of(Recall, decisions) if r.reason == "urgent_preempt"]


def with_cap(n):
    return CFG.model_copy(update={"defaults": CFG.defaults.model_copy(update={"urgent_max_concurrent": n})})


def test_defaults_are_what_the_plan_says():
    assert CFG.defaults.urgent_preempt_grace_sec == 5
    assert CFG.defaults.urgent_max_concurrent == 3
    assert LeaseView("x", "agent", "system", "queued", T0).reason is None


# --- U1: pause one background (then system) hold per waiting urgent lease -------------------
def test_urgent_behind_a_background_hold_pauses_it_with_the_short_grace():
    v = hold("granted", "agent", lease_id="v", granted_at=T0 - timedelta(seconds=30))
    u = urgent(lease_id="u")
    decisions = run([v, u])
    assert grants(decisions) == {}
    assert of(Recall, decisions) == [Recall("v", T0 + GRACE, "urgent_preempt")]


def test_background_victim_is_picked_before_system():
    sys_h = hold("granted", "agent", priority="system", lease_id="s", granted_at=T0 - timedelta(seconds=1))
    bg = hold("granted", "agent", lease_id="b", granted_at=T0 - timedelta(seconds=100))
    assert preempts(run([sys_h, bg, urgent(lease_id="u")], roles=agent_slots(2))) == ["b"]
    # with no background left, a system hold is paused
    assert preempts(run([sys_h, lease("agent", "granted", "agent"), urgent(lease_id="u")],
                        roles=agent_slots(2))) == ["s"]


def test_most_recently_granted_background_hold_is_picked():
    old = hold("granted", "agent", lease_id="old", granted_at=T0 - timedelta(seconds=100))
    new = hold("granted", "agent", lease_id="new", granted_at=T0 - timedelta(seconds=10))
    assert preempts(run([old, new, urgent(lease_id="u")], roles=agent_slots(2))) == ["new"]


@pytest.mark.parametrize("occupant", [
    lambda: lease("agent", "granted", "agent", priority="interactive", lease_id="occ"),
    lambda: urgent(status="granted", role="agent", lease_id="occ"),
    lambda: hold("granted", "agent", operator=True, lease_id="occ"),
    lambda: lease("agent", "granted", "agent", priority="system", lease_id="occ"),   # agent-burst request
    lambda: lease("agent", "granted", "agent", priority="background", lease_id="occ"),
    lambda: lease("agent", "granted", "agent", priority="background", hold_lease_id="elsewhere",
                  lease_id="occ"),                                                    # a child
], ids=["interactive", "urgent_hold", "operator_hold", "system_request", "background_request", "child"])
def test_never_paused(occupant):
    decisions = run([occupant(), urgent(lease_id="u")])
    assert grants(decisions) == {} and of(Recall, decisions) == []


def test_a_pause_already_under_way_is_not_repeated_every_tick():
    pausing = hold("recalling", "agent", lease_id="p", reason="urgent_preempt", recall_by=T0 + timedelta(seconds=3))
    other = hold("granted", "agent", lease_id="o", granted_at=T0 - timedelta(seconds=10))
    u = urgent(lease_id="u")
    assert of(Recall, run([pausing, other, u], roles=agent_slots(2))) == []
    # a recall for another reason does not count as a pause for the urgent lease
    owner_recall = hold("recalling", "agent", lease_id="p", reason="max_hold", recall_by=T0 + timedelta(seconds=3))
    assert preempts(run([owner_recall, other, u], roles=agent_slots(2))) == ["o"]


def test_two_waiting_urgent_leases_pause_two_holds():
    a = hold("granted", "agent", lease_id="a", granted_at=T0 - timedelta(seconds=20))
    b = hold("granted", "agent", lease_id="b", granted_at=T0 - timedelta(seconds=10))
    u1, u2 = urgent(lease_id="u1", age=2), urgent(lease_id="u2", age=1)
    assert preempts(run([a, b, u1, u2], roles=agent_slots(2))) == ["b", "a"]
    # one already pausing covers the first; the second urgent lease pauses one more
    pausing = hold("recalling", "agent", lease_id="b", reason="urgent_preempt", recall_by=T0 + timedelta(seconds=3))
    assert preempts(run([a, pausing, u1, u2], roles=agent_slots(2))) == ["a"]


def test_expired_pause_aborts_with_urgent_preempt_other_recalls_keep_their_reason():
    p = hold("recalling", "agent", lease_id="p", reason="urgent_preempt", recall_by=T0 - timedelta(seconds=1))
    assert of(Abort, run([p])) == [Abort("p", "urgent_preempt")]
    r = hold("recalling", "agent", lease_id="r", reason="owner_waiting", recall_by=T0 - timedelta(seconds=1))
    assert of(Abort, run([r])) == [Abort("r", "recall_grace_exceeded")]


def test_freed_slot_goes_to_urgent_before_older_background_and_interactive():
    u = urgent(lease_id="u", age=0)
    old_bg = hold(lease_id="bg", age=100)
    old_int = lease("agent", priority="interactive", lease_id="i", age=100)
    assert grants(run([old_bg, old_int, u])) == {"u": "agent"}


def test_background_hold_still_never_preempts():
    busy = hold("granted", "agent", lease_id="busy")
    bg = hold(lease_id="h")
    sys_h = hold(lease_id="s", priority="system")
    decisions = run([busy, bg, sys_h])
    assert grants(decisions) == {} and of(Recall, decisions) == []


def test_expiring_hold_is_not_paused():
    gone = hold("granted", "agent", lease_id="g", expires_at=T0 - timedelta(seconds=1))
    decisions = run([gone, urgent(lease_id="u")])
    assert preempts(decisions) == [] and grants(decisions) == {"u": "agent"}


def test_borrowing_urgent_hold_does_not_pause_where_the_owner_would_still_block_it():
    gpu3_full = [lease("metacog", "granted", "metacog") for _ in range(4)] + \
                [lease("fast", "granted", "fast") for _ in range(4)]
    v = hold("granted", "agent", lease_id="v")                                  # agent-owned background run
    u = lease("metacog", priority="urgent", kind="hold", retryable=True, lease_id="u")
    # another agent-owned call on agent: the metacog hold could not borrow agent even once v left
    owner_busy = lease("agent", "granted", "agent", priority="system")
    assert preempts(run(gpu3_full + [v, owner_busy, u], roles=agent_slots(2))) == []
    # v is the only owner there: pausing it frees agent for the urgent borrower
    assert preempts(run(gpu3_full + [v, u])) == ["v"]


# --- U3: stacking past H1, cap -----------------------------------------------------------
def test_urgent_hold_stacks_on_a_role_that_already_has_a_hold():
    bg = hold("granted", "agent", lease_id="bg")
    u = urgent(lease_id="u")
    h2 = hold(lease_id="h2", age=10)
    assert grants(run([bg, h2], roles=agent_slots(2))) == {}               # H1 still refuses non-urgent
    assert grants(run([bg, u, h2], roles=agent_slots(2))) == {"u": "agent"}
    assert grants(run([bg, hold("granted", "agent", priority="urgent"), h2], roles=agent_slots(3))) == {}


def test_urgent_stacking_is_bounded_by_slots():
    bg = hold("granted", "agent", lease_id="bg")
    u1 = urgent(status="granted", role="agent", lease_id="u1")
    u2 = urgent(lease_id="u2")
    assert grants(run([bg, u1, u2], roles=agent_slots(2))) == {}


def test_cap_blocks_grant_and_pause_for_a_fourth_urgent_lease():
    active = [urgent(status="granted", role="agent", lease_id=f"a{i}") for i in range(3)]
    bg = hold("granted", "agent", lease_id="bg", granted_at=T0 - timedelta(seconds=5))
    u4 = urgent(lease_id="u4")
    full = run(active + [bg, u4], roles=agent_slots(4))
    assert grants(full) == {} and of(Recall, full) == []
    assert grants(run(active + [u4], roles=agent_slots(5))) == {}          # free slot, still capped
    # under the cap, the same shape pauses the background hold
    assert preempts(run(active[:2] + [bg, u4], roles=agent_slots(3))) == ["bg"]


def test_cap_counts_grants_made_this_tick():
    us = [urgent(lease_id=f"u{i}", age=10 - i) for i in range(4)]
    got = grants(run(us, roles=agent_slots(4)))
    assert set(got) == {"u0", "u1", "u2"}


def test_rollback_cap_zero_makes_urgent_plain_background():
    cfg0 = with_cap(0)
    v = hold("granted", "agent", lease_id="v", granted_at=T0 - timedelta(seconds=30))
    d = schedule(cfg0, live(), cards(), [v, urgent(lease_id="u")], T0)
    assert grants(d) == {} and of(Recall, d) == []
    stack = schedule(cfg0, agent_slots(2), cards(), [v, urgent(lease_id="u")], T0)
    assert grants(stack) == {}                                              # H1 applies again
    # and it queues behind older background work instead of jumping it
    older = hold(lease_id="older", age=100)
    assert grants(schedule(cfg0, live(), cards(), [urgent(lease_id="u"), older], T0)) == {"older": "agent"}


# --- chat: owners win on their own role ---------------------------------------------------
def test_gpu0_not_lent_urgent_never_touches_chat():
    busy = lease("agent", "granted", "agent")                               # not a victim
    on_chat = hold("granted", "chat", lease_id="c")                         # borrowing, being unlent
    decisions = run([busy, on_chat, urgent(lease_id="u")])
    assert grants(decisions) == {}
    assert [(r.lease_id, r.reason) for r in of(Recall, decisions)] == [("c", "card_unlent")]
    assert grants(run([busy, urgent(lease_id="u")])) == {}                 # free chat slot, not lent


def test_gpu0_lent_background_hold_on_chat_may_be_paused():
    busy = lease("agent", "granted", "agent")
    on_chat = hold("granted", "chat", lease_id="c")
    assert preempts(run([busy, on_chat, urgent(lease_id="u")], crds=LENT)) == ["c"]


def test_gpu0_lent_waiting_chat_owner_beats_urgent_to_chat():
    busy = lease("agent", "granted", "agent")
    owner = lease("chat", priority="interactive", lease_id="o", age=0)
    u = urgent(lease_id="u", age=100)
    decisions = run([busy, owner, u], crds=LENT)
    assert grants(decisions) == {"o": "chat"}
    assert preempts(decisions) == []


def test_gpu0_lent_chat_owner_not_paused_for_urgent():
    busy = lease("agent", "granted", "agent")
    chatting = lease("chat", "granted", "chat", priority="interactive")
    decisions = run([busy, chatting, urgent(lease_id="u")], crds=LENT)
    assert grants(decisions) == {} and of(Recall, decisions) == []


def test_chat_owner_uses_the_gap_of_an_urgent_hold_borrowing_chat():
    busy = lease("agent", "granted", "agent")
    u = urgent(status="granted", role="chat", lease_id="u")
    owner = lease("chat", priority="interactive", lease_id="o")
    decisions = run([busy, u, owner], crds=LENT)
    assert grants(decisions) == {"o": "chat"}
    assert [(r.lease_id, r.reason) for r in of(Recall, decisions)] == [("u", "owner_waiting")]
    # on its own role an urgent hold's gap stays closed to lower priority
    home = urgent(status="granted", role="agent", lease_id="h")
    assert grants(run([home, lease("agent", priority="interactive", lease_id="i")])) == {}


# --- U3: swap seat after_wait_sec skipped, guards still apply ------------------------------
def _seat_demand(**kw):
    return [lease("agent", "granted", "agent"), urgent(lease_id="u", queued_since=T0, **kw)]


def test_urgent_skips_after_wait_sec_for_a_swap_seat():
    d = schedule(CFG, live(), cards(), _seat_demand(), T0, guards=CLEAR)
    assert of(SwapLoad, d) == [SwapLoad("agent-gpu2", "demand")]
    bg = [lease("agent", "granted", "agent"), hold(lease_id="w", queued_since=T0)]
    assert of(SwapLoad, schedule(CFG, live(), cards(), bg, T0, guards=CLEAR)) == []


def test_urgent_seat_load_still_obeys_guards():
    hot = {"thermal": "hot", "visual_baseline": None}
    d = schedule(CFG, live(), cards(), _seat_demand(), T0, guards=hot)
    assert not of(SwapLoad, d)
    assert [(b.role, b.reason, b.detail) for b in of(SwapBlocked, d)] == [("agent-gpu2", "guard:thermal", "hot")]


def test_urgent_being_served_by_a_pause_does_not_also_load_a_seat():
    v = hold("granted", "agent", lease_id="v")
    d = schedule(CFG, live(), cards(), [v, urgent(lease_id="u", queued_since=T0)], T0, guards=CLEAR)
    assert preempts(d) == ["v"] and not of(SwapLoad, d)


def test_capped_urgent_does_not_skip_after_wait_sec():
    active = [urgent(status="granted", role="agent", lease_id=f"a{i}") for i in range(3)]
    d = schedule(CFG, agent_slots(3), cards(), active + [urgent(lease_id="u4", queued_since=T0)], T0,
                 guards=CLEAR)
    assert not of(SwapLoad, d)
