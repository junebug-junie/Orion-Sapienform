"""Urgent scheduling rules (U1, U3, cap, rollback in orion/gpu_pool/scheduler.py).
Spec: docs/superpowers/specs/2026-09-28-urgent-curiosity-and-hardware-watch-design.md Part 1."""
from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest

from orion.gpu_pool.scheduler import (
    Abort, CardLive, Grant, LeaseView, Recall, RoleLive, SwapBlocked, SwapLoad, schedule,
)
from orion.gpu_pool.tests.test_scheduler import CFG, T0, cards, grants, lease, live, of, run

CLEAR = {"thermal": None}
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


def test_one_preempt_reason_constant():
    from orion.gpu_pool import lease_graph, scheduler
    from orion.schemas.gpu_pool import URGENT_PREEMPT
    assert scheduler.PREEMPT is URGENT_PREEMPT and lease_graph.URGENT_PREEMPT is URGENT_PREEMPT


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
    lambda: hold("granted", "agent", priority="interactive", lease_id="occ"),
    lambda: urgent(status="granted", role="agent", lease_id="occ"),
    lambda: hold("granted", "agent", operator=True, lease_id="occ"),
    lambda: lease("agent", "granted", "agent", priority="system", lease_id="occ"),   # agent-burst request
    lambda: lease("agent", "granted", "agent", priority="background", lease_id="occ"),
    lambda: lease("agent", "granted", "agent", priority="background", hold_lease_id="elsewhere",
                  lease_id="occ"),                                                    # a child
], ids=["interactive", "interactive_hold", "urgent_hold", "operator_hold", "system_request",
        "background_request", "child"])
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


def test_grace_expiry_tick_aborts_the_victim_without_pausing_another():
    p = hold("recalling", "agent", lease_id="p", reason="urgent_preempt", recall_by=T0)
    other = hold("granted", "agent", lease_id="o", granted_at=T0 - timedelta(seconds=10))
    decisions = run([p, other, urgent(lease_id="u")], roles=agent_slots(2))
    assert of(Abort, decisions) == [Abort("p", "urgent_preempt")]
    assert of(Recall, decisions) == []


def test_a_pause_serves_only_an_urgent_lease_that_could_use_that_role():
    # a pause under way on diffusion frees nothing a metacog urgent lease can use
    p = lease("diffusion", "recalling", "diffusion", priority="background", kind="hold", retryable=True,
              lease_id="p", reason="urgent_preempt", recall_by=T0 + timedelta(seconds=3))
    fast_v = lease("fast", "granted", "agent", priority="background", kind="hold", retryable=True,
                   lease_id="fv", granted_at=T0 - timedelta(seconds=10))
    assert preempts(run(_gpu3_full() + [p, fast_v, _metacog_urgent()])) == ["fv"]
    # a pause under way on agent (by a borrower it could replace) does count
    p_agent = replace(fast_v, lease_id="p2", status="recalling", reason="urgent_preempt",
                      recall_by=T0 + timedelta(seconds=3))
    assert preempts(run(_gpu3_full() + [p_agent, _metacog_urgent()])) == []


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


def _gpu3_full():
    return [lease("metacog", "granted", "metacog") for _ in range(4)] + \
           [lease("fast", "granted", "fast") for _ in range(4)]


def _metacog_urgent():
    return lease("metacog", priority="urgent", kind="hold", retryable=True, lease_id="u")


def test_borrowing_urgent_hold_never_pauses_the_roles_owner():
    v = hold("granted", "agent", lease_id="v")                                  # agent-owned background run
    owner_busy = lease("agent", "granted", "agent", priority="system")
    assert preempts(run(_gpu3_full() + [v, owner_busy, _metacog_urgent()], roles=agent_slots(2))) == []
    # even as the only owner there: re-queued in place, v would win its own role back (owners first)
    assert preempts(run(_gpu3_full() + [v, _metacog_urgent()])) == []


def _simulate(leases, ticks=5, roles=None, crds=None, before_tick=None, cfg=CFG):
    """Tick the scheduler, applying its decisions the way the runtime does: a preempt Abort
    re-queues the hold in place (same lease_id and created_at; Task 3's resume-in-place).
    ``before_tick(tick, state)`` may change the world between ticks (a child call ending)."""
    state = {l.lease_id: l for l in leases}
    now, paused, got = T0, [], {}
    for tick in range(ticks):
        if before_tick is not None:
            before_tick(tick, state)
        for x in schedule(cfg, roles or live(), crds or cards(), list(state.values()), now):
            if isinstance(x, Grant):
                state[x.lease_id] = replace(state[x.lease_id], status="granted", role=x.role, granted_at=now)
                got[x.lease_id] = x.role
            elif isinstance(x, Recall):
                paused += [x.lease_id] if x.reason == "urgent_preempt" else []
                state[x.lease_id] = replace(state[x.lease_id], status="recalling", recall_by=x.recall_by,
                                            reason=x.reason)
            elif isinstance(x, Abort):
                state[x.lease_id] = replace(state[x.lease_id], status="queued", role=None, recall_by=None,
                                            granted_at=None, reason=x.reason)
        now += GRACE
    return paused, got


def test_no_pause_loop_when_the_victim_owns_the_role_the_urgent_lease_borrows():
    v = hold("granted", "agent", lease_id="v", age=100, granted_at=T0 - timedelta(seconds=50))
    paused, got = _simulate(_gpu3_full() + [v, _metacog_urgent()])
    assert paused == [] and "u" not in got


def test_pause_abort_requeue_gives_the_slot_to_urgent_exactly_once():
    # same owner: urgent is first in line for the freed slot, ahead of the re-queued victim
    v = hold("granted", "agent", lease_id="v", age=100, granted_at=T0 - timedelta(seconds=50))
    paused, got = _simulate([v, urgent(lease_id="u")])
    assert paused == ["v"] and got == {"u": "agent"}
    # both borrow agent: the urgent borrower is still ahead of the re-queued one
    fast_v = lease("fast", "granted", "agent", priority="background", kind="hold", retryable=True,
                   lease_id="v", age=100, granted_at=T0 - timedelta(seconds=50))
    paused, got = _simulate(_gpu3_full() + [fast_v, _metacog_urgent()])
    assert paused == ["v"] and got == {"u": "agent"}


# --- a paused hold's call still in flight is the pause, not a reason for another one -----------
def _child(hold_id="v", role="agent", lease_id="c"):
    return lease("agent", "granted", role, priority="background", hold_lease_id=hold_id, lease_id=lease_id)


def test_a_requeued_holds_child_still_on_the_slot_is_the_pause_in_flight():
    v = hold(lease_id="v", age=100, reason="urgent_preempt")               # re-queued in place
    w = hold("granted", "chat", lease_id="w", granted_at=T0 - timedelta(seconds=100))   # borrowing lent chat
    u = urgent(lease_id="u")
    d = run([v, _child(), w, u], crds=LENT)
    assert of(Recall, d) == [] and grants(d) == {}
    # the call ends: the slot is urgent's
    assert grants(run([v, w, u], crds=LENT)) == {"u": "agent"}


def test_the_in_flight_child_only_serves_an_urgent_lease_that_could_take_its_slot():
    # the child sits on diffusion: nothing a metacog urgent lease can use, so it is owed a pause
    v = lease("diffusion", priority="background", kind="hold", retryable=True, lease_id="v", age=100,
              reason="urgent_preempt")
    c = lease("diffusion", "granted", "diffusion", priority="background", hold_lease_id="v", lease_id="c")
    fast_v = lease("fast", "granted", "agent", priority="background", kind="hold", retryable=True,
                   lease_id="fv", granted_at=T0 - timedelta(seconds=10))
    assert preempts(run(_gpu3_full() + [v, c, fast_v, _metacog_urgent()])) == ["fv"]
    # a child of a hold re-queued for another reason is no pause
    other = replace(hold(lease_id="v", age=100), reason="owner_waiting")
    w = hold("granted", "chat", lease_id="w", granted_at=T0 - timedelta(seconds=100))
    assert preempts(run([other, _child(), w, urgent(lease_id="u")], crds=LENT)) == ["w"]


def test_one_pause_total_while_the_paused_runs_call_finishes():
    v = hold("granted", "agent", lease_id="v", age=100, granted_at=T0 - timedelta(seconds=10))
    w = hold("granted", "chat", lease_id="w", age=100, granted_at=T0 - timedelta(seconds=100))

    def call_ends_late(tick, state):
        if tick == 4:
            state.pop("c", None)

    paused, got = _simulate([v, _child(), w, urgent(lease_id="u")], ticks=6, crds=LENT,
                            before_tick=call_ends_late)
    assert paused == ["v"] and got == {"u": "agent"}


# --- U1 only pauses a hold on a role the urgent lease's class may use --------------------------
def _world_full():
    return [lease("world", "granted", "world") for _ in range(2)]


def _world_urgent():
    return lease("world", priority="urgent", kind="hold", retryable=True, lease_id="u")


def test_urgent_never_pauses_a_hold_on_a_role_outside_its_class():
    on_chat = hold("granted", "chat", lease_id="c")                        # agent run borrowing lent chat
    d = run(_world_full() + [on_chat, _world_urgent()], crds=LENT)
    assert of(Recall, d) == [] and grants(d) == {}
    paused, got = _simulate(_world_full() + [on_chat, _world_urgent()], crds=LENT)
    assert paused == [] and got == {}


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


# --- urgent owner reclaiming a borrowed role: the borrower is paused, not given 600 s ---------
def _metacog_bg_on_agent(lease_id="mb", priority="background", **kw):
    return lease("metacog", "granted", "agent", priority=priority, kind="hold", retryable=True,
                 lease_id=lease_id, **kw)


GPU2_LOADED = cards(gpu2=CardLive("gpu2", swapped_in={"agent-gpu2"}))


def test_urgent_owner_reclaims_a_borrowing_hold_with_the_preempt_grace():
    mb = _metacog_bg_on_agent()
    v2 = hold("granted", "agent-gpu2", lease_id="v2")               # another pausable run it could use
    decisions = run(_gpu3_full() + [mb, v2, urgent(lease_id="u")], crds=GPU2_LOADED)
    assert of(Recall, decisions) == [Recall("mb", T0 + GRACE, "urgent_preempt")]   # no second victim


def test_non_urgent_owner_still_reclaims_with_owner_waiting_and_the_hold_grace():
    mb = _metacog_bg_on_agent()
    decisions = run(_gpu3_full() + [mb, hold(priority="system", lease_id="o")])
    assert of(Recall, decisions) == [
        Recall("mb", T0 + timedelta(seconds=CFG.defaults.hold_clawback_grace_sec), "owner_waiting")]


@pytest.mark.parametrize("borrower", [
    lambda: _metacog_bg_on_agent(priority="interactive"),
    lambda: _metacog_bg_on_agent(operator=True),
    lambda: lease("metacog", "granted", "agent", priority="background", lease_id="mb"),   # a request
], ids=["interactive_hold", "operator_hold", "request"])
def test_ineligible_borrower_keeps_owner_waiting_for_an_urgent_owner(borrower):
    decisions = run(_gpu3_full() + [borrower(), urgent(lease_id="u")])
    assert [(r.lease_id, r.reason) for r in of(Recall, decisions)] == [("mb", "owner_waiting")]


def test_one_borrower_is_paused_per_urgent_owner_the_rest_keep_owner_waiting():
    newer = _metacog_bg_on_agent("new", granted_at=T0 - timedelta(seconds=10))
    older = _metacog_bg_on_agent("old", granted_at=T0 - timedelta(seconds=100))
    decisions = run(_gpu3_full() + [newer, older, urgent(lease_id="u")], roles=agent_slots(2))
    assert sorted((r.lease_id, r.reason) for r in of(Recall, decisions)) == [
        ("new", "urgent_preempt"), ("old", "owner_waiting")]


def test_urgent_owner_reclaim_pauses_once_then_takes_the_slot():
    mb = _metacog_bg_on_agent(age=100, granted_at=T0 - timedelta(seconds=50))
    paused, got = _simulate(_gpu3_full() + [mb, urgent(lease_id="u")])
    assert paused == ["mb"] and got == {"u": "agent"}


def test_capped_urgent_is_not_a_waiting_owner():
    active = [urgent(status="granted", role="agent", lease_id=f"a{i}") for i in range(3)]
    borrower = lease("fast", "granted", "agent", lease_id="b")
    queued_borrower = lease("fast", lease_id="qb")
    u4 = urgent(lease_id="u4")
    decisions = run(_gpu3_full() + active + [borrower, queued_borrower, u4], roles=agent_slots(5))
    assert of(Recall, decisions) == []                       # no owner_waiting recall for u4
    assert grants(decisions) == {"qb": "agent"}              # and u4 does not block borrowers


def test_only_urgent_owners_within_the_cap_room_count_as_waiting_owners():
    active = [urgent(status="granted", role="agent", lease_id=f"a{i}") for i in range(2)]
    borrowers = [lease("fast", "granted", "agent", lease_id=f"b{i}", granted_at=T0 - timedelta(seconds=i))
                 for i in range(3)]
    waiting = [urgent(lease_id=f"u{i}", age=10 - i) for i in range(3)]
    decisions = run(_gpu3_full() + active + borrowers + waiting, roles=agent_slots(5))
    assert [(r.lease_id, r.reason) for r in of(Recall, decisions)] == [("b0", "owner_waiting")]


def test_cap_room_is_recounted_after_this_ticks_urgent_grants():
    # one urgent lease takes the free slot this tick and fills the cap: the others wait on nothing
    active = [urgent(status="granted", role="agent", lease_id=f"a{i}") for i in range(2)]
    borrower = lease("fast", "granted", "agent", lease_id="b")
    waiting = [urgent(lease_id=f"u{i}", age=10 - i) for i in range(2)]
    decisions = run(_gpu3_full() + active + [borrower] + waiting, roles=agent_slots(4))
    assert grants(decisions) == {"u0": "agent"} and of(Recall, decisions) == []


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
    hot = {"thermal": "hot"}
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
