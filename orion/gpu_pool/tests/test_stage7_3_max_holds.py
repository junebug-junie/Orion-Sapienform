"""Stage 7.3 (docs/superpowers/specs/2026-09-30-gpu-pool-stage7-concurrency.md): ``max_holds`` per
role replaces H1's hard-coded one, and a gap-sharing call is pinned to ONE idle hold so it cannot
stall two runs. Acceptance checks 4 (two runs at once), 9 (gap pinning) and 10 (rollback drill)."""
from __future__ import annotations

from datetime import timedelta

import pytest

from orion.gpu_pool.config import PoolConfig, RoleSpec, check_max_holds, load_pool_config
from orion.gpu_pool.discovery import load_profiles
from orion.gpu_pool.scheduler import CardLive, Grant, Recall, RoleLive, SwapUnload, hold_cap
from orion.gpu_pool.tests.test_scheduler import CFG, T0, cards, grants, lease, live, of, run

SEAT = "agent-gpu2"
TWO = live(**{SEAT: RoleLive(SEAT, True, 2, 131072, False)})   # Bonsai 2 x 131072 (stage 7.2, live)
LOADED = cards(gpu2=CardLive("gpu2", swapped_in={SEAT}, loaded_at=T0, last_active_at=T0))


def hold(lease_id, status="queued", role=None, priority="background", age=0, **kw):
    return lease("agent", status, role, priority=priority, kind="hold", retryable=True,
                 lease_id=lease_id, age=age, **kw)


def child(hold_id, lease_id, status="queued", role=None, **kw):
    return lease("agent", status, role, priority="system", hold_lease_id=hold_id, lease_id=lease_id, **kw)


def with_role(cfg: PoolConfig, role: str, **update) -> PoolConfig:
    roles = dict(cfg.roles)
    roles[role] = roles[role].model_copy(update=update)
    return cfg.model_copy(update={"roles": roles})


def sched(leases, cfg=CFG, roles=TWO, crds=LOADED, now=T0):
    from orion.gpu_pool.scheduler import schedule
    return schedule(cfg, roles, crds, leases, now, guards={"thermal": None})


def home_busy():
    """gpu1's agent role carries its one hold with a call in flight: new runs and one-off agent
    calls both go to the seat (agent-deep and chat are lend-gated and not lent here)."""
    return [hold("home", "granted", "agent", age=1000, granted_at=T0 - timedelta(seconds=1000)),
            child("home", "home-call", "granted", "agent")]


# --- config -------------------------------------------------------------------------------
def test_agent_gpu2_takes_two_holds_and_every_other_role_keeps_one():
    assert CFG.roles[SEAT].max_holds == 2
    assert CFG.roles[SEAT].reserve_one_off_slots == 0
    assert {r: s.max_holds for r, s in CFG.roles.items() if r != SEAT} == {r: 1 for r in CFG.roles if r != SEAT}


def test_service_role_with_more_holds_than_declared_slots_is_refused():
    with pytest.raises(ValueError, match="max_holds 2 > slots 1"):
        RoleSpec(kind="service", cards=["gpu2"], port=9000, slots=1, vram_gb=1, max_holds=2)
    with pytest.raises(ValueError, match="leaves no slot"):
        RoleSpec(kind="service", cards=["gpu2"], port=9000, slots=2, vram_gb=1, reserve_one_off_slots=2)


def test_static_gate_max_holds_vs_the_profile_the_pool_loads():
    profiles = load_profiles()
    assert check_max_holds(CFG, profiles) == []          # Bonsai first: n_parallel 2
    launch = CFG.roles[SEAT].launch
    rolled_back = with_role(CFG, SEAT, launch=launch.model_copy(update={"profiles": list(reversed(launch.profiles))}))
    [problem] = check_max_holds(rolled_back, profiles)   # Q4 first (1 slot) with max_holds still 2
    assert "max_holds 2 > n_parallel 1" in problem
    reserved = with_role(CFG, SEAT, reserve_one_off_slots=1)
    assert any("never be reached" in p for p in check_max_holds(reserved, profiles))


# --- hold_cap: clamp to discovered slots, with a reason --------------------------------------
def test_hold_cap_clamps_to_discovered_slots_and_says_why():
    assert hold_cap(CFG, SEAT, TWO[SEAT]) == (2, None)
    assert hold_cap(CFG, SEAT, RoleLive(SEAT, True, 1, 131072)) == (1, "max_holds 2 > discovered slots 1")
    assert hold_cap(CFG, SEAT, None) == (0, "no_slots")
    assert hold_cap(CFG, "agent", live()["agent"]) == (1, None)


def test_reserve_keeps_a_slot_for_one_offs_but_never_the_first_hold():
    reserved = with_role(CFG, SEAT, reserve_one_off_slots=1)
    assert hold_cap(reserved, SEAT, TWO[SEAT]) == (1, "reserve_one_off_slots 1 of 2 slots")
    one_slot = with_role(CFG, "agent", reserve_one_off_slots=1)
    assert hold_cap(one_slot, "agent", live()["agent"]) == (1, None)


# --- acceptance 4: two runs at once on the seat ----------------------------------------------
def test_second_run_is_granted_on_the_seat_beside_the_first():
    a = hold("a", "granted", SEAT, granted_at=T0 - timedelta(seconds=60))
    b = hold("b")
    assert grants(sched(home_busy() + [a, b])) == {"b": SEAT}


def test_a_third_run_waits_at_max_holds():
    a = hold("a", "granted", SEAT)
    b = hold("b", "granted", SEAT)
    c = hold("c")
    assert grants(sched(home_busy() + [a, b, c])) == {}


def test_one_tick_grants_two_queued_runs_on_an_empty_seat_oldest_first():
    b = hold("b", age=20)
    c = hold("c", age=10)
    d = hold("d", age=5)
    assert grants(sched(home_busy() + [b, c, d])) == {"b": SEAT, "c": SEAT}


def test_clamped_when_discovery_shows_one_slot():
    one = live(**{SEAT: RoleLive(SEAT, True, 1, 131072, False)})
    a = hold("a", "granted", SEAT)
    assert grants(sched(home_busy() + [a, hold("b")], roles=one)) == {}


def test_reserve_one_off_slot_keeps_the_second_slot_for_one_off_calls():
    reserved = with_role(CFG, SEAT, reserve_one_off_slots=1)
    a = hold("a", "granted", SEAT)
    s = lease("agent", priority="system", lease_id="s", age=1)
    got = grants(sched(home_busy() + [a, hold("b", age=5), s], cfg=reserved))
    assert got == {"s": SEAT}


def test_urgent_holds_still_stack_past_max_holds_bounded_by_slots():
    a = hold("a", "granted", SEAT)
    u = hold("u", priority="urgent")
    assert grants(sched(home_busy() + [a, u])) == {"u": SEAT}


# --- acceptance 10: rollback drill ---------------------------------------------------------
def test_rollback_to_one_hold_grants_no_second_hold_next_tick_and_recalls_nothing():
    rolled = with_role(CFG, SEAT, max_holds=1)
    a = hold("a", "granted", SEAT)
    b = hold("b")
    decisions = sched(home_busy() + [a, b], cfg=rolled)
    assert grants(decisions) == {}
    assert not of(Recall, decisions)                    # running runs finish; only new grants stop
    # And the shipped default (no key) is the same rule.
    data = CFG.model_dump(by_alias=True)
    data["roles"][SEAT].pop("max_holds")
    assert PoolConfig.model_validate(data).roles[SEAT].max_holds == 1


# --- acceptance 9: gap pinning ---------------------------------------------------------------
def two_idle_runs():
    a = hold("a", "granted", SEAT, granted_at=T0 - timedelta(seconds=300))
    b = hold("b", "granted", SEAT, granted_at=T0 - timedelta(seconds=200))
    return [a, b]


def test_one_interloper_never_stalls_both_runs():
    """The spec's replay: before 7.3 neither run's next call was granted (2 - 1 used - 1 other idle)."""
    s = lease("agent", "granted", SEAT, priority="system", lease_id="s")
    ca, cb = child("a", "ca", age=2), child("b", "cb", age=1)
    got = grants(sched(home_busy() + two_idle_runs() + [s, ca, cb]))
    assert len(got) == 1 and set(got.values()) == {SEAT}
    # Both runs want their slot: the gap is charged to the most recently granted (b), so a goes.
    assert got == {"ca": SEAT}


def test_the_run_that_wants_its_slot_gets_it_even_if_it_is_the_newest():
    s = lease("agent", "granted", SEAT, priority="system", lease_id="s")
    cb = child("b", "cb")
    assert grants(sched(home_busy() + two_idle_runs() + [s, cb])) == {"cb": SEAT}
    ca = child("a", "ca")
    assert grants(sched(home_busy() + two_idle_runs() + [s, ca])) == {"ca": SEAT}


def test_a_runs_next_call_goes_ahead_of_a_queued_one_off():
    s = lease("agent", "granted", SEAT, priority="system", lease_id="s")
    waiting = lease("agent", priority="interactive", lease_id="w", age=100)
    ca = child("a", "ca")
    got = grants(sched(home_busy() + two_idle_runs() + [s, waiting, ca]))
    assert got == {"ca": SEAT}   # w is not granted: a's call took a's slot, b's gap is s's


def test_one_off_still_uses_a_gap_and_each_gap_takes_at_most_one():
    s1 = lease("agent", priority="system", lease_id="s1", age=3)
    s2 = lease("agent", priority="system", lease_id="s2", age=2)
    s3 = lease("agent", priority="system", lease_id="s3", age=1)
    got = grants(sched(home_busy() + two_idle_runs() + [s1, s2, s3]))
    assert {k: v for k, v in got.items() if v == SEAT} == {"s1": SEAT, "s2": SEAT}
    assert "s3" not in got or got["s3"] != SEAT


def test_two_interlopers_cost_each_run_one_call_never_more():
    s1 = lease("agent", "granted", SEAT, priority="system", lease_id="s1")
    s2 = lease("agent", "granted", SEAT, priority="system", lease_id="s2")
    ca, cb = child("a", "ca"), child("b", "cb")
    assert grants(sched(home_busy() + two_idle_runs() + [s1, s2, ca, cb])) == {}
    assert grants(sched(home_busy() + two_idle_runs() + [s2, ca, cb])) == {"ca": SEAT}


def test_lower_priority_never_borrows_a_gap_on_a_full_seat():
    low = lease("agent", priority="background", lease_id="low")
    assert grants(sched(home_busy() + two_idle_runs() + [low])) == {}


def test_busy_run_and_idle_run_one_off_takes_the_idle_gap_only():
    a, b = two_idle_runs()
    ca = child("a", "ca", "granted", SEAT)
    s = lease("agent", priority="system", lease_id="s")
    assert grants(sched(home_busy() + [a, b, ca, s])) == {"s": SEAT}
    s_on = lease("agent", "granted", SEAT, priority="system", lease_id="s")
    cb = child("b", "cb")
    assert grants(sched(home_busy() + [a, b, ca, s_on, cb])) == {}   # b waits one call (s), a's runs


def test_one_slot_one_hold_is_unchanged():
    """Gap pinning at 1 slot and 1 hold is the old rule exactly (gpu1's agent)."""
    h = hold("h", "granted", "agent")
    s = lease("agent", "granted", "agent", priority="system", lease_id="s")
    c = child("h", "c")
    assert grants(run([h, s, c])) == {}
    assert grants(run([h, c])) == {"c": "agent"}


# --- recall / drain with two holds on the seat -----------------------------------------------
def test_owner_reclaim_recalls_both_runs_then_unloads_once_both_leave():
    """Diffusion (gpu2's owner) wants its card: the seat drains -- both holds recalled in the same
    tick with the hold grace (two take-backs), nothing new lands, and the unload waits for both."""
    a, b = two_idle_runs()
    img = lease("diffusion", priority="background", kind="hold", retryable=True, lease_id="img")
    decisions = sched(home_busy() + [a, b, img, hold("c")])
    recalls = sorted((r.lease_id, r.reason) for r in of(Recall, decisions))
    assert recalls == [("a", "draining"), ("b", "draining")]
    assert all(r.recall_by == T0 + timedelta(seconds=CFG.defaults.hold_clawback_grace_sec)
               for r in of(Recall, decisions))
    assert "c" not in grants(decisions)
    assert not of(SwapUnload, decisions)
    a_rec = hold("a", "recalling", SEAT, recall_by=T0 + timedelta(seconds=600))
    assert not of(SwapUnload, sched(home_busy() + [a_rec, img]))        # one still on the seat
    assert [u.reason for u in of(SwapUnload, sched(home_busy() + [img]))] == ["owner_reclaim"]


def test_max_hold_drain_recalls_both_runs():
    old = cards(gpu2=CardLive("gpu2", swapped_in={SEAT}, loaded_at=T0 - timedelta(seconds=9000)))
    a, b = two_idle_runs()
    recalls = sorted((r.lease_id, r.reason) for r in of(Recall, sched(home_busy() + [a, b], crds=old)))
    assert recalls == [("a", "max_hold"), ("b", "max_hold")]


def test_urgent_pauses_only_the_most_recent_of_two_runs():
    a, b = two_idle_runs()
    home = hold("home", "granted", "agent", granted_at=T0 - timedelta(seconds=1000))
    u = hold("u", priority="urgent")
    decisions = sched([home, a, b, u])
    assert [(r.lease_id, r.reason) for r in of(Recall, decisions)] == [("b", "urgent_preempt")]
    assert not of(Grant, decisions)
