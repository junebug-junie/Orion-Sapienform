"""Warm standby (Juniper 2026-10-14: "gpu2 should not sit idle while agent work exists"):
the agent-gpu2 seat (``swap.standby: warm`` in config/gpu_pool.yaml) is loaded whenever
diffusion -- what it evicts -- has been quiet for standby_after_idle_sec, is never unloaded for
being idle or on the max_hold_sec clock, and is still drained and unloaded by a diffusion lease
exactly as before. Guards, cooldown, min_residency and an active shed block the warm load; a
failed warm load backs off standby_retry_after_fail_sec."""
from __future__ import annotations

from datetime import timedelta

from orion.gpu_pool.scheduler import (
    CardLive, Grant, Recall, RoleLive, SwapBlocked, SwapLoad, SwapUnload, schedule,
)
from orion.gpu_pool.tests.test_scheduler import CFG, COLD, T0, cards, grants, lease, live, of

SEAT = "agent-gpu2"
SPEC = CFG.roles[SEAT].swap
QUIET = SPEC.standby_after_idle_sec
CLEAR = {"thermal": None}


def gpu2(*, diffusion_quiet=QUIET + 1, loaded=False, **kw):
    """gpu2 with diffusion last active ``diffusion_quiet`` seconds ago (None: clock not started)."""
    active = {} if diffusion_quiet is None else {"diffusion": T0 - timedelta(seconds=diffusion_quiet),
                                                 "world": T0}   # world is not evicted: never matters
    return cards(gpu2=CardLive("gpu2", swapped_in={SEAT} if loaded else set(), role_active_at=active, **kw))


# The live seat: Bonsai, 2 slots (discovery reads them from /props; max_holds is clamped to them).
ROLES = live(**{SEAT: RoleLive(SEAT, True, 2, 131072, False)})


def sched(leases=(), crds=None, cfg=CFG, now=T0, guards=CLEAR, shed=None):
    return schedule(cfg, ROLES, crds or gpu2(), list(leases), now, guards=guards, shed=shed)


def diffusion(status="queued", role=None, **kw):
    return lease("diffusion", status, role, priority="background", kind="hold", retryable=True, **kw)


def reading_hold(**kw):
    return lease("agent", "queued", None, priority="background", kind="hold", retryable=True, **kw)


def loads(decisions):
    return [(s.role, s.reason) for s in of(SwapLoad, decisions)]


def blocked(decisions):
    return [(b.role, b.reason) for b in of(SwapBlocked, decisions)]


def test_live_config_ships_agent_gpu2_warm():
    assert SPEC.standby == "warm" and SPEC.evicts == ["diffusion"]
    assert COLD.roles[SEAT].swap.standby == "off"


def test_warm_loads_with_no_demand_once_diffusion_is_quiet():
    assert loads(sched()) == [(SEAT, "standby")]


def test_off_never_loads_without_demand():
    assert loads(sched(cfg=COLD)) == []


def test_hysteresis_waits_out_recent_diffusion_silently():
    d = sched(crds=gpu2(diffusion_quiet=QUIET - 60))
    assert loads(d) == [] and blocked(d) == []


def test_clock_not_started_fails_closed():
    assert loads(sched(crds=gpu2(diffusion_quiet=None))) == []


def test_diffusion_wanted_or_busy_blocks_the_warm_load():
    assert loads(sched([diffusion()])) == []
    assert loads(sched([diffusion("granted", "diffusion")])) == []


def test_shed_blocks_warm_up_and_says_so():
    d = sched(shed={"background": "cabinet_hot"})
    assert loads(d) == [] and blocked(d) == [(SEAT, "shed:cabinet_hot")]


def test_thermal_guard_cooldown_and_min_residency_block_warm_up():
    assert blocked(sched(guards={"thermal": "hot"})) == [(SEAT, "guard:thermal")]
    assert blocked(sched(crds=gpu2(cooldown_until=T0 + timedelta(seconds=60)))) == [(SEAT, "cooldown")]
    assert blocked(sched(crds=gpu2(residency_until=T0 + timedelta(seconds=60)))) == [(SEAT, "min_residency")]


def test_a_failed_warm_load_backs_off_longer_than_the_swap_cooldown():
    failed = {"action": "load", "reason": "standby", "outcome": "failed_restored",
              "finished_at": (T0 - timedelta(seconds=SPEC.standby_retry_after_fail_sec - 60)).isoformat()}
    assert blocked(sched(crds=gpu2(swap_action=failed))) == [(SEAT, "standby_backoff")]
    old = {**failed, "finished_at": (T0 - timedelta(seconds=SPEC.standby_retry_after_fail_sec + 1)).isoformat()}
    assert loads(sched(crds=gpu2(swap_action=old))) == [(SEAT, "standby")]
    demand = {**failed, "reason": "demand"}   # a failed DEMAND load backs the warm retry off too
    assert blocked(sched(crds=gpu2(swap_action=demand))) == [(SEAT, "standby_backoff")]
    unload = {**failed, "action": "unload"}   # a failed unload is not a reason to stay cold
    assert loads(sched(crds=gpu2(swap_action=unload))) == [(SEAT, "standby")]


def test_warm_seat_stays_loaded_idle_and_past_max_hold():
    idle = gpu2(loaded=True, last_active_at=T0 - timedelta(hours=6), loaded_at=T0 - timedelta(hours=6))
    d = sched(crds=idle)
    assert of(SwapUnload, d) == [] and of(Recall, d) == []
    cold = sched(crds=idle, cfg=COLD)   # control: the same card unloads with standby off
    assert [u.reason for u in of(SwapUnload, cold)] in (["idle"], ["max_hold"])


def test_diffusion_still_evicts_the_warm_seat():
    warm = gpu2(loaded=True)
    on_seat = lease("agent", "granted", SEAT, priority="background", kind="hold", retryable=True, lease_id="s")
    img = diffusion(lease_id="img")
    d = sched([on_seat, img], crds=warm)
    assert [(r.lease_id, r.reason) for r in of(Recall, d)] == [("s", "draining")]
    assert [(u.role, u.reason) for u in of(SwapUnload, sched([img], crds=warm))] == [(SEAT, "owner_reclaim")]


def test_no_flapping_through_a_painting_burst():
    """Seat unloads for a painting -> min_residency -> diffusion keeps painting -> it only re-warms
    once diffusion has been quiet for standby_after_idle_sec, never in between."""
    res = CFG.defaults.swap_min_residency_sec
    for quiet_for, expect in [(0, []), (res / 2, []), (QUIET - 1, []), (QUIET + 1, [(SEAT, "standby")])]:
        now = T0 + timedelta(seconds=res + quiet_for)
        crds = cards(gpu2=CardLive("gpu2", residency_until=T0 + timedelta(seconds=res),
                                   role_active_at={"diffusion": T0 + timedelta(seconds=res)}))
        assert loads(sched(crds=crds, now=now)) == expect, quiet_for


def test_warm_seat_spreads_a_short_reading_queue_over_gpu1_and_gpu2():
    """Three reading holds arriving together: one on gpu1 (agent holds one), two on the warm
    gpu2 seat (max_holds 2) -- no after_wait_sec wait, no queue needed."""
    hs = [reading_hold(lease_id=f"r{i}", queued_since=T0) for i in range(3)]
    g = grants(sched(hs, crds=gpu2(loaded=True)))
    assert sorted(g.values()) == ["agent", SEAT, SEAT]
    # Cold, the same queue gets one read on gpu1 and the rest wait (the 2026-10-14 live state).
    assert sorted(grants(sched(hs, crds=gpu2(), cfg=COLD)).values()) == ["agent"]
    assert not any(isinstance(x, Grant) and x.role == SEAT for x in sched(hs, crds=gpu2(), cfg=COLD))
