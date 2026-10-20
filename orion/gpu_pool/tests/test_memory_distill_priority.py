"""memory_distill gets the system priority its route promises on gpu1 (2026-10-20).

config/gpu_pool.yaml routes memory_distill at system priority "ahead of curiosity's background
holds", but until 2026-10-20 the ``agent`` role (gpu1, memory_distill's only role) listed only
``agent`` as its owner. Scheduler rule 5 gives a role's owners every grant before any borrower,
whatever the priority, and recalls a borrower when an owner waits. Live:

* 2026-10-18 11:39:25 a distill hold got gpu1; 30 s later it was recalled ``owner_waiting``;
* 2026-10-20 two distill holds (queued 04:52 and 06:05, system) waited all day while queued
  ``agent`` background reading holds got gpu1 one after another in every cool window.

These tests replay those two states through the shipped scheduler and the shipped config.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.gpu_pool.scheduler import CardLive, Grant, Recall, RoleLive, schedule
from orion.gpu_pool.tests.test_scheduler import CFG, cards, lease, live, of

NOW = datetime(2026, 10, 20, 14, 41, 43, tzinfo=timezone.utc)   # an agent background hold got gpu1 here


def _old_cfg():
    """The config as it was: ``agent`` is the gpu1 role's only owner."""
    spec = CFG.roles["agent"]
    return CFG.model_copy(update={"roles": {**CFG.roles, "agent": spec.model_copy(update={"owner": ["agent"]})}})


def _ago(ts: str) -> float:
    return (NOW - datetime.fromisoformat(ts)).total_seconds()


def _distill(lease_id, created, **kw):
    return lease("memory_distill", priority="system", kind="hold", retryable=True, lease_id=lease_id,
                 age=_ago(created), **kw)


def _reading(lease_id, created, **kw):
    return lease("agent", priority="background", kind="hold", retryable=True, lease_id=lease_id,
                 age=_ago(created), **kw)


def _run(cfg, leases, now=NOW, gpu2_loaded=True):
    # gpu1's 27B: one slot. On 10-20 the gpu2 seat was loaded (agent work spread onto it live that
    # day); on 10-18 it was not (no agent-gpu2 grant that day), so gpu1 was the only agent card.
    roles = live(agent=RoleLive("agent", True, 1, 131072, False),
                 **{"agent-gpu2": RoleLive("agent-gpu2", True, 2, 131072, False)})
    crds = cards(gpu2=CardLive("gpu2", swapped_in={"agent-gpu2"} if gpu2_loaded else set()))
    return schedule(cfg, roles, crds, list(leases), now, guards={"thermal": None})


def _grants(decisions):
    return {d.lease_id: d.role for d in decisions if isinstance(d, Grant)}


def _queue_2026_10_20():
    """The live queue at 14:41 on 10-20 (gpu_pool_leases), shed lifted (a cool window)."""
    return [
        _distill("d1", "2026-10-20T04:52:50+00:00"),
        _distill("d2", "2026-10-20T06:05:59+00:00"),
        _reading("r1", "2026-10-20T13:58:51+00:00"),
        _reading("r2", "2026-10-20T13:59:06+00:00"),
        _reading("r3", "2026-10-20T14:18:42+00:00"),
    ]


def test_live_config_makes_memory_distill_an_owner_of_gpu1():
    assert CFG.owns("memory_distill", "agent")
    assert CFG.owns("agent", "agent")
    assert list(CFG.classes["memory_distill"].roles) == ["agent"]   # still gpu1 only
    # Not an owner anywhere else: no new claim on gpu2's seat or the lent cards.
    assert [r for r in CFG.roles if CFG.owns("memory_distill", r)] == ["agent"]


def test_replay_2026_10_20_old_config_starves_the_distill_holds():
    """Reproduces the bug: gpu1 goes to a background reading hold, never the system distill hold."""
    got = _grants(_run(_old_cfg(), _queue_2026_10_20()))
    assert got.get("r1") == "agent"
    assert "d1" not in got and "d2" not in got


def test_replay_2026_10_20_distill_gets_gpu1_first():
    got = _grants(_run(CFG, _queue_2026_10_20()))
    assert got.get("d1") == "agent"            # the oldest system hold takes gpu1
    assert "d2" not in got                     # one hold per role (H1)
    assert got.get("r1") != "agent"            # the background reading waits, or runs on gpu2


def test_replay_2026_10_18_distill_hold_is_not_recalled_for_an_agent_owner():
    """10-18 11:39:55: the granted distill hold was recalled owner_waiting 30 s after its grant."""
    now = datetime(2026, 10, 18, 11, 39, 55, tzinfo=timezone.utc)
    running = lease("memory_distill", "granted", "agent", priority="system", kind="hold", retryable=True,
                    lease_id="d", age=(now - datetime(2026, 10, 18, 11, 39, 25, tzinfo=timezone.utc)).total_seconds(),
                    granted_at=datetime(2026, 10, 18, 11, 39, 25, tzinfo=timezone.utc))
    waiting = lease("agent", priority="background", kind="hold", retryable=True, lease_id="r", age=5)
    old = [(r.lease_id, r.reason) for r in of(Recall, _run(_old_cfg(), [running, waiting], now=now, gpu2_loaded=False))]
    new = [r.lease_id for r in of(Recall, _run(CFG, [running, waiting], now=now, gpu2_loaded=False))]
    assert old == [("d", "owner_waiting")]          # the bug: a background owner recalls a system run
    assert new == []


def test_an_agent_system_call_still_beats_a_younger_distill_hold():
    """Both are owners now: queue order (priority, then age) decides, as rule 6 says."""
    call = lease("agent", priority="system", lease_id="a", age=60)
    distill = _distill("d", "2026-10-20T14:41:00+00:00")
    got = _grants(_run(CFG, [call, distill]))
    assert got.get("a") == "agent"
