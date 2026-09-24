"""One test per scheduling rule in docs/superpowers/specs/2026-09-24-gpu-pool-design.md."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion.gpu_pool.config import load_pool_config
from orion.gpu_pool.scheduler import (
    Abort, Backlog, Expire, CardLive, DeadLetter, Grant, LeaseView, Recall, Requeue, RoleLive,
    SwapLoad, SwapUnload, Unavailable, schedule,
)

T0 = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)
CFG = load_pool_config()


def live(**overrides):
    base = {
        "chat": RoleLive("chat", True, 1, 65536, True),
        "agent": RoleLive("agent", True, 1, 131072, False),
        "agent-gpu2": RoleLive("agent-gpu2", True, 1, 131072, False),
        "metacog": RoleLive("metacog", True, 4, 4096, False),
        "fast": RoleLive("fast", True, 4, 4096, False),
        "world": RoleLive("world", True, 2),
        "diffusion": RoleLive("diffusion", True, 1),
        "experiment": RoleLive("experiment", True, 1, 8192, False),
    }
    base.update(overrides)
    return base


def cards(**overrides):
    base = {c: CardLive(c) for c in CFG.cards}
    base.update(overrides)
    return base


_n = 0


def lease(work_class, status="queued", role=None, priority="system", age=0, **kw):
    global _n
    _n += 1
    return LeaseView(lease_id=kw.pop("lease_id", f"l{_n}"), work_class=work_class, priority=priority,
                     status=status, role=role, created_at=T0 - timedelta(seconds=age), **kw)


def run(leases, roles=None, crds=None, now=T0):
    return schedule(CFG, roles or live(), crds or cards(), leases, now)


def grants(decisions):
    return {d.lease_id: d.role for d in decisions if isinstance(d, Grant)}


def of(kind, decisions):
    return [d for d in decisions if isinstance(d, kind)]


def test_home_first():
    q = lease("metacog", lease_id="m")
    assert grants(run([q])) == {"m": "metacog"}


def test_metacog_uses_fast_when_its_own_role_is_full():
    held = [lease("metacog", "granted", "metacog") for _ in range(4)]
    q = lease("metacog", lease_id="m")
    assert grants(run(held + [q]))["m"] == "fast"


def test_metacog_spills_up_to_agent_when_gpu3_is_full():
    held = [lease("metacog", "granted", "metacog") for _ in range(4)]
    held += [lease("fast", "granted", "fast") for _ in range(4)]
    q = lease("metacog", lease_id="m")
    assert grants(run(held + [q]))["m"] == "agent"


def test_nothing_big_spills_down_to_gpu3():
    for cls in ("chat", "agent"):
        assert "metacog" not in CFG.classes[cls].roles and "fast" not in CFG.classes[cls].roles
    held = [lease("agent", "granted", "agent")]
    q = lease("agent", lease_id="a")
    assert "a" not in grants(run(held + [q]))


def test_context_requirement_blocks_small_slot_roles():
    # metacog/fast full? no -- they are free, but the 30k prompt does not fit their 4k slots.
    q = lease("metacog", lease_id="m", min_ctx_tokens=30000)
    assert grants(run([q]))["m"] == "agent"


def test_gpu0_refused_to_borrowers_unless_lent():
    held = [lease("agent", "granted", "agent")]
    q = lease("agent", lease_id="a")
    assert "a" not in grants(run(held + [q]))
    lent = cards(gpu0=CardLive("gpu0", lent=True))
    assert grants(run(held + [q], crds=lent))["a"] == "chat"


def test_chat_owner_recalls_lent_gpu0_borrower_then_aborts_after_grace():
    borrower = lease("agent", "granted", "chat", lease_id="b", granted_at=T0)
    owner = lease("chat", lease_id="c", priority="interactive")
    lent = cards(gpu0=CardLive("gpu0", lent=True))
    decisions = run([borrower, owner], crds=lent)
    assert "c" not in grants(decisions)
    [r] = of(Recall, decisions)
    assert r.lease_id == "b" and r.recall_by == T0 + timedelta(seconds=CFG.defaults.clawback_grace_sec)
    recalling = lease("agent", "recalling", "chat", lease_id="b", recall_by=r.recall_by)
    later = run([recalling, owner], crds=lent, now=r.recall_by)
    assert [a.lease_id for a in of(Abort, later)] == ["b"]


def test_unlending_recalls_every_borrower_on_gpu0():
    borrower = lease("metacog", "granted", "chat", lease_id="b")
    decisions = run([borrower])
    assert [(r.lease_id, r.reason) for r in of(Recall, decisions)] == [("b", "card_unlent")]


def test_borrower_never_granted_while_owner_waits_for_that_role():
    # fast is owned by metacog+fast; agent owns agent. One agent slot free, an agent owner
    # and an older metacog borrower both want it: the owner gets it.
    older_borrower = lease("metacog", lease_id="m", age=100, min_ctx_tokens=30000)
    owner = lease("agent", lease_id="a", age=1)
    got = grants(run([older_borrower, owner]))
    assert got == {"a": "agent"}


def test_priority_then_age_ordering():
    held = [lease("metacog", "granted", "metacog") for _ in range(4)]
    held += [lease("fast", "granted", "fast") for _ in range(3)]
    bg = lease("fast", lease_id="bg", priority="background", age=500)
    inter = lease("fast", lease_id="in", priority="interactive", age=1)
    got = grants(run(held + [bg, inter]))
    assert got["in"] == "fast"
    assert got.get("bg") != "fast"


def test_background_never_takes_last_slot_while_system_work_waits():
    held = [lease("metacog", "granted", "metacog") for _ in range(4)]
    held += [lease("fast", "granted", "fast") for _ in range(3)]
    held += [lease("agent", "granted", "agent")]
    bg = lease("fast", lease_id="bg", priority="background", age=50)
    sysq = lease("metacog", lease_id="sy", priority="system", age=1)
    got = grants(run(held + [bg, sysq]))
    assert got == {"sy": "fast"}


def test_queued_past_deadline_is_unavailable_not_silent():
    q = lease("chat", lease_id="c", deadline_at=T0)
    held = [lease("chat", "granted", "chat")]
    assert [(u.lease_id, u.reason) for u in of(Unavailable, run(held + [q]))] == [("c", "deadline")]


def test_backlog_when_no_role_can_serve_then_requeue_when_it_returns():
    down = live(world=RoleLive("world", False, 2))
    q = lease("world", lease_id="w", retryable=True)
    assert [b.lease_id for b in of(Backlog, run([q], roles=down))] == ["w"]
    parked = lease("world", "backlogged", lease_id="w", retryable=True)
    assert of(Requeue, run([parked], roles=down)) == []
    assert [r.lease_id for r in of(Requeue, run([parked]))] == ["w"]


def test_backlog_older_than_max_age_is_dead_lettered():
    parked = lease("world", "backlogged", lease_id="w", age=CFG.defaults.backlog_max_age_sec + 1)
    assert [d.lease_id for d in of(DeadLetter, run([parked]))] == ["w"]


def test_wait_policy_stays_queued_when_nothing_serves():
    down = live(chat=RoleLive("chat", False, 1, 65536))
    q = lease("chat", lease_id="c")
    decisions = run([q], roles=down)
    assert not of(Backlog, decisions) and not of(Unavailable, decisions) and not grants(decisions)


def test_retry_wait_requeues_when_due_and_keeps_its_place():
    early = lease("metacog", "retry_wait", lease_id="r", not_before=T0 + timedelta(seconds=5), age=100)
    assert run([early]) == []
    due = lease("metacog", "retry_wait", lease_id="r", not_before=T0, age=100)
    decisions = run([due])
    assert of(Requeue, decisions)[0].lease_id == "r" and grants(decisions)["r"] == "metacog"


def test_gpu2_swap_loads_agent_seat_only_after_wait_and_keeps_world():
    held = [lease("agent", "granted", "agent")]
    fresh = lease("agent", lease_id="a", queued_since=T0)
    assert not of(SwapLoad, run(held + [fresh]))
    waited = lease("agent", lease_id="a", queued_since=T0 - timedelta(seconds=31))
    world_busy = lease("world", "granted", "world")
    [s] = of(SwapLoad, run(held + [waited, world_busy]))
    assert s.role == "agent-gpu2"
    assert CFG.evicted_by("agent-gpu2") == ["diffusion"]


def test_gpu2_swap_blocked_while_diffusion_busy_or_wanted():
    held = [lease("agent", "granted", "agent")]
    waited = lease("agent", lease_id="a", queued_since=T0 - timedelta(seconds=60))
    diff_busy = lease("diffusion", "granted", "diffusion")
    assert not of(SwapLoad, run(held + [waited, diff_busy]))


def test_loaded_seat_serves_agent_and_evicts_diffusion():
    swapped = cards(gpu2=CardLive("gpu2", swapped_in={"agent-gpu2"}))
    held = [lease("agent", "granted", "agent")]
    q = lease("agent", lease_id="a")
    assert grants(run(held + [q], crds=swapped))["a"] == "agent-gpu2"
    d = lease("diffusion", lease_id="d")
    assert "d" not in grants(run([d], crds=swapped))
    w = lease("world", lease_id="w")
    assert grants(run([w], crds=swapped))["w"] == "world"


def test_diffusion_reclaims_gpu2():
    swapped = cards(gpu2=CardLive("gpu2", swapped_in={"agent-gpu2"}))
    on_seat = lease("agent", "granted", "agent-gpu2", lease_id="s")
    d = lease("diffusion", lease_id="d")
    decisions = run([on_seat, d], crds=swapped)
    assert [(r.lease_id, r.reason) for r in of(Recall, decisions)] == [("s", "draining")]
    assert not of(SwapUnload, decisions)
    assert [u.role for u in of(SwapUnload, run([d], crds=swapped))] == ["agent-gpu2"]


def test_cooldown_blocks_reload():
    cooling = cards(gpu2=CardLive("gpu2", cooldown_until=T0 + timedelta(seconds=100)))
    held = [lease("agent", "granted", "agent")]
    waited = lease("agent", lease_id="a", queued_since=T0 - timedelta(seconds=60))
    assert not of(SwapLoad, run(held + [waited], crds=cooling))


def test_idle_seat_unloads():
    idle = cards(gpu2=CardLive("gpu2", swapped_in={"agent-gpu2"},
                               last_active_at=T0 - timedelta(seconds=CFG.defaults.swap_idle_unload_sec)))
    assert [(u.role, u.reason) for u in of(SwapUnload, run([], crds=idle))] == [("agent-gpu2", "idle")]


def test_experiment_drains_every_card_then_loads():
    exp = lease("experiment", lease_id="x", operator=True)
    busy = [lease("chat", "granted", "chat", lease_id="c"), lease("metacog", "granted", "metacog", lease_id="m")]
    new_meta = lease("metacog", lease_id="n")
    decisions = run(busy + [exp, new_meta])
    assert {r.lease_id for r in of(Recall, decisions)} == {"c", "m"}
    assert "n" not in grants(decisions) and not of(SwapLoad, decisions)
    # metacog is background-able work: it waits for the drain instead of failing
    assert not of(Backlog, decisions)
    [s] = of(SwapLoad, run([exp]))
    assert s.role == "experiment"


def test_experiment_loaded_backlogs_background_and_chat_waits():
    all_cards = cards(**{c: CardLive(c, swapped_in={"experiment"}) for c in CFG.cards})
    holder = lease("experiment", "granted", "experiment", operator=True, granted_at=T0)
    meta = lease("metacog", lease_id="m", retryable=True)
    chat = lease("chat", lease_id="c", deadline_at=T0 + timedelta(seconds=30))
    decisions = run([holder, meta, chat], crds=all_cards)
    assert [b.lease_id for b in of(Backlog, decisions)] == ["m"]
    assert "c" not in grants(decisions) and not of(Unavailable, decisions)


def test_experiment_release_restores_residents():
    all_cards = cards(**{c: CardLive(c, swapped_in={"experiment"}) for c in CFG.cards})
    assert [u.role for u in of(SwapUnload, run([], crds=all_cards))] == ["experiment"]


def test_non_operator_cannot_take_operator_seat():
    q = lease("experiment", lease_id="x", operator=False)
    all_cards = cards(**{c: CardLive(c, swapped_in={"experiment"}) for c in CFG.cards})
    assert "x" not in grants(run([q], crds=all_cards))


def test_unhealthy_role_gets_no_grants():
    down = live(metacog=RoleLive("metacog", False, 4, 4096))
    q = lease("metacog", lease_id="m")
    assert grants(run([q], roles=down))["m"] == "fast"


@pytest.mark.parametrize("cls", sorted(CFG.classes))
def test_every_class_resolves(cls):
    assert CFG.classes[cls].roles


def test_lost_heartbeat_expires_and_frees_the_slot_this_tick():
    dead = lease("chat", "granted", "chat", lease_id="dead", expires_at=T0)
    q = lease("chat", lease_id="c", priority="interactive")
    decisions = run([dead, q])
    assert [e.lease_id for e in of(Expire, decisions)] == ["dead"]
    assert grants(decisions) == {"c": "chat"}


def test_one_waiting_owner_recalls_exactly_one_borrower_across_ticks():
    two_slot_chat = live(chat=RoleLive("chat", True, 2, 65536))
    lent = cards(gpu0=CardLive("gpu0", lent=True))
    b1 = lease("agent", "granted", "chat", lease_id="b1", granted_at=T0 - timedelta(seconds=20))
    b2 = lease("agent", "granted", "chat", lease_id="b2", granted_at=T0 - timedelta(seconds=10))
    owner = lease("chat", lease_id="o", priority="interactive")
    [r] = of(Recall, run([b1, b2, owner], roles=two_slot_chat, crds=lent))
    assert r.lease_id == "b2"
    b2r = lease("agent", "recalling", "chat", lease_id="b2", recall_by=T0 + timedelta(seconds=60))
    assert of(Recall, run([b1, b2r, owner], roles=two_slot_chat, crds=lent, now=T0 + timedelta(seconds=1))) == []


def test_retry_past_deadline_is_unavailable_not_regranted():
    due = lease("metacog", "retry_wait", lease_id="r", not_before=T0, deadline_at=T0 - timedelta(seconds=1))
    decisions = run([due])
    assert [(u.lease_id, u.reason) for u in of(Unavailable, decisions)] == [("r", "deadline")]
    assert not grants(decisions)


def test_non_retryable_backlog_class_waits_instead_of_backlogging():
    down = live(world=RoleLive("world", False, 2))
    q = lease("world", lease_id="w")
    decisions = run([q], roles=down)
    assert not of(Backlog, decisions) and not of(Unavailable, decisions)


def test_owner_reclaiming_gpu2_waits_instead_of_backlogging():
    swapped = cards(gpu2=CardLive("gpu2", swapped_in={"agent-gpu2"}))
    d = lease("diffusion", lease_id="d", retryable=True)
    decisions = run([d], crds=swapped)
    assert not of(Backlog, decisions)
    assert [u.role for u in of(SwapUnload, decisions)] == ["agent-gpu2"]
