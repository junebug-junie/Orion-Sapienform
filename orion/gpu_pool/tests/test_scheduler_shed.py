"""U4 shed (orion/gpu_pool/scheduler.py) and the shed board (orion/gpu_pool/shed.py).
Plan: docs/superpowers/plans/2026-09-29-urgent-curiosity-plan-4-5-hardware-watch-and-shedding.md."""
from __future__ import annotations

from datetime import timedelta

import pytest

from orion.gpu_pool.scheduler import Backlog, CardLive, Grant, Recall, Requeue, Shed, SwapLoad, Unavailable, schedule
from orion.gpu_pool.shed import SHED_REASONS, ShedBoard, ShedReasonSpec, ShedSignal
from orion.gpu_pool.tests.test_scheduler import CFG, T0, cards, grants, lease, live, of
from orion.gpu_pool.tests.test_scheduler_urgent import CLEAR, hold, urgent, with_cap

COOLING = {"background": "cooling_incident", "system": "cooling_incident"}


def run(leases, roles=None, crds=None, now=T0, shed=COOLING, cfg=CFG, guards=None):
    return schedule(cfg, roles or live(), crds or cards(), leases, now, guards=guards, shed=shed)


# --- U4 in the scheduler ------------------------------------------------------------------------

@pytest.mark.parametrize("priority", ["background", "system"])
def test_shed_priorities_get_no_new_grant_and_are_reported(priority):
    q = lease("metacog", priority=priority, lease_id="q")
    d = run([q])
    assert grants(d) == {}
    assert of(Shed, d) == [Shed("q", "shed:cooling_incident")]
    assert grants(run([q], shed=None)) == {"q": "metacog"}


@pytest.mark.parametrize("priority", ["interactive", "urgent"])
def test_interactive_and_urgent_are_never_shed(priority):
    q = lease("agent", priority=priority, lease_id="q")
    d = run([q], shed=COOLING)
    assert grants(d) == {"q": "agent"} and of(Shed, d) == []


def test_urgent_is_not_shed_even_under_the_urgent_rollback():
    """urgent_max_concurrent=0 rewrites urgent to background; shed is decided on the original."""
    q = urgent(lease_id="u")
    d = run([q], cfg=with_cap(0))
    assert grants(d) == {"u": "agent"} and of(Shed, d) == []


def test_running_work_is_not_recalled_and_its_calls_still_run():
    h = hold("granted", "agent", lease_id="h", granted_at=T0 - timedelta(seconds=60))
    child = lease("agent", priority="background", lease_id="c", hold_lease_id="h")
    running = lease("metacog", "granted", "metacog", priority="system", lease_id="r")
    d = run([h, child, running])
    assert of(Recall, d) == []
    assert grants(d) == {"c": "agent"}
    assert of(Shed, d) == []


def test_shed_lease_keeps_its_place_and_is_granted_first_when_cleared():
    old = lease("metacog", priority="background", lease_id="old", age=100)
    new = lease("metacog", priority="background", lease_id="new", age=1)
    full = [lease("metacog", "granted", "metacog", priority="interactive") for _ in range(3)] + \
           [lease("fast", "granted", "fast", priority="interactive") for _ in range(4)] + \
           [lease("agent", "granted", "agent", priority="interactive")] + \
           [lease("chat", "granted", "chat", priority="interactive")]
    assert [s.lease_id for s in of(Shed, run(full + [old, new]))] == ["old", "new"]
    assert grants(run(full + [old, new], shed=None)) == {"old": "metacog"}


def test_shed_owner_does_not_recall_a_borrower():
    """A shed lease is not demand: an interactive borrower on agent is not recalled for it."""
    borrower = lease("metacog", "granted", "agent", priority="interactive", lease_id="b")
    owner = hold(lease_id="o")   # background agent hold, owner of agent
    d = run([borrower, owner])
    assert of(Recall, d) == []
    assert of(Recall, run([borrower, owner], shed=None))


def test_shed_demand_does_not_load_a_swap_seat():
    """Background agent work waiting past after_wait_sec would load agent-gpu2; shed, it does not."""
    busy = lease("agent", "granted", "agent", priority="interactive", lease_id="busy")
    waiting = lease("agent", priority="background", lease_id="w", age=5000, queued_since=T0 - timedelta(seconds=5000))
    roles = live(**{"agent-gpu2": live()["agent-gpu2"].__class__("agent-gpu2", False, 0)})
    crds = cards(gpu2=CardLive("gpu2", swapped_in={"diffusion"}))
    assert not of(SwapLoad, run([busy, waiting], roles=roles, crds=crds, guards=CLEAR))
    assert of(SwapLoad, run([busy, waiting], roles=roles, crds=crds, guards=CLEAR, shed=None)) == \
        [SwapLoad("agent-gpu2", "demand")]


def test_shed_lease_is_not_backlogged_or_failed_for_want_of_a_role():
    q = lease("agent", priority="background", lease_id="q", retryable=True)
    roles = live(agent=live()["agent"].__class__("agent", False, 0))
    d = run([q], roles=roles)
    assert of(Backlog, d) == [] and of(Unavailable, d) == []


def test_deadline_still_applies_to_a_shed_lease():
    q = lease("metacog", priority="background", lease_id="q", deadline_at=T0 - timedelta(seconds=1))
    assert of(Unavailable, run([q])) == [Unavailable("q", "deadline")]


def test_retrying_shed_lease_is_requeued_but_not_granted():
    q = lease("metacog", "retry_wait", priority="system", lease_id="q", not_before=T0 - timedelta(seconds=1))
    d = run([q])
    assert of(Requeue, d) == [Requeue("q", "retry_due")]
    assert grants(d) == {} and of(Shed, d) == [Shed("q", "shed:cooling_incident")]


def test_only_background_shed_leaves_system_alone():
    bg = lease("metacog", priority="background", lease_id="bg")
    sys = lease("metacog", priority="system", lease_id="sys")
    d = run([bg, sys], shed={"background": "orion_x"})
    assert grants(d) == {"sys": "metacog"}
    assert of(Shed, d) == [Shed("bg", "shed:orion_x")]
    assert not any(isinstance(x, Grant) and x.lease_id == "bg" for x in d)


# --- the board -------------------------------------------------------------------------------

def sig(reason="cooling_incident", source="inc1", valid=300, **detail):
    return ShedSignal(reason, source, T0, T0 + timedelta(seconds=valid), detail)


def test_board_blocks_background_and_system_for_cooling():
    b = ShedBoard()
    assert b.set(sig())
    v = b.view(T0, enabled=True)
    assert v.blocked == COOLING and v.active_reason == "cooling_incident"
    assert v.reasons[0]["sources"][0]["source_id"] == "inc1"


def test_board_kill_switch_blocks_nothing_but_still_shows_the_signal():
    b = ShedBoard()
    b.set(sig())
    v = b.view(T0, enabled=False)
    assert v.blocked == {} and v.active_reason is None
    assert v.reasons[0]["active"] and not v.reasons[0]["effective"]


def test_board_signal_lapses_at_valid_until():
    b = ShedBoard()
    b.set(sig(valid=60))
    assert b.view(T0 + timedelta(seconds=59), True).blocked
    assert b.view(T0 + timedelta(seconds=60), True).blocked == {}
    assert [s.source_id for s in b.prune(T0 + timedelta(seconds=60))] == ["inc1"]


def test_board_clear_removes_only_that_source():
    b = ShedBoard()
    b.set(sig(source="a"))
    b.set(sig(source="b"))
    b.clear("cooling_incident", "a")
    assert b.view(T0, True).blocked == COOLING
    b.clear("cooling_incident", "b")
    assert b.view(T0, True).blocked == {}


def test_board_refuses_unknown_reasons():
    assert not ShedBoard().set(sig(reason="made_up"))


def test_precedence_attributes_each_priority_to_the_winning_reason():
    """The extension point: a lower-precedence reason adds blocks, never lifts or renames them."""
    reasons = dict(SHED_REASONS)
    reasons["orion_shed_background"] = ShedReasonSpec("orion_shed_background", 10, ("background",), "later PR")
    b = ShedBoard(reasons)
    b.set(sig(reason="orion_shed_background", source="orion"))
    v = b.view(T0, True)
    assert v.blocked == {"background": "orion_shed_background"} and v.active_reason == "orion_shed_background"
    b.set(sig())
    v = b.view(T0, True)
    assert v.blocked == COOLING and v.active_reason == "cooling_incident"
    b.clear("cooling_incident", "inc1")
    assert b.view(T0, True).blocked == {"background": "orion_shed_background"}


@pytest.mark.parametrize("blocks", [("interactive",), ("urgent",), ("background", "interactive"), ()])
def test_a_reason_may_never_shed_interactive_or_urgent(blocks):
    with pytest.raises(ValueError):
        ShedBoard({"bad": ShedReasonSpec("bad", 5, blocks, "x")})
