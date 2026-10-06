"""U4 shed (orion/gpu_pool/scheduler.py) and the shed board (orion/gpu_pool/shed.py).
Plan: docs/superpowers/plans/2026-09-29-urgent-curiosity-plan-4-5-hardware-watch-and-shedding.md.
D3 one-shot refusal: docs/superpowers/specs/2026-10-06-thermal-controller-redesign-design.md."""
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
    """A lease someone comes back for (retryable backlog class) waits, reported as Shed."""
    q = lease("metacog", priority=priority, lease_id="q", retryable=True)
    d = run([q])
    assert grants(d) == {}
    assert of(Shed, d) == [Shed("q", "shed:cooling_incident")]
    assert of(Unavailable, d) == []
    assert grants(run([q], shed=None)) == {"q": "metacog"}


# --- D3: one-shot requests are refused at once ------------------------------------------------

@pytest.mark.parametrize("priority", ["background", "system"])
@pytest.mark.parametrize("work_class", ["fast", "metacog", "agent", "chat"])
def test_one_shot_request_is_refused_at_once(work_class, priority):
    """kind=request, no hold, not retryable: the caller's fallback runs now, not at its deadline.
    metacog/agent are backlog classes, but a non-retryable backlog lease is a "wait" lease (rule 10)
    -- that is every gateway call (orion-mind, memory annotation)."""
    q = lease(work_class, priority=priority, lease_id="q", deadline_at=T0 + timedelta(seconds=600))
    d = run([q])
    assert of(Unavailable, d) == [Unavailable("q", "shed:cooling_incident")]
    assert grants(d) == {} and of(Shed, d) == []


def test_one_shot_refusal_names_the_winning_reason():
    q = lease("fast", priority="background", lease_id="q")
    assert of(Unavailable, run([q], shed={"background": "orion_x"})) == [Unavailable("q", "shed:orion_x")]


def test_durable_hold_stays_queued_under_shed():
    h = hold(lease_id="h")
    d = run([h])
    assert of(Shed, d) == [Shed("h", "shed:cooling_incident")]
    assert of(Unavailable, d) == [] and grants(d) == {}


def test_backlog_lease_stays_backlogged_under_shed():
    b = lease("agent", "backlogged", priority="background", lease_id="b", retryable=True)
    d = run([b])
    # It is servable again, so it rejoins the queue in place -- and is held there, not refused.
    assert of(Requeue, d) == [Requeue("b", "role_available")]
    assert of(Shed, d) == [Shed("b", "shed:cooling_incident")]
    assert of(Unavailable, d) == [] and grants(d) == {}
    # Nothing can serve it: it simply stays backlogged (no decision at all).
    roles = live(**{r: live()[r].__class__(r, False, 0) for r in CFG.classes["agent"].roles})
    d = run([b], roles=roles)
    assert of(Unavailable, d) == [] and of(Requeue, d) == [] and grants(d) == {}


def test_granted_holds_child_is_still_granted_under_shed():
    h = hold("granted", "agent", lease_id="h", granted_at=T0 - timedelta(seconds=60))
    child = lease("agent", priority="background", lease_id="c", hold_lease_id="h")
    d = run([h, child])
    assert grants(d) == {"c": "agent"}
    assert of(Unavailable, d) == [] and of(Shed, d) == []


@pytest.mark.parametrize("priority", ["interactive", "urgent"])
def test_one_shot_interactive_and_urgent_are_never_refused(priority):
    q = lease("fast", priority=priority, lease_id="q")
    d = run([q])
    assert grants(d) == {"q": "fast"} and of(Unavailable, d) == [] and of(Shed, d) == []


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
    """Waiting leases (retryable backlog) keep their order; one-shots are refused instead (D3)."""
    old = lease("metacog", priority="background", lease_id="old", age=100, retryable=True)
    new = lease("metacog", priority="background", lease_id="new", age=1, retryable=True)
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
    waiting = lease("agent", priority="background", lease_id="w", age=5000, queued_since=T0 - timedelta(seconds=5000),
                    retryable=True)   # a lease that waits under shed (a one-shot would be refused, D3)
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
    """retry_wait is not refused in the tick it is Requeue'd: one transition per lease per tick (D3)."""
    q = lease("metacog", "retry_wait", priority="system", lease_id="q", not_before=T0 - timedelta(seconds=1),
              retryable=True)
    d = run([q])
    assert of(Requeue, d) == [Requeue("q", "retry_due")]
    assert grants(d) == {} and of(Shed, d) == [Shed("q", "shed:cooling_incident")]
    assert of(Unavailable, d) == []


def test_only_background_shed_leaves_system_alone():
    bg = lease("metacog", priority="background", lease_id="bg", retryable=True)
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
    assert _row(v, "cooling_incident")["sources"][0]["source_id"] == "inc1"


def _row(view, name):
    return next(r for r in view.reasons if r["name"] == name)


def test_board_kill_switch_blocks_nothing_but_still_shows_the_signal():
    b = ShedBoard()
    b.set(sig())
    v = b.view(T0, enabled=False)
    assert v.blocked == {} and v.active_reason is None
    assert _row(v, "cooling_incident")["active"] and not _row(v, "cooling_incident")["effective"]


# --- thermal controller v2 (D2): the reflex reasons -----------------------------------------

def test_cabinet_hot_blocks_background_and_system_cabinet_unknown_background_only():
    b = ShedBoard()
    b.set(sig(reason="cabinet_unknown", source="hw"))
    assert b.view(T0, True).blocked == {"background": "cabinet_unknown"}
    b.set(sig(reason="cabinet_hot", source="hw"))
    assert b.view(T0, True).blocked == {"background": "cabinet_hot", "system": "cabinet_hot"}


def test_reflex_signal_lapses_after_valid_until_when_the_watcher_stops_sending():
    """D2: re-sent per tick with valid_until = now + 3 ticks; a dead watcher's shed lapses (fail-open)."""
    b = ShedBoard()
    b.set(sig(reason="cabinet_hot", source="hw", valid=90))
    assert b.view(T0 + timedelta(seconds=89), True).blocked
    assert b.view(T0 + timedelta(seconds=90), True).blocked == {}


def test_reflex_reasons_are_precedence_zero_and_never_shed_interactive():
    for name in ("cabinet_hot", "cabinet_unknown"):
        spec = SHED_REASONS[name]
        assert spec.precedence == 0 and "interactive" not in spec.blocks and "urgent" not in spec.blocks


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
