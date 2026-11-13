"""Operator work classes (leased evals, Phase 1 of
docs/superpowers/specs/2026-11-12-leased-evals-and-server-refusal-design.md).

Evals take a pool hold at an ``operator*`` class, ``background`` priority. These tests pin the four
promises Juniper approved on 2026-11-12: they wait behind Orion's own background work (C2), an owner
that waits recalls them (rule 5), they never take chat's card unless it is lent (answer 4), and the
heat reflex sheds them on the same lines as Orion's background work (answer 9)."""
from __future__ import annotations

from datetime import timedelta

from orion.gpu_pool.scheduler import CardLive, Recall, Shed, _order, queue_position, schedule
from orion.gpu_pool.tests.test_scheduler import CFG, T0, cards, grants, lease, live, of
from orion.gpu_pool.value_order import is_operator_class


def op_hold(work_class="operator_agent_deep", status="queued", role=None, **kw):
    return lease(work_class, status, role, priority="background", kind="hold", **kw)


def orion_hold(status="queued", role=None, **kw):
    return lease("agent", status, role, priority="background", kind="hold", retryable=True, **kw)


def _hecate_lent_agent_busy():
    """gpu1 (agent) already carries a run's hold, gpu2's seat is not loaded, hecate is lent: both an
    Orion agent hold and an operator hold can only borrow agent-deep."""
    crds = cards(**{"hecate-gpu0": CardLive("hecate-gpu0", lent=True)})
    busy = orion_hold("granted", "agent", lease_id="busy", age=500)
    return crds, busy


def _run(leases, crds=None, roles=None, **kw):
    return schedule(CFG, roles or live(), crds or cards(), leases, T0, **kw)


def test_operator_classes_are_recognised_by_prefix_only():
    assert is_operator_class("operator") and is_operator_class("operator_chat")
    assert not is_operator_class("agent") and not is_operator_class(None) and not is_operator_class("")


def test_every_operator_route_is_background_and_no_role_owns_an_operator_class():
    ops = [name for name in CFG.classes if is_operator_class(name)]
    assert set(ops) == {"operator", "operator_chat", "operator_agent", "operator_agent_gpu2",
                        "operator_agent_deep", "operator_metacog", "operator_fast"}
    for name in ops:
        assert CFG.routes[name].priority == "background"
        assert CFG.routes[name].work_class == name
        assert not any(CFG.owns(name, r) for r in CFG.roles), name


def test_operator_hold_waits_behind_a_younger_orion_background_hold_for_the_same_role():
    crds, busy = _hecate_lent_agent_busy()
    op = op_hold(lease_id="op", age=300)          # arrived first
    mine = orion_hold(lease_id="mine", age=10)    # Orion's own work, arrived later
    assert grants(_run([busy, op, mine], crds)) == {"mine": "agent-deep"}


def test_operator_last_off_is_plain_arrival_order():
    crds, busy = _hecate_lent_agent_busy()
    op = op_hold(lease_id="op", age=300)
    mine = orion_hold(lease_id="mine", age=10)
    assert grants(_run([busy, op, mine], crds, operator_last=False)) == {"op": "agent-deep"}


def test_operator_hold_is_granted_when_no_orion_background_lease_waits():
    crds, busy = _hecate_lent_agent_busy()
    op = op_hold(lease_id="op")
    assert grants(_run([busy, op], crds)) == {"op": "agent-deep"}


def test_an_owner_waiting_recalls_a_granted_operator_hold():
    crds = cards(**{"hecate-gpu0": CardLive("hecate-gpu0", lent=True)})
    op = op_hold(status="granted", role="agent-deep", lease_id="op", granted_at=T0 - timedelta(minutes=5))
    owner = lease("agent-deep", priority="system", lease_id="owner")
    recalls = [(r.lease_id, r.reason) for r in of(Recall, _run([op, owner], crds))]
    assert ("op", "owner_waiting") in recalls


def test_operator_chat_waits_while_chats_card_is_not_lent():
    op = op_hold("operator_chat", lease_id="op")
    assert grants(_run([op])) == {}
    lent = cards(gpu0=CardLive("gpu0", lent=True))
    assert grants(_run([op], lent)) == {"op": "chat"}


def test_heat_reflex_sheds_operator_work_with_orions_background_work():
    crds, busy = _hecate_lent_agent_busy()
    op = op_hold(lease_id="op")
    out = _run([busy, op], crds, shed={"background": "cabinet_hot"})
    assert grants(out) == {}
    assert [(s.lease_id, s.reason) for s in of(Shed, out)] == [("op", "shed:cabinet_hot")]


def test_queue_order_puts_operator_background_last_but_leaves_higher_priorities_alone():
    op_bg = op_hold("operator", lease_id="op_bg", age=900)
    hub_button = lease("experiment", priority="interactive", kind="hold", operator=True, lease_id="hub", age=1)
    mine = orion_hold(lease_id="mine", age=5)
    sys_call = lease("agent", priority="system", lease_id="sys", age=1)
    ids = [l.lease_id for l in _order(CFG, [op_bg, hub_button, mine, sys_call], T0)]
    assert ids == ["hub", "sys", "mine", "op_bg"]
    assert queue_position(CFG, [op_bg, mine], "op_bg", T0) == 2
    assert queue_position(CFG, [op_bg, mine], "op_bg", T0, operator_last=False) == 1


def test_a_queued_operator_hold_is_not_orions_hunger():
    from orion.gpu_pool.shed import deferrable_queued, is_deferrable_self_work

    base = {"status": "queued", "priority": "background", "kind": "hold", "hold_lease_id": None}
    assert is_deferrable_self_work({**base, "work_class": "agent"})
    assert not is_deferrable_self_work({**base, "work_class": "operator_agent"})
    rows = [{**base, "work_class": "agent"}, {**base, "work_class": "operator"}]
    assert deferrable_queued(rows, {}) == {"agent": 1}
