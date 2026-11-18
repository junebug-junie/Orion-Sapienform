"""The ``starved`` pool priority (2026-11-03, Orion Prime reliability).

Orion's own work whose previous try lost its whole seat wait to the queue. It ranks below
Juniper's turn and Orion's senses, above ordinary system work, and heat sheds it exactly like
system work. Today only Orion Prime's ``prime_starved`` route uses it.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
import yaml

from orion.gpu_pool.config import PRIORITIES, PoolConfig, load_pool_config
from orion.gpu_pool.scheduler import PREEMPTIBLE, schedule
from orion.gpu_pool.shed import REFLEX_REASONS, SHED_REASONS, SHEDDABLE, ShedBoard, ShedSignal
from orion.gpu_pool.tests.test_route_priorities_explicit import POOL_YAML
from orion.gpu_pool.tests.test_scheduler import CFG, T0, cards, grants, lease, live

NOW = datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)


def test_starved_sits_below_perception_and_above_system():
    order = CFG.priorities
    assert order.index("interactive") < order.index("perception") < order.index("starved") < order.index("system")
    assert set(PRIORITIES) == set(order)


def test_a_config_without_starved_still_loads_with_it_just_above_system():
    """The lane controller and mesh-guardian read the live checkout's YAML with their image's code:
    new code must keep loading an old YAML (perception's precedent)."""
    doc = yaml.safe_load(POOL_YAML.read_text(encoding="utf-8"))
    doc["priorities"] = ["urgent", "interactive", "perception", "system", "background"]
    doc["routes"] = {k: v for k, v in doc["routes"].items() if v.get("priority") != "starved"}
    cfg = PoolConfig.model_validate(doc)
    assert cfg.priorities == list(PRIORITIES)
    doc["priorities"] = ["urgent", "interactive", "system", "background"]   # pre-perception too
    assert PoolConfig.model_validate(doc).priorities == list(PRIORITIES)


def test_a_starved_lease_is_granted_before_older_system_work_but_after_perception():
    sys_old = lease("chat", priority="system", lease_id="sys", age=300)
    starved = lease("chat", priority="starved", lease_id="st", age=1)
    assert grants(schedule(CFG, live(), cards(), [sys_old, starved], T0)) == {"st": "chat"}
    seen = lease("chat", priority="perception", lease_id="pe", age=0)
    juniper = lease("chat", priority="interactive", lease_id="ju", age=0)
    assert grants(schedule(CFG, live(), cards(), [starved, seen], T0)) == {"pe": "chat"}
    assert grants(schedule(CFG, live(), cards(), [starved, juniper], T0)) == {"ju": "chat"}


def test_heat_sheds_starved_exactly_where_it_sheds_system():
    assert "starved" in SHEDDABLE and "starved" in PREEMPTIBLE
    for name, spec in SHED_REASONS.items():
        assert ("starved" in spec.blocks) == ("system" in spec.blocks), name
    # The 34 C reflex (cabinet_hot) is one of them.
    assert "starved" in SHED_REASONS["cabinet_hot"].blocks and "cabinet_hot" in REFLEX_REASONS
    board = ShedBoard()
    board.set(ShedSignal("cabinet_hot", "hw", NOW, NOW + timedelta(minutes=5)))
    view = board.view(NOW, enabled=True)
    assert view.blocked["starved"] == "cabinet_hot"
    for never in ("urgent", "interactive", "perception"):
        assert never not in view.blocked


def test_a_starved_lease_is_refused_under_the_heat_reflex():
    q = lease("chat", priority="starved", lease_id="q", deadline_at=T0 + timedelta(seconds=600))
    d = schedule(CFG, live(), cards(), [q], T0, shed={"starved": "cabinet_hot", "system": "cabinet_hot"})
    assert grants(d) == {}


@pytest.mark.parametrize("route, priority", [("prime", "system"), ("prime_starved", "starved")])
def test_prime_routes_are_the_chat_class(route, priority):
    cfg = load_pool_config()
    assert cfg.routes[route].work_class == cfg.routes["chat"].work_class == "chat"
    assert cfg.routes[route].priority == priority
    # Only Prime's escalation route uses the level.
    assert [r for r, s in cfg.routes.items() if s.priority == "starved"] == ["prime_starved"]
