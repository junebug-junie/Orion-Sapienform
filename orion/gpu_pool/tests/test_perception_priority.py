"""``perception`` priority (Patch A0a, docs/superpowers/specs/2026-10-23-organism-body-economy-design.md).

Orion's senses (vision-council scene reads) get a priority below Juniper's turn and above Orion's
own system/background work, which the heat reflex can never shed. This patch only teaches every
loader the value; no YAML names it yet, so a pre-perception config must still load.
"""
from __future__ import annotations

import copy
from datetime import timedelta

import pytest

from orion.gpu_pool.config import DEFAULT_PATH, PRIORITIES, PoolConfig, load_pool_config
from orion.gpu_pool.shed import SHED_REASONS, SHEDDABLE, ShedBoard, ShedSignal
from orion.gpu_pool.tests.test_scheduler import CFG, T0, grants, lease, of
from orion.gpu_pool.tests.test_scheduler_shed import run
from orion.gpu_pool.scheduler import Shed, Unavailable
from orion.schemas.gpu_pool import GpuLeaseRequestV1

import yaml

RAW = yaml.safe_load(DEFAULT_PATH.read_text())


def test_priority_order_puts_perception_between_turn_and_orions_own_work():
    assert PRIORITIES == ("urgent", "interactive", "perception", "system", "background")


def test_config_without_perception_still_loads_and_ranks_it_below_interactive():
    """The live YAML on circe's lane controller predates perception; it must not become unloadable."""
    data = copy.deepcopy(RAW)
    data["priorities"] = ["urgent", "interactive", "system", "background"]
    cfg = PoolConfig.model_validate(data)
    assert cfg.priorities == list(PRIORITIES)
    assert cfg.priority_rank("interactive") < cfg.priority_rank("perception") < cfg.priority_rank("system")


def test_config_naming_perception_explicitly_loads():
    data = copy.deepcopy(RAW)
    data["priorities"] = list(PRIORITIES)
    assert PoolConfig.model_validate(data).priorities == list(PRIORITIES)


def test_route_may_declare_perception_priority():
    data = copy.deepcopy(RAW)
    data["routes"]["probe_perception"] = {"class": "metacog", "priority": "perception"}
    assert PoolConfig.model_validate(data).routes["probe_perception"].priority == "perception"


def test_unknown_priority_still_rejected():
    data = copy.deepcopy(RAW)
    data["priorities"] = ["urgent", "interactive", "system", "background", "whatever"]
    with pytest.raises(ValueError):
        PoolConfig.model_validate(data)


def test_shipped_config_loads():
    assert "perception" in load_pool_config().priorities


def test_lease_request_schema_accepts_perception():
    req = GpuLeaseRequestV1(verb="acquire", work_class="metacog", holder="vision-council", priority="perception")
    assert req.priority == "perception"


def test_no_shed_reason_can_block_perception():
    assert "perception" not in SHEDDABLE
    assert all("perception" not in spec.blocks for spec in SHED_REASONS.values())


def test_cabinet_hot_on_the_board_leaves_perception_unblocked():
    board = ShedBoard()
    board.set(ShedSignal(reason="cabinet_hot", source_id="hw", set_at=T0,
                         valid_until=T0 + timedelta(seconds=90)))
    assert "perception" not in board.view(T0, True).blocked


@pytest.mark.parametrize("kind_kw", [{}, {"retryable": True}])
def test_perception_lease_is_granted_while_heat_sheds_system_and_background(kind_kw):
    hot = {"background": "cabinet_hot", "system": "cabinet_hot"}
    p = lease("metacog", priority="perception", lease_id="p", **kind_kw)
    s = lease("metacog", priority="system", lease_id="s", **kind_kw)
    d = run([p, s], shed=hot)
    assert grants(d).get("p") == "metacog"
    assert "s" not in grants(d)
    assert not [x for x in of(Shed, d) + of(Unavailable, d) if x.lease_id == "p"]


def test_interactive_is_served_before_perception_when_one_slot_is_left():
    """Run the scheduler with one free metacog slot: the turn wins it, perception waits."""
    from orion.gpu_pool.tests.test_scheduler import RoleLive, live
    # metacog spills to fast/agent/chat seats; take those away so exactly one slot exists.
    down = {r: RoleLive(r, False, 0) for r in CFG.classes["metacog"].roles if r != "metacog"}
    one_slot = live(metacog=RoleLive("metacog", True, 1, 4096, False), **down)
    p = lease("metacog", priority="perception", lease_id="p", age=10)   # older, still loses
    i = lease("metacog", priority="interactive", lease_id="i")
    d = run([p, i], roles=one_slot, shed=None)
    assert grants(d) == {"i": "metacog"}
