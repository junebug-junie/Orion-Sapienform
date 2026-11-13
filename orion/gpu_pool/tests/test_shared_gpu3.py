"""The active GPU3 config offers one ten-slot worker to both work classes."""
from __future__ import annotations

from orion.gpu_pool.config import load_pool_config
from orion.gpu_pool.scheduler import CardLive, RoleLive, schedule
from orion.gpu_pool.tests.test_scheduler import T0, grants, lease


def test_fast_and_metacog_share_one_physical_role_and_ten_leases():
    cfg = load_pool_config()
    assert "metacog" not in cfg.roles
    assert cfg.roles["fast"].port == 8013
    assert cfg.classes["fast"].roles[0] == cfg.classes["metacog"].roles[0] == "fast"
    assert cfg.classes["operator_metacog"].roles == ["fast"]
    assert cfg.routes["metacog_perception"].priority == "perception"
    assert cfg.routes["quick_embodied"].priority == "interactive"

    roles = {name: RoleLive(name, name == "fast", 10 if name == "fast" else 0,
                            32768 if name == "fast" else 0, False) for name in cfg.roles}
    cards = {name: CardLive(name) for name in cfg.cards}
    waiting = [lease("fast", lease_id=f"q{i}") for i in range(5)] + [
        lease("metacog", lease_id=f"m{i}", priority="perception") for i in range(6)]
    placed = grants(schedule(cfg, roles, cards, waiting, T0))
    assert len(placed) == 10
    assert set(placed.values()) == {"fast"}
    assert len(set(placed) & {f"m{i}" for i in range(6)}) == 6
