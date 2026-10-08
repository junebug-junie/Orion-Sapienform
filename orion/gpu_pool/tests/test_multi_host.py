"""Multi-host pool config (2026-10-08): a card names its node, the pool reaches a role at its card's
node. First use: hecate's V100 serving ``agent-deep``, lendable like chat's gpu0."""
from __future__ import annotations

import copy

import pytest
import yaml

from orion.gpu_pool.config import DEFAULT_PATH, PoolConfig, launch_digest
from orion.gpu_pool.scheduler import CardLive, RoleLive, schedule
from orion.gpu_pool.tests.test_scheduler import CFG, T0, cards, grants, lease, live

RAW = yaml.safe_load(DEFAULT_PATH.read_text())


def _cfg(mutate=None) -> PoolConfig:
    data = copy.deepcopy(RAW)
    if mutate:
        mutate(data)
    return PoolConfig.model_validate(data)


def _bad(mutate, match):
    with pytest.raises(ValueError, match=match):
        _cfg(mutate)


def test_each_role_is_reached_at_its_own_nodes_address():
    assert CFG.role_host("agent-deep") == "hecate"
    assert CFG.url("agent-deep") == f"http://{CFG.hosts['hecate'].address}:{CFG.roles['agent-deep'].port}"
    assert CFG.role_host("chat") == "circe" and CFG.url("chat") == f"http://{CFG.host.address}:8011"


def test_agent_deep_is_a_lendable_resident_with_no_actuator():
    spec = CFG.roles["agent-deep"]
    assert spec.launch is None and spec.swap is None and "agent-deep" in CFG.resident_roles()
    assert CFG.lendable_cards("agent-deep") == ["hecate-gpu0"]
    assert CFG.routes["agent-deep"].work_class == "agent-deep"
    for cls in ("agent", "metacog", "fast"):
        roles = CFG.classes[cls].roles
        assert roles.index("agent-deep") < roles.index("chat"), cls


def test_index_repeats_across_nodes_but_not_within_one():
    assert CFG.cards["hecate-gpu0"].index == CFG.cards["gpu0"].index == 0
    _bad(lambda d: d["cards"]["hecate-gpu0"].pop("host"), "share index 0")


def test_rejects_unknown_card_host():
    _bad(lambda d: d["cards"]["hecate-gpu0"].update(host="nowhere"), "unknown host nowhere")


def test_rejects_the_pool_host_relisted_under_hosts():
    _bad(lambda d: d["hosts"].update(circe={"address": "x"}), "is the pool host")


def test_rejects_a_role_spanning_nodes():
    _bad(lambda d: d["roles"]["agent-deep"].update(cards=["hecate-gpu0", "gpu1"]), "cards span hosts")


def test_an_actuator_may_live_on_another_node_but_only_acts_on_its_cards():
    _cfg(lambda d: d["actuators"].update(hecate={"host": "hecate"}))

    def wrong_node(d):
        d["actuators"]["hecate"] = {"host": "hecate"}
        d["roles"]["agent-gpu2"]["launch"]["actuator"] = "hecate"
    _bad(wrong_node, "card gpu2 is on circe, its actuator hecate on hecate")


def test_existing_launch_digests_do_not_move():
    """Adding hosts/card.host must not change circe's digests: pool and controller deploy apart."""
    before = copy.deepcopy(RAW)
    before.pop("hosts")
    before["cards"].pop("hecate-gpu0")
    before["roles"].pop("agent-deep")
    before["classes"].pop("agent-deep")
    before["routes"].pop("agent-deep")
    for cls in before["classes"].values():
        if "agent-deep" in cls["roles"]:
            cls["roles"].remove("agent-deep")
    old = PoolConfig.model_validate(before)
    for role in ("agent-gpu2", "diffusion"):
        assert launch_digest(CFG, role) == launch_digest(old, role)


def _agent_elsewhere_busy():
    # Every other agent-class role is down, so only a lent card can take the lease.
    return live(agent=RoleLive("agent", False, 0, None), chat=RoleLive("chat", False, 0, None),
                **{"agent-gpu2": RoleLive("agent-gpu2", False, 0, None)})


def test_agent_work_borrows_hecate_only_while_lent():
    q = lease("agent", lease_id="q")
    assert grants(schedule(CFG, _agent_elsewhere_busy(), cards(), [q], T0)) == {}
    lent = cards(**{"hecate-gpu0": CardLive("hecate-gpu0", lent=True)})
    assert grants(schedule(CFG, _agent_elsewhere_busy(), lent, [q], T0)) == {"q": "agent-deep"}


def test_agent_deep_route_owns_its_card_unlent():
    q = lease("agent-deep", lease_id="d")
    assert grants(schedule(CFG, live(), cards(), [q], T0)) == {"d": "agent-deep"}
