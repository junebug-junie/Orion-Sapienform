"""Ladder retirement PR 4: the image prune has ONE trigger, and the disk rule's floor is the skill's own.

- The disk rule (orion/hardware_watch/rules.py) must ask at exactly the skill's own disk floor
  (services/orion-cortex-exec/app/verb_adapters.py IMAGE_PRUNE_MIN_DISK_PCT). Read with ast, because the
  two live in different services and neither imports the other.
- Every piece of the old dispatch trigger is gone: the three prune templates, their routes, the
  `maintain` fallback route, the maintenance reserved slot, the mutating-dispatch flag and the
  approved_maintenance decision.
- The two other prune skills stay (hand-run): their verbs and adapters still exist.
"""
from __future__ import annotations

import ast
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PRUNES = ("prune_build_cache", "prune_dangling_images", "prune_stopped_containers")


def _module_constant(path: Path, name: str):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {path}")


def test_the_disk_rules_floor_is_the_skills_own_floor():
    from orion.hardware_watch.rules import IMAGE_PRUNE_MIN_DISK_PCT, ImagePruneRuleConfig

    skill = _module_constant(ROOT / "services/orion-cortex-exec/app/verb_adapters.py", "IMAGE_PRUNE_MIN_DISK_PCT")
    assert IMAGE_PRUNE_MIN_DISK_PCT == skill == ImagePruneRuleConfig().min_disk_pct


def test_the_proposal_policy_has_no_prune_template():
    raw = yaml.safe_load((ROOT / "config/proposals/proposal_policy.v1.yaml").read_text())
    templates = raw["proposal_templates"]
    assert not set(PRUNES) & set(templates)
    assert not [k for k, t in templates.items() if t.get("kind") == "maintain"]
    from orion.proposals.templates import _TEMPLATE_COPY

    assert not set(PRUNES) & set(_TEMPLATE_COPY)


def test_dispatch_has_no_maintenance_route_slot_flag_or_decision():
    from orion.execution_dispatch import policy as P

    raw = yaml.safe_load((ROOT / "config/execution_dispatch/execution_dispatch_policy.v1.yaml").read_text())
    assert "maintain" not in raw["proposal_kind_to_cortex"]
    assert not set(PRUNES) & set(raw["template_to_cortex"])
    routes = list(raw["proposal_kind_to_cortex"].values()) + list(raw["template_to_cortex"].values())
    assert not [r for r in routes if r["allowed_scope"] == "maintenance_bounded" or "prune" in r["cortex_verb"]]
    assert "allow_mutating_dispatch" not in raw["mode"]
    assert "reserved_slots_by_scope" not in raw.get("limits", {})
    assert "approved_maintenance" not in raw["allowed_policy_decisions"]
    assert not hasattr(P, "MAINTENANCE_SCOPE")
    assert "allow_mutating_dispatch" not in P.DispatchModeConfigV1.model_fields
    loaded = P.load_execution_dispatch_policy(ROOT / "config/execution_dispatch/execution_dispatch_policy.v1.yaml")
    assert loaded.limits.reserved_slots_by_scope == {}


def test_the_two_hand_run_prune_skills_stay():
    """Build cache and stopped containers stay on the hand-run skill catalogue (decision 4)."""
    verbs = ROOT / "orion/cognition/verbs"
    adapters = (ROOT / "services/orion-cortex-exec/app/verb_adapters.py").read_text()
    for verb in ("skills.runtime.builder_prune.v1", "skills.runtime.docker_prune_stopped_containers.v1",
                 "skills.runtime.image_prune.v1"):
        assert (verbs / f"{verb}.yaml").exists()
        assert f'@verb("{verb}")' in adapters


def test_hardware_watch_is_a_declared_producer_of_the_exec_request_it_sends():
    raw = yaml.safe_load((ROOT / "orion/bus/channels.yaml").read_text())
    chans = {c["name"]: c for c in raw["channels"]}
    assert "orion-hardware-watch" in chans["orion:cortex:exec:request:background"]["producer_services"]
    assert "orion-hardware-watch" in chans["orion:exec:result:*"]["consumer_services"]
