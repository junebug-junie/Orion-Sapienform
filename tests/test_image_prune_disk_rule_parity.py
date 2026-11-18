"""Ladder retirement PR 4: the image prune has ONE trigger, and the disk rule's floor is the skill's own.

- The disk rule (orion/hardware_watch/rules.py) must ask at exactly the skill's own disk floor
  (services/orion-cortex-exec/app/verb_adapters.py IMAGE_PRUNE_MIN_DISK_PCT). Read with ast, because the
  two live in different services and neither imports the other.
- The old dispatch trigger is gone: since PR 6 (2026-11-07) the whole proposal/policy/dispatch ladder is,
  so the only other sender of the prune verb cannot come back.
- The two other prune skills stay (hand-run): their verbs and adapters still exist.
"""
from __future__ import annotations

import ast
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


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


def test_the_dispatch_ladder_that_also_sent_prunes_is_gone():
    for path in ("config/proposals/proposal_policy.v1.yaml", "config/execution_dispatch",
                 "orion/execution_dispatch", "orion/proposals", "services/orion-execution-dispatch-runtime"):
        assert not (ROOT / path).exists(), path


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
