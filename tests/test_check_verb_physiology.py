"""Body-economy "every verb declares its physiology" gate (scripts/check_verb_physiology.py).

Seeded fixture verb dirs (tmp_path) with hand-written YAML; the expected verdicts come from the
rule in the spec, not from the checker's output. Plus: the real repo passes, and plan_loader
exposes the block without changing the plan it builds.
"""
from __future__ import annotations

import importlib.util
import re
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("check_verb_physiology", REPO / "scripts" / "check_verb_physiology.py")
gate = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = gate
_spec.loader.exec_module(gate)

POOL = """\
roles:
  chat: {kind: llm, cards: [gpu0]}
  agent: {kind: llm, cards: [gpu1]}
  agent-gpu2: {kind: llm, cards: [gpu2]}
  diffusion: {kind: service, cards: [gpu2]}
classes:
  chat: {roles: [chat]}
  agent: {roles: [agent, agent-gpu2]}
  memory_distill: {roles: [agent]}
"""

VALID = """\
name: {name}
physiology:
  tier: 2
  family: thought.painting
  lanes: [diffusion, agent-gpu2]
  deferrable_window_s: 5400
"""


def _dirs(tmp_path: Path, verbs: dict[str, str]) -> tuple[Path, Path]:
    verbs_dir = tmp_path / "verbs"
    verbs_dir.mkdir(parents=True)
    for name, text in verbs.items():
        (verbs_dir / f"{name}.yaml").write_text(textwrap.dedent(text), encoding="utf-8")
    pool = tmp_path / "gpu_pool.yaml"
    pool.write_text(POOL, encoding="utf-8")
    return verbs_dir, pool


def _run(tmp_path: Path, verbs: dict[str, str]) -> tuple[int, list[str]]:
    verbs_dir, pool = _dirs(tmp_path, verbs)
    errors, _ = gate.check(verbs_dir, pool)
    return gate.main(["--verbs-dir", str(verbs_dir), "--pool", str(pool)]), errors


def test_valid_block_passes(tmp_path: Path) -> None:
    rc, errors = _run(tmp_path, {"paint": VALID.format(name="paint")})
    assert (rc, errors) == (0, [])


def test_active_manifest_is_not_a_verb(tmp_path: Path) -> None:
    rc, errors = _run(tmp_path, {"paint": VALID.format(name="paint"), "active": "default: {allow: [], deny: []}\n"})
    assert (rc, errors) == (0, [])


def test_missing_block_fails(tmp_path: Path) -> None:
    rc, errors = _run(tmp_path, {"paint": VALID.format(name="paint"), "bare": "name: bare\ntimeout_ms: 1000\n"})
    assert rc == 1
    assert len(errors) == 1 and "bare.yaml" in errors[0] and "missing physiology" in errors[0]


@pytest.mark.parametrize("tier", ["3", "-1", "'1'", "true", "1.0", "null"])
def test_invalid_tier_fails(tmp_path: Path, tier: str) -> None:
    text = VALID.format(name="v").replace("tier: 2", f"tier: {tier}")
    rc, errors = _run(tmp_path, {"v": text})
    assert rc == 1 and len(errors) == 1 and "tier" in errors[0]


def test_unknown_lane_fails(tmp_path: Path) -> None:
    text = VALID.format(name="v").replace("[diffusion, agent-gpu2]", "[diffusion, gpu7-turbo]")
    rc, errors = _run(tmp_path, {"v": text})
    assert rc == 1 and len(errors) == 1 and "gpu7-turbo" in errors[0]


def test_class_name_is_a_valid_lane(tmp_path: Path) -> None:
    text = VALID.format(name="v").replace("[diffusion, agent-gpu2]", "[memory_distill]")
    assert _run(tmp_path, {"v": text}) == (0, [])


def test_empty_lanes_means_no_gpu_and_passes(tmp_path: Path) -> None:
    text = VALID.format(name="v").replace("[diffusion, agent-gpu2]", "[]")
    assert _run(tmp_path, {"v": text}) == (0, [])


@pytest.mark.parametrize(
    "drop", ["  tier: 2\n", "  family: thought.painting\n", "  lanes: [diffusion, agent-gpu2]\n",
             "  deferrable_window_s: 5400\n"],
)
def test_every_field_is_required(tmp_path: Path, drop: str) -> None:
    text = VALID.format(name="v").replace(drop, "")
    rc, errors = _run(tmp_path, {"v": text})
    assert rc == 1 and len(errors) == 1


@pytest.mark.parametrize("window", ["-1", "1.5", "'60'"])
def test_bad_window_fails(tmp_path: Path, window: str) -> None:
    text = VALID.format(name="v").replace("deferrable_window_s: 5400", f"deferrable_window_s: {window}")
    rc, _ = _run(tmp_path, {"v": text})
    assert rc == 1


def test_unknown_field_fails(tmp_path: Path) -> None:
    text = VALID.format(name="v") + "  price: 3\n"
    rc, _ = _run(tmp_path, {"v": text})
    assert rc == 1


def test_tier0_must_be_turn_or_autonomic(tmp_path: Path) -> None:
    # Tier 0 never bids: an Orion-chosen family on tier 0 would dodge the market.
    text = VALID.format(name="v").replace("tier: 2", "tier: 0")
    rc, errors = _run(tmp_path, {"v": text})
    assert rc == 1 and "tier 0" in errors[0]
    for family in ("turn", "autonomic"):
        ok = text.replace("thought.painting", family).replace("deferrable_window_s: 5400", "deferrable_window_s: 0")
        assert gate.check(*_dirs(tmp_path / family, {"v": ok}))[0] == []


def test_turn_family_on_tier_2_fails(tmp_path: Path) -> None:
    text = VALID.format(name="v").replace("thought.painting", "turn")
    rc, _ = _run(tmp_path, {"v": text})
    assert rc == 1


def test_block_not_a_mapping_fails(tmp_path: Path) -> None:
    rc, errors = _run(tmp_path, {"v": "name: v\nphysiology: tier-2\n"})
    assert rc == 1 and "mapping" in errors[0]


def test_real_repo_passes() -> None:
    errors, parsed = gate.check(gate.DEFAULT_VERBS_DIR, gate.DEFAULT_POOL)
    assert errors == []
    verb_files = [p for p in gate.DEFAULT_VERBS_DIR.glob("*.yaml") if p.name != "active.yaml"]
    assert len(parsed) == len(verb_files) > 100


def _const(path: str, name: str) -> str:
    """A module-level string constant, read from source (the services are not importable here)."""
    match = re.search(rf'^{name}\s*=\s*"([^"]+)"', (REPO / path).read_text(encoding="utf-8"), re.M)
    assert match, f"{name} not found in {path}"
    return match.group(1)


def test_families_match_act_ledger_actors() -> None:
    # Patch C3 joins bids to acts by family, so where an act actor exists the verb must use it.
    # Actor names are read from the producers, so renaming one without the verb fails here.
    _, parsed = gate.check(gate.DEFAULT_VERBS_DIR, gate.DEFAULT_POOL)
    expected = {
        "daily_metacog_v1": _const("services/orion-actions/app/act_emit.py", "DAILY_METACOG_ACTOR"),
        "journal.compose": _const("services/orion-actions/app/act_emit.py", "JOURNAL_ACTOR"),
        "dream_cycle": _const("services/orion-dream/app/act_emit.py", "ACTOR"),
    }
    policy = yaml.safe_load((REPO / "config/execution_dispatch/execution_dispatch_policy.v1.yaml").read_text())
    routes = {**(policy.get("proposal_kind_to_cortex") or {}), **(policy.get("template_to_cortex") or {})}
    with_yaml = 0
    for route in routes.values():
        actor = (route.get("act") or {}).get("actor")
        verb = route.get("cortex_verb")
        if not actor or verb not in parsed:  # orion.* verbs are code-registered, no YAML to check
            continue
        with_yaml += 1
        expected[verb] = actor
    assert with_yaml >= 1, "no dispatch route with an act actor maps to a verb YAML any more"
    assert {verb: parsed[verb].family for verb in expected} == expected
    assert parsed["chat_general"].tier == 0 and parsed["chat_general"].family == "turn"
    assert parsed["stance_react"].tier == 0


def test_tier0_window_must_be_zero(tmp_path: Path) -> None:
    text = VALID.format(name="v").replace("tier: 2", "tier: 0").replace("thought.painting", "turn")
    rc, errors = _run(tmp_path, {"v": text})
    assert rc == 1 and "deferrable_window_s" in errors[0]
    ok = text.replace("deferrable_window_s: 5400", "deferrable_window_s: 0")
    assert gate.check(*_dirs(tmp_path / "ok", {"v": ok}))[0] == []


def test_plan_loader_exposes_block_without_changing_the_plan() -> None:
    from orion.cognition import plan_loader

    block = plan_loader.load_verb_physiology("dream_cycle")
    assert block is not None and (block.tier, block.family) == (1, "dream.cycle")
    plan = plan_loader.build_plan_for_verb("dream_cycle")
    dumped = plan.model_dump()
    assert "physiology" not in dumped and "physiology" not in plan.metadata
