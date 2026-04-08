from __future__ import annotations

from pathlib import Path

from app.actions_skill_registry import ActionsSkillRegistry
from app.capability_bridge import normalize_capability_observation, resolve_capability_decision


def test_capability_selector_resolves_semantic_verb_to_system_inspection_skill():
    registry = ActionsSkillRegistry(verbs_dir=Path("/workspace/Orion-Sapienform/orion/cognition/verbs"))
    decision = resolve_capability_decision(
        verb="assess_runtime_state",
        preferred_skill_families=["system_inspection"],
        registry=registry,
    )
    assert decision.needs_skill is True
    assert decision.skill_family == "system_inspection"
    assert decision.selected_skill in decision.candidate_skills
    assert decision.selected_skill is not None


def test_normalized_capability_observation_shape():
    registry = ActionsSkillRegistry(verbs_dir=Path("/workspace/Orion-Sapienform/orion/cognition/verbs"))
    decision = resolve_capability_decision(
        verb="assess_runtime_state",
        preferred_skill_families=["system_inspection"],
        registry=registry,
    )
    observation = normalize_capability_observation(
        decision=decision,
        execution_summary="Executed skill successfully.",
        raw_payload={"status": "success"},
    )
    assert observation["selected_verb"] == "assess_runtime_state"
    assert observation["selected_skill_family"] == "system_inspection"
    assert observation["selected_skill"] == decision.selected_skill
    assert "execution_summary" in observation
    assert "raw_payload_ref" in observation


def test_housekeep_runtime_selects_mutating_skill_with_execute_opt_in_policy():
    registry = ActionsSkillRegistry(verbs_dir=Path("/workspace/Orion-Sapienform/orion/cognition/verbs"))
    decision = resolve_capability_decision(
        verb="housekeep_runtime",
        preferred_skill_families=["runtime_housekeeping"],
        registry=registry,
    )
    assert decision.selected_skill == "skills.runtime.docker_prune_stopped_containers.v1"
    assert decision.policy["risk_class"] == "high_impact"
    assert decision.policy["execute_opt_in"] is True
    assert decision.observational is False
