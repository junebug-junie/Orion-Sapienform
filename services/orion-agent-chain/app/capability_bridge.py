from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field

from .actions_skill_registry import ActionsSkillRegistry


class CapabilityDecision(BaseModel):
    verb: str
    needs_skill: bool
    skill_family: str | None = None
    candidate_skills: list[str] = Field(default_factory=list)
    selected_skill: str | None = None
    reason: str
    confidence: float = 0.0
    policy: Dict[str, Any] = Field(default_factory=dict)
    observational: bool = True


def resolve_capability_decision(
    *,
    verb: str,
    preferred_skill_families: list[str] | None,
    registry: ActionsSkillRegistry,
) -> CapabilityDecision:
    families = [str(item).strip() for item in (preferred_skill_families or []) if str(item).strip()]
    if not families:
        families = ["system_inspection"]
    family = families[0]
    candidates = registry.by_family(family)
    candidate_ids = [item.skill_id for item in candidates]
    selected = candidate_ids[0] if candidate_ids else None
    if family == "runtime_housekeeping":
        for candidate_id in candidate_ids:
            if "docker_prune_stopped_containers" in candidate_id:
                selected = candidate_id
                break
    policy: Dict[str, Any] = {"risk_class": "read_only", "confirmation_required": False, "execute_opt_in": False}
    observational = True
    if candidates:
        selected_entry = next((item for item in candidates if item.skill_id == selected), candidates[0])
        policy = {
            "risk_class": selected_entry.risk_class,
            "confirmation_required": bool(selected_entry.requires_confirmation),
            "execute_opt_in": bool(selected_entry.requires_execute_opt_in),
        }
        observational = bool(selected_entry.observational)
    reason = (
        f"Verb '{verb}' prefers family '{family}', selected '{selected}'."
        if selected
        else f"No registered skill available for preferred family '{family}'."
    )
    return CapabilityDecision(
        verb=verb,
        needs_skill=True,
        skill_family=family,
        candidate_skills=candidate_ids,
        selected_skill=selected,
        reason=reason,
        confidence=0.87 if selected else 0.0,
        policy=policy,
        observational=observational,
    )


def normalize_capability_observation(
    *,
    decision: CapabilityDecision,
    execution_summary: str,
    raw_payload: Dict[str, Any] | None,
) -> Dict[str, Any]:
    return {
        "selected_verb": decision.verb,
        "selected_skill_family": decision.skill_family,
        "selected_skill": decision.selected_skill,
        "reason": decision.reason,
        "observational": decision.observational,
        "execution_summary": execution_summary,
        "raw_payload_ref": raw_payload or {},
        "capability_decision": decision.model_dump(mode="json"),
    }
