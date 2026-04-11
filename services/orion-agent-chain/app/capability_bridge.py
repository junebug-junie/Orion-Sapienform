from __future__ import annotations

import json
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
    rejection_reasons: list[str] = Field(default_factory=list)
    resolution_failure: str | None = None


def resolve_capability_decision(
    *,
    verb: str,
    preferred_skill_families: list[str] | None,
    registry: ActionsSkillRegistry,
) -> CapabilityDecision:
    families = [str(item).strip() for item in (preferred_skill_families or []) if str(item).strip()]
    if not families:
        families = ["system_inspection"]
    family = None
    candidates = []
    rejection_reasons: list[str] = []
    for candidate_family in families:
        family = candidate_family
        candidates = registry.by_family(candidate_family)
        if candidates:
            break
        rejection_reasons.append(f"empty_candidates:{candidate_family}")
    family = family or families[0]
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
    resolution_failure = None
    reason = (
        f"Verb '{verb}' prefers family '{family}', selected '{selected}'."
        if selected
        else f"No registered skill available for preferred family '{family}'."
    )
    if not selected:
        resolution_failure = "no_skill_for_preferred_families"
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
        rejection_reasons=rejection_reasons,
        resolution_failure=resolution_failure,
    )


def _domain_reason_from_skill_result(sr: Dict[str, Any]) -> str | None:
    """In-domain failure for mesh/storage-style skill JSON (mirrors cortex-exec executor heuristics)."""
    if sr.get("unsupported") is True and sr.get("available") is False:
        return str(sr.get("reason") or "unsupported").strip() or "unsupported"
    if sr.get("available") is False and sr.get("reason"):
        return str(sr.get("reason")).strip()
    devices = sr.get("devices")
    summary = sr.get("summary")
    if isinstance(devices, list) and isinstance(summary, dict) and len(devices) > 0:
        unsup = int(summary.get("unsupported") or 0)
        healthy = int(summary.get("healthy") or 0)
        if unsup >= len(devices) and healthy == 0:
            return "disk_health_unavailable_for_all_probed_devices"
    return None


def skill_domain_failure_reason_from_skill_payload(obj: Any) -> str | None:
    """
    Detect domain-negative skill output from nested cortex final_text.
    Accepts either a full SkillVerbOutput-shaped dict or the inner skill_result JSON
    (what skills emit as final_text).
    """
    if not isinstance(obj, dict):
        return None
    if obj.get("ok") is False:
        err = obj.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err.get("message") or "").strip() or None
        return str(obj.get("status") or "failed").strip() or None
    meta = obj.get("metadata")
    if isinstance(meta, dict):
        sr = meta.get("skill_result")
        if isinstance(sr, dict):
            r = _domain_reason_from_skill_result(sr)
            if r:
                return r
    return _domain_reason_from_skill_result(obj)


def skill_domain_failure_reason_from_final_text(final_text: str) -> str | None:
    ft = (final_text or "").strip()
    if not ft:
        return None
    try:
        parsed = json.loads(ft)
    except Exception:
        return None
    return skill_domain_failure_reason_from_skill_payload(parsed)


def humanize_domain_failure(selected_skill: str, reason: str) -> str:
    sk = (selected_skill or "").lower()
    if "mesh" in sk or "tailscale" in sk:
        return f"Could not check mesh status: {reason}"
    if "storage" in sk or "disk" in sk:
        return f"Could not check disk health: {reason}"
    return f"Could not complete this check: {reason}"


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
