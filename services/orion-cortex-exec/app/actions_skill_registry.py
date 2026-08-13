from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml
from pydantic import BaseModel, Field


class ActionSkillManifestEntry(BaseModel):
    skill_id: str
    label: str
    description: str
    family: str
    read_only: bool
    idempotent: bool
    requires_confirmation: bool
    requires_execute_opt_in: bool = False
    observational: bool = True
    risk_class: str
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)


# 2026-08-13: ONE list, used by all three classifiers below.
#
# Before this, each of `_family_for_skill`, `_risk_for_skill`, and the
# `requires_execute_opt_in` expression carried its own hand-written substring
# check, and a skill had to be added to all three independently. builder_prune
# was added to none of them on arrival and spent its early life advertising
# itself as read_only + idempotent + observational with
# requires_confirmation=False -- the one skill in this repo that deletes host
# data. That was found and patched three separate times on 2026-08-12.
#
# A new host-mutating skill now goes in exactly one place. Adding it here
# cannot half-register it.
HOST_MUTATING_SKILL_MARKERS = (
    "docker_prune_stopped_containers",
    "builder_prune",
    "image_prune",
    "up_all_services",
    "refresh_service_envs",
)


def _is_host_mutating_skill(skill_id: str) -> bool:
    sid = str(skill_id or "").lower()
    return any(marker in sid for marker in HOST_MUTATING_SKILL_MARKERS)


def _family_for_skill(skill_id: str) -> str:
    sid = str(skill_id or "").lower()
    if "tailscale_mesh_status" in sid:
        return "mesh_presence"
    if "disk_health_snapshot" in sid:
        return "storage_health"
    if "github_recent_prs" in sid:
        return "repo_change_intel"
    if "docker.ps_status" in sid or ("docker" in sid and "ps_status" in sid):
        return "docker_inventory"
    if "docker_prune_stopped_containers" in sid:
        return "runtime_housekeeping"
    if "mesh_ops_round" in sid:
        return "runtime_housekeeping"
    if "up_all_services" in sid or "refresh_service_envs" in sid:
        return "runtime_housekeeping"
    # 2026-08-12: without this, builder_prune fell through to the final
    # `return "system_inspection"` -- which is capability_bridge.py's DEFAULT
    # family when preferred_skill_families is empty. It was not auto-selected
    # only because it sorted to index 1 rather than 0. That is an alphabetical
    # accident, not a gate.
    if _is_host_mutating_skill(sid):
        return "runtime_housekeeping"
    if "nvidia_smi" in sid or "gpu.nvidia" in sid:
        return "gpu_presence"
    if "biometrics.raw_recent" in sid or ("biometrics" in sid and "raw_recent" in sid):
        return "biometrics_recent"
    if "biometrics.snapshot" in sid or ("biometrics" in sid and "snapshot" in sid):
        return "biometrics_snapshot"
    if "notify" in sid:
        return "notification"
    if "time_now" in sid:
        return "temporal_context"
    if "discussion_window" in sid:
        return "chat_transcript"
    if "docker" in sid or "gpu" in sid:
        return "system_inspection"
    if "biometrics" in sid:
        return "runtime_health"
    return "system_inspection"


def _risk_for_skill(skill_id: str) -> tuple[str, bool, bool]:
    sid = str(skill_id or "").lower()
    # 2026-08-12: builder_prune matched NO case here and fell through to the
    # `read_only, True, True` default below, so the one skill in this repo that
    # deletes host data advertised itself as read_only + idempotent +
    # observational, requires_confirmation=False, requires_execute_opt_in=False.
    # Its sibling prune skill on the line below was already high_impact. The
    # misclassification also propagated to orion/normalizers/agent_trace.py,
    # which decides "did this have an effect" from risk_class alone -- so the
    # traces normalized as non-side-effecting too.
    if _is_host_mutating_skill(sid):
        return "high_impact", False, False
    if "notify" in sid:
        return "benign_actuation", False, False
    return "read_only", True, True


class ActionsSkillRegistry:
    """Normalized orion-actions skill manifest derived from skills.* verb YAMLs."""

    def __init__(self, *, verbs_dir: Path) -> None:
        self._verbs_dir = verbs_dir
        self._loaded = False
        self._skills: Dict[str, ActionSkillManifestEntry] = {}

    def _load(self) -> None:
        if self._loaded:
            return
        self._skills.clear()
        for path in sorted(self._verbs_dir.glob("skills.*.yaml")):
            raw = yaml.safe_load(path.read_text()) or {}
            if not isinstance(raw, dict):
                continue
            skill_id = str(raw.get("name") or "").strip()
            if not skill_id:
                continue
            risk_class, read_only, idempotent = _risk_for_skill(skill_id)
            entry = ActionSkillManifestEntry(
                skill_id=skill_id,
                label=str(raw.get("label") or skill_id),
                description=str(raw.get("description") or f"Skill {skill_id}"),
                family=_family_for_skill(skill_id),
                read_only=read_only,
                idempotent=idempotent,
                requires_confirmation=(risk_class == "high_impact"),
                requires_execute_opt_in=(
                    _is_host_mutating_skill(skill_id)
                ),
                observational=read_only,
                risk_class=risk_class,
                input_schema=raw.get("input_schema") if isinstance(raw.get("input_schema"), dict) else {},
                output_schema=raw.get("output_schema") if isinstance(raw.get("output_schema"), dict) else {},
            )
            self._skills[skill_id] = entry
        self._loaded = True

    def list(self) -> List[ActionSkillManifestEntry]:
        self._load()
        return list(self._skills.values())

    def by_family(self, family: str) -> List[ActionSkillManifestEntry]:
        fam = str(family or "").strip().lower()
        return [item for item in self.list() if item.family == fam]
