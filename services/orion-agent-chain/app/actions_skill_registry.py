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
    risk_class: str
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)


def _family_for_skill(skill_id: str) -> str:
    sid = str(skill_id or "").lower()
    if "docker" in sid or "gpu" in sid or "landing_pad" in sid:
        return "system_inspection"
    if "biometrics" in sid:
        return "runtime_health"
    if "notify" in sid:
        return "notification"
    if "time_now" in sid:
        return "temporal_context"
    return "system_inspection"


def _risk_for_skill(skill_id: str) -> tuple[str, bool, bool]:
    sid = str(skill_id or "").lower()
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
