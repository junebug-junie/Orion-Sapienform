from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class SkillManifestEntry(BaseModel):
    skill_id: str
    label: str
    description: str
    family: str
    read_only: bool
    idempotent: bool
    risk_class: str
    requires_confirmation: bool = False
    requires_execute_opt_in: bool = False
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)


def _default_verbs_dir() -> Path:
    return Path(__file__).resolve().parent / "verbs"


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
    if "nvidia_smi" in sid or "gpu.nvidia" in sid:
        return "gpu_presence"
    if "biometrics.raw_recent" in sid or ("biometrics" in sid and "raw_recent" in sid):
        return "biometrics_recent"
    if "biometrics.snapshot" in sid or ("biometrics" in sid and "snapshot" in sid):
        return "biometrics_snapshot"
    if "landing_pad.metrics" in sid or ("landing_pad" in sid and "metrics" in sid):
        return "landing_pad_metrics"
    if "landing_pad.last_events" in sid or ("landing_pad" in sid and "last_events" in sid):
        return "landing_pad_events"
    if "notify" in sid:
        return "notification"
    if "time_now" in sid:
        return "temporal_context"
    if "discussion_window" in sid:
        return "chat_transcript"
    if "docker" in sid or "gpu" in sid or "landing_pad" in sid:
        return "system_inspection"
    if "biometrics" in sid:
        return "runtime_health"
    return "system_inspection"


def _risk_for_skill(skill_id: str) -> tuple[str, bool, bool]:
    sid = str(skill_id or "").lower()
    if "docker_prune_stopped_containers" in sid:
        return "high_impact", False, False
    if "notify" in sid:
        return "benign_actuation", False, False
    return "read_only", True, True


def load_skill_manifest(*, verbs_dir: Path | None = None) -> list[SkillManifestEntry]:
    root = verbs_dir or _default_verbs_dir()
    items: list[SkillManifestEntry] = []
    for path in sorted(root.glob("skills.*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            continue
        skill_id = str(raw.get("name") or "").strip()
        if not skill_id:
            continue
        risk_class, read_only, idempotent = _risk_for_skill(skill_id)
        items.append(
            SkillManifestEntry(
                skill_id=skill_id,
                label=str(raw.get("label") or skill_id),
                description=str(raw.get("description") or f"Skill {skill_id}"),
                family=_family_for_skill(skill_id),
                read_only=read_only,
                idempotent=idempotent,
                risk_class=risk_class,
                requires_confirmation=(risk_class == "high_impact"),
                requires_execute_opt_in=("docker_prune_stopped_containers" in skill_id.lower()),
                input_schema=raw.get("input_schema") if isinstance(raw.get("input_schema"), dict) else {},
                output_schema=raw.get("output_schema") if isinstance(raw.get("output_schema"), dict) else {},
            )
        )
    return items


def build_compact_skill_catalog(*, verbs_dir: Path | None = None) -> str:
    payload = [
        {
            "skill_id": item.skill_id,
            "label": item.label,
            "description": item.description[:200],
            "read_only": item.read_only,
            "risk_class": item.risk_class,
        }
        for item in load_skill_manifest(verbs_dir=verbs_dir)
    ]
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)
