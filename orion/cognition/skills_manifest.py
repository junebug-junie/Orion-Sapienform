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
    if "up_all_services" in sid or "refresh_service_envs" in sid:
        return "runtime_housekeeping"
    # 2026-08-12: without this, builder_prune fell through to the final
    # `return "system_inspection"` -- which is capability_bridge.py's DEFAULT
    # family when preferred_skill_families is empty. It was not auto-selected
    # only because it sorted to index 1 rather than 0. That is an alphabetical
    # accident, not a gate.
    if "builder_prune" in sid:
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
    if "docker_prune_stopped_containers" in sid or "builder_prune" in sid:
        return "high_impact", False, False
    if "up_all_services" in sid or "refresh_service_envs" in sid:
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
                requires_execute_opt_in=(
                    "docker_prune_stopped_containers" in skill_id.lower()
                    # 2026-08-12: added. This skill deletes host data; it
                    # belongs on the same footing as the sibling prune above,
                    # not on the read-only default it was silently taking.
                    or "builder_prune" in skill_id.lower()
                    or "up_all_services" in skill_id.lower()
                    or "refresh_service_envs" in skill_id.lower()
                ),
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
