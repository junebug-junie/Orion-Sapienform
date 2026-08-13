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
                    _is_host_mutating_skill(skill_id)
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
