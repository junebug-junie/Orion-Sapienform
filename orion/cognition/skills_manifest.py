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

# 2026-10-01: skills that change the world but are not in the host-mutating
# list above. Both fell through to the `read_only` default, so the daily
# selector could offer them as "read-only skill probes" (daily pulse picked
# compose_service_bringup as one on 2026-08-30).
#
# STATE_CHANGING: changes host/runtime state. compose_service_bringup runs
# `docker compose build` + `up -d`. Deliberately NOT added to
# HOST_MUTATING_SKILL_MARKERS: that would also move its family to
# runtime_housekeeping and change which skill cortex-exec's capability bridge
# resolves for the system_inspection family (assess_runtime_state). The real
# runtime gate stays SKILLS_ALLOW_DOCKER_COMPOSE_BRINGUP in cortex-exec.
STATE_CHANGING_SKILL_MARKERS = (
    "compose_service_bringup",
)

# ACTUATING: acts outward without mutating the host. render_scene spends GPU
# watts on circe and persists a new image through orion-thought's chain; it is
# not an observation, so it is neither read-only nor idempotent.
ACTUATING_SKILL_MARKERS = (
    "notify",
    "render_scene",
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
    if "skills.perception." in sid or ("perception" in sid and "look_at_camera" in sid):
        return "perception"
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
    if any(marker in sid for marker in STATE_CHANGING_SKILL_MARKERS):
        return "state_change", False, False
    if any(marker in sid for marker in ACTUATING_SKILL_MARKERS):
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


def _short_purpose(entry: SkillManifestEntry, *, max_chars: int) -> str:
    """First sentence of the description (falling back to the label), cut to max_chars."""
    text = " ".join(str(entry.description or "").split())
    if not text or text == f"Skill {entry.skill_id}":
        text = str(entry.label or "").replace("Skills — ", "").strip()
    for stop in (". ", "; ", " — ", " -- "):
        idx = text.find(stop)
        if idx > 0:
            text = text[:idx]
            break
    text = text.rstrip(". ")
    if len(text) > max_chars:
        text = text[: max(0, max_chars - 3)].rstrip() + "..."
    return text


def build_bounded_skill_catalog(
    *,
    max_chars: int,
    entries: list[SkillManifestEntry] | None = None,
    read_only_only: bool = True,
    purpose_chars: int = 60,
    verbs_dir: Path | None = None,
) -> tuple[str, int]:
    """One line per skill (``skill_id: purpose``), total length <= max_chars.

    Built for prompts with a hard char budget (daily_metacog_v1). The full JSON
    catalog from ``build_compact_skill_catalog`` grows ~300 chars per skill and
    pushed that prompt over its limit on 2026-09-03; this form grows ~100 chars
    per skill and degrades in steps instead of failing:

    1. ``skill_id: purpose`` for every skill, if it fits;
    2. otherwise bare ``skill_id`` lines (ids are what the model must copy);
    3. otherwise as many ids as fit, plus a ``(+N more not listed)`` line.

    Returns ``(text, listed_count)`` where listed_count is how many skill ids the
    text actually names, so a caller can report the true count to the model.
    ``read_only_only`` defaults True because the daily selectors reject any
    non-read-only id anyway (orion-actions ``_normalize_daily_skill_selection``).
    """
    items = entries if entries is not None else load_skill_manifest(verbs_dir=verbs_dir)
    if read_only_only:
        items = [item for item in items if item.read_only]
    items = sorted(items, key=lambda item: item.skill_id)
    budget = max(0, int(max_chars))

    full = "\n".join(f"{item.skill_id}: {_short_purpose(item, max_chars=purpose_chars)}" for item in items)
    if len(full) <= budget:
        return full, len(items)

    ids_only = "\n".join(item.skill_id for item in items)
    if len(ids_only) <= budget:
        return ids_only, len(items)

    lines: list[str] = []
    for idx, item in enumerate(items):
        remaining = len(items) - idx - 1
        tail = f"(+{remaining} more not listed)" if remaining else ""
        candidate = "\n".join([*lines, item.skill_id, *([tail] if tail else [])])
        if len(candidate) > budget:
            break
        lines.append(item.skill_id)
    omitted = len(items) - len(lines)
    if omitted:
        lines.append(f"(+{omitted} more not listed)")
    text = "\n".join(lines)
    if len(text) > budget:
        # Budget too small for even the overflow note: list nothing rather than overrun.
        return "", 0
    return text, len(lines) - (1 if omitted else 0)
