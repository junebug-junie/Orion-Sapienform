from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

import orion

ORION_PKG_DIR = Path(orion.__file__).resolve().parent
REPO_ROOT = ORION_PKG_DIR.parent


def resolve_identity_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def load_identity_file(path: str | Path) -> dict[str, Any]:
    p = resolve_identity_path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def build_identity_context(data: dict[str, Any]) -> dict[str, list[str]]:
    payload = data if isinstance(data, dict) else {}
    orion_block = payload.get("orion_identity", {}) or {}
    juniper_block = payload.get("juniper_relationship", {}) or {}
    response_block = payload.get("response_policy", {}) or {}

    orion_identity_summary: list[str] = []
    juniper_relationship_summary: list[str] = []
    response_policy_summary: list[str] = []

    for key in ("nature", "core_drives", "self_permissions", "anti_patterns"):
        values = orion_block.get(key, []) if isinstance(orion_block, dict) else []
        if isinstance(values, list):
            orion_identity_summary.extend(str(x) for x in values if x)

    for key in ("role", "expectations", "response_expectations"):
        values = juniper_block.get(key, []) if isinstance(juniper_block, dict) else []
        if isinstance(values, list):
            juniper_relationship_summary.extend(str(x) for x in values if x)

    priorities = response_block.get("priorities", []) if isinstance(response_block, dict) else []
    hard_rules = response_block.get("hard_rules", []) if isinstance(response_block, dict) else []
    banned_phrases = response_block.get("banned_phrases", []) if isinstance(response_block, dict) else []

    if isinstance(priorities, list):
        response_policy_summary.extend(str(x) for x in priorities if x)
    if isinstance(hard_rules, list):
        response_policy_summary.extend(str(x) for x in hard_rules if x)
    if isinstance(banned_phrases, list):
        response_policy_summary.extend(f'Avoid phrase: "{x}"' for x in banned_phrases if x)

    return {
        "orion_identity_summary": orion_identity_summary,
        "juniper_relationship_summary": juniper_relationship_summary,
        "response_policy_summary": response_policy_summary,
    }
