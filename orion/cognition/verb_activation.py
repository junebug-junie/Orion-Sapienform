from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import yaml

VERBS_DIR = Path(__file__).resolve().parent / "verbs"
ACTIVE_MANIFEST_PATH = VERBS_DIR / "active.yaml"


@lru_cache(maxsize=1)
def _discover_verbs() -> Dict[str, Dict[str, str]]:
    verbs: Dict[str, Dict[str, str]] = {}
    if not VERBS_DIR.exists():
        return verbs

    for path in sorted(VERBS_DIR.glob("*.yaml")):
        if path.name == "active.yaml":
            continue
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            raw = {}

        name = str(raw.get("name") or path.stem).strip()
        if not name:
            continue

        desc = raw.get("description")
        if not isinstance(desc, str) or not desc.strip():
            first_line = next((line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()), "")
            desc = first_line

        verbs[name] = {
            "name": name,
            "description": str(desc or "").strip(),
            "path": str(path),
        }
    return verbs


@lru_cache(maxsize=1)
def _load_manifest() -> Dict[str, Any]:
    if not ACTIVE_MANIFEST_PATH.exists():
        return {"default": {"allow": [], "deny": []}, "nodes": {}}

    try:
        raw = yaml.safe_load(ACTIVE_MANIFEST_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        raw = {}

    default = raw.get("default") if isinstance(raw.get("default"), dict) else {}
    nodes = raw.get("nodes") if isinstance(raw.get("nodes"), dict) else {}

    def _normalize_rule(rule: Dict[str, Any]) -> Dict[str, List[str]]:
        allow = rule.get("allow") if isinstance(rule.get("allow"), list) else []
        deny = rule.get("deny") if isinstance(rule.get("deny"), list) else []
        return {
            "allow": sorted({str(v).strip() for v in allow if str(v).strip()}),
            "deny": sorted({str(v).strip() for v in deny if str(v).strip()}),
        }

    normalized_nodes: Dict[str, Dict[str, List[str]]] = {}
    for node_name, node_rule in nodes.items():
        if not isinstance(node_rule, dict):
            continue
        normalized_nodes[str(node_name).strip()] = _normalize_rule(node_rule)

    return {
        "default": _normalize_rule(default),
        "nodes": normalized_nodes,
    }


def refresh_verb_activation_cache() -> None:
    _discover_verbs.cache_clear()
    _load_manifest.cache_clear()


def list_all_verbs() -> List[str]:
    return sorted(_discover_verbs().keys())


def is_active(verb_name: str, node_name: str | None = None) -> bool:
    if not verb_name:
        return False

    manifest = _load_manifest()
    default_rule = manifest.get("default") or {}
    node_rule = (manifest.get("nodes") or {}).get(str(node_name or "").strip(), {})

    allow = set(default_rule.get("allow") or [])
    deny = set(default_rule.get("deny") or [])
    allow.update(node_rule.get("allow") or [])
    deny.update(node_rule.get("deny") or [])

    if verb_name in deny:
        return False
    if allow:
        return verb_name in allow
    return verb_name in _discover_verbs()


def build_verb_list(node_name: str | None = None, include_inactive: bool = False) -> List[Dict[str, Any]]:
    verbs = _discover_verbs()
    out: List[Dict[str, Any]] = []

    for verb_name in sorted(verbs.keys()):
        item = verbs[verb_name]
        active = is_active(verb_name, node_name=node_name)
        if not include_inactive and not active:
            continue
        out.append(
            {
                "name": verb_name,
                "description": item.get("description", ""),
                "active": active,
                "path": item.get("path", ""),
            }
        )
    return out
