from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

from orion.schemas.cortex.schemas import ExecutionStep


def _normalize_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return str(value).lower() not in {"false", "0", "no", "off"}
    return bool(value)


def recall_enabled_value(recall_cfg: Dict[str, Any]) -> bool:
    return _normalize_bool(recall_cfg.get("enabled", True), default=True)


def resolve_mode_profile(mode: str | None) -> Tuple[str, str]:
    normalized = str(mode or "").lower()
    if normalized == "deep":
        return "deep.graph.v1", "mode"
    if normalized == "graph":
        return "graphtri.v1", "mode"
    return "reflect.v1", "mode"


def resolve_profile(
    recall_cfg: Dict[str, Any],
    *,
    verb_profile: Optional[str] = None,
    step: Optional[ExecutionStep] = None,
    is_recall_step: bool = False,
) -> Tuple[str, str]:
    explicit = recall_cfg.get("profile")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip(), "explicit"
    if is_recall_step and step and step.recall_profile:
        return step.recall_profile, "step"
    if isinstance(verb_profile, str) and verb_profile.strip():
        return verb_profile.strip(), "verb"
    mode = recall_cfg.get("mode")
    if mode:
        profile, source = resolve_mode_profile(str(mode))
        return profile, source
    return "reflect.v1", "fallback"


def should_run_recall(
    recall_cfg: Dict[str, Any],
    steps: Iterable[ExecutionStep],
) -> Tuple[bool, str]:
    recall_required = bool(recall_cfg.get("required", False))
    recall_enabled = recall_enabled_value(recall_cfg)
    enabled_or_required = recall_enabled or recall_required
    needs_memory = any(step.requires_memory for step in steps)
    if not enabled_or_required:
        return False, "disabled_by_client"
    if not (needs_memory or recall_required):
        return False, "no_memory_required"
    return True, "enabled"


def has_inline_recall(steps: Iterable[ExecutionStep]) -> bool:
    return any("RecallService" in step.services for step in steps)
