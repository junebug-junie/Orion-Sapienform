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


DELIVERY_SAFE_OUTPUT_MODES = frozenset(
    {
        "implementation_guide",
        "tutorial",
        "code_delivery",
        "comparative_analysis",
        "decision_support",
    }
)


def _is_concrete_ops_query(text: str | None) -> bool:
    lowered = str(text or "").lower()
    if not lowered.strip():
        return False
    needles = (
        "v100",
        "a100",
        "h100",
        "ups",
        "apc",
        "power",
        "watt",
        "runtime",
        "battery",
        "hardware",
        "gpu",
        "troubleshoot",
    )
    return any(needle in lowered for needle in needles)


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


def delivery_safe_recall_decision(
    recall_cfg: Dict[str, Any],
    steps: Iterable[ExecutionStep],
    *,
    output_mode: str | None = None,
    verb_profile: Optional[str] = None,
    user_text: str | None = None,
) -> Dict[str, Any]:
    recall_required = bool(recall_cfg.get("required", False))
    recall_enabled = recall_enabled_value(recall_cfg)
    explicit_profile = recall_cfg.get("profile")
    base_profile, profile_source = resolve_profile(recall_cfg, verb_profile=verb_profile)
    base_should_run, base_reason = should_run_recall(recall_cfg, steps)

    if output_mode in DELIVERY_SAFE_OUTPUT_MODES and not explicit_profile:
        if recall_required:
            return {
                "run_recall": base_should_run,
                "reason": "delivery_required_profile_narrowed",
                "profile": "assist.light.v1",
                "profile_source": "delivery_safe_default",
                "recall_gating_reason": "delivery_required_profile_narrowed",
                "effective_enabled": True,
            }
        if recall_enabled:
            return {
                "run_recall": False,
                "reason": "delivery_safe_default_disabled",
                "profile": base_profile,
                "profile_source": profile_source,
                "recall_gating_reason": "delivery_safe_default_disabled",
                "effective_enabled": False,
            }

    if _is_concrete_ops_query(user_text) and not explicit_profile and not recall_required:
        return {
            "run_recall": False,
            "reason": "concrete_ops_default_disabled",
            "profile": "assist.light.v1",
            "profile_source": "concrete_ops_guardrail",
            "recall_gating_reason": "concrete_ops_default_disabled",
            "effective_enabled": False,
        }

    return {
        "run_recall": base_should_run,
        "reason": base_reason,
        "profile": base_profile,
        "profile_source": profile_source,
        "recall_gating_reason": base_reason,
        "effective_enabled": recall_enabled or recall_required,
    }


def has_inline_recall(steps: Iterable[ExecutionStep]) -> bool:
    return any("RecallService" in step.services for step in steps)
