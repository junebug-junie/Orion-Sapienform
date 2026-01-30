from __future__ import annotations

from typing import Any, Dict, List


_SEVERITY_ORDER = {"info": 0, "warn": 1, "error": 2}


def recommend_actions_from_alerts(alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
    actions: List[str] = []
    rationales: List[Dict[str, Any]] = []
    severity_max = "info"

    def _set_severity(severity: str) -> None:
        nonlocal severity_max
        if _SEVERITY_ORDER.get(severity, -1) > _SEVERITY_ORDER.get(severity_max, -1):
            severity_max = severity

    def _add_action(action: str) -> None:
        if action not in actions:
            actions.append(action)

    for alert in alerts:
        rule = alert.get("rule")
        severity = alert.get("severity") or "info"
        _set_severity(severity)
        if rule == "coherence_drop" and severity in {"warn", "error"}:
            if severity == "error":
                _add_action("stabilize_mode")
            _add_action("shorten_responses")
            _add_action("ask_one_clarifying_question")
            _add_action("reduce_novelty_seeking")
            _add_action("confirm_shared_goal")
            rationales.append(
                {"rule": rule, "severity": severity, "why": "coherence_drop suggests stabilization + clarity"}
            )
        elif rule == "valence_drop" and severity in {"warn", "error"}:
            _add_action("use_gentle_tone")
            _add_action("validate_and_confirm_intent")
            _add_action("avoid_confrontational_framing")
            rationales.append({"rule": rule, "severity": severity, "why": "valence_drop suggests softer framing"})
        elif rule == "novelty_spike" and severity in {"warn", "error"}:
            _add_action("capture_insight_candidate")
            _add_action("summarize_key_shift")
            if severity == "error":
                _add_action("slow_down")
            rationales.append({"rule": rule, "severity": severity, "why": "novelty_spike suggests capturing shift"})

    if any(alert.get("severity") == "error" for alert in alerts):
        _add_action("stabilize_mode")

    return {
        "severity_max": severity_max,
        "actions": actions,
        "rationales": rationales,
    }


def summarize_recommended_actions(policy: Dict[str, Any]) -> str:
    actions = policy.get("actions") if isinstance(policy, dict) else None
    if not actions:
        return "Actions: none"
    return "Actions: " + ", ".join(actions)
