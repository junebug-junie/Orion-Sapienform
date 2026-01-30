from __future__ import annotations

from typing import Any, Dict, List


_RULE_EXPLANATIONS = {
    "coherence_drop": {
        "likely_causes": [
            "context_fragmentation",
            "goal_mismatch",
            "multi-thread overload",
            "assumption_mismatch",
        ],
        "suggested_questions": [
            "What outcome are we optimizing for right now?",
            "Which thread should we prioritize?",
            "What assumption might be wrong?",
        ],
        "summary": "Coherence dropped sharply; likely context fragmentation or mismatch.",
    },
    "valence_drop": {
        "likely_causes": [
            "frustration_signal",
            "misalignment",
            "tone_mismatch",
        ],
        "suggested_questions": [
            "Did something feel off in my last response?",
            "Do you want directness or gentleness right now?",
        ],
        "summary": "Valence dropped; likely frustration, misalignment, or tone mismatch.",
    },
    "novelty_spike": {
        "likely_causes": [
            "new_concept_entry",
            "insight_candidate",
            "topic_shift",
        ],
        "suggested_questions": [
            "Should we capture this as an insight?",
            "Is this a new direction or a side-thread?",
        ],
        "summary": "Novelty spiked; likely new concept, insight, or topic shift.",
    },
}


def explain_alerts(alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_rule: List[Dict[str, Any]] = []
    likely_causes: List[str] = []
    suggested_questions: List[str] = []
    summary_parts: List[str] = []

    for alert in alerts:
        rule = alert.get("rule")
        severity = alert.get("severity") or "info"
        if rule not in _RULE_EXPLANATIONS:
            continue
        base = _RULE_EXPLANATIONS[rule]
        entry = {
            "rule": rule,
            "severity": severity,
            "likely_causes": list(base["likely_causes"]),
            "suggested_questions": list(base["suggested_questions"]),
        }
        by_rule.append(entry)
        for cause in base["likely_causes"]:
            if cause not in likely_causes:
                likely_causes.append(cause)
        for question in base["suggested_questions"]:
            if question not in suggested_questions:
                suggested_questions.append(question)
        summary_parts.append(f"{rule} ({severity})")

    summary = "No alert explanations."
    if summary_parts:
        summary = "Alert explanations: " + "; ".join(summary_parts)

    return {
        "summary": summary,
        "likely_causes": likely_causes,
        "suggested_questions": suggested_questions,
        "by_rule": by_rule,
    }


def summarize_explanations(expl: Dict[str, Any]) -> str:
    if not isinstance(expl, dict):
        return "Alert Explanation: none"
    summary = expl.get("summary")
    if not summary:
        return "Alert Explanation: none"
    return f"Alert Explanation: {summary}"
