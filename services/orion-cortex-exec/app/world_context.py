from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def filter_world_context_capsule(
    candidate: dict[str, Any] | None,
    *,
    min_confidence: float,
    max_topics: int,
    max_age_hours: int,
    politics_default: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    diag = {
        "capsule_age_hours": None,
        "capsule_filtered_reason": "none",
        "stance_world_context_items_used": 0,
        "politics_context_suppressed": True,
    }
    if not isinstance(candidate, dict):
        diag["capsule_filtered_reason"] = "missing_capsule"
        return None, diag
    generated_at = candidate.get("generated_at")
    if isinstance(generated_at, str):
        try:
            gen_dt = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
            diag["capsule_age_hours"] = max(0.0, (datetime.now(timezone.utc) - gen_dt).total_seconds() / 3600.0)
        except Exception:
            diag["capsule_filtered_reason"] = "invalid_generated_at"
    topics = candidate.get("salient_topics") if isinstance(candidate.get("salient_topics"), list) else []
    filtered = []
    for topic in topics:
        if not isinstance(topic, dict):
            continue
        if topic.get("expires_at"):
            try:
                if datetime.fromisoformat(str(topic.get("expires_at")).replace("Z", "+00:00")) <= datetime.now(timezone.utc):
                    continue
            except Exception:
                continue
        if bool(topic.get("disputed")):
            continue
        if bool(topic.get("do_not_volunteer")) and politics_default == "only_when_requested":
            diag["politics_context_suppressed"] = True
        confidence = float(topic.get("confidence") or 0.0)
        if confidence < min_confidence:
            continue
        filtered.append(topic)
        if len(filtered) >= max_topics:
            break
    if diag["capsule_age_hours"] is not None and float(diag["capsule_age_hours"]) > float(max_age_hours):
        diag["capsule_filtered_reason"] = "capsule_expired"
        return None, diag
    if not filtered:
        diag["capsule_filtered_reason"] = "no_eligible_topics"
        return None, diag
    out = dict(candidate)
    out["salient_topics"] = filtered
    diag["stance_world_context_items_used"] = len(filtered)
    return out, diag
