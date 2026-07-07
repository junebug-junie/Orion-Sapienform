"""orion-thought → orion-mind advisory enrichment (unified turn coloring).

The unified turn computes stance cold via the ``stance_react`` verb. This module
optionally runs Mind first and selects a strict, mode-agnostic self/attention
subset as an *advisory* prompt prior. ``stance_react`` remains the sole author of
ThoughtEventV1 and reconciles this coloring. Everything fails open.
"""
from __future__ import annotations

import logging
from typing import Any

from orion.mind.v1 import MindRunResultV1

logger = logging.getLogger("orion-thought.mind_enrichment")

# Strict allow-list of coloring keys. Any un-listed ChatStanceBrief / decision
# field is absent by construction (no deny-list, no leakage of future fields).
MIND_COLORING_ALLOWED_KEYS: frozenset[str] = frozenset(
    {
        "attention_frontier",
        "reflective_themes",
        "curiosity_threads",
        "self_relevance",
        "identity_salience",
        "juniper_relevance",
        "mind_quality",
        "mind_run_id",
        "snapshot_hash",
    }
)

_MAX_STR_CHARS = 240


def _clip(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()[:_MAX_STR_CHARS]
    return value


def _str_list(value: Any, *, max_items: int) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    out: list[str] = []
    for item in value:
        text = str(item).strip()[:_MAX_STR_CHARS]
        if text:
            out.append(text)
        if len(out) >= max_items:
            break
    return out


def select_mind_coloring(result: MindRunResultV1, *, max_items: int = 3) -> dict[str, Any] | None:
    """Project the mode-agnostic self/attention subset of a Mind run.

    Returns None (skip enrichment) unless the run is ok AND produced
    meaningful_synthesis AND carries at least one substantive signal. Never
    injects an empty shell. Selection is a strict allow-list.
    """
    if not result.ok:
        return None
    brief = result.brief
    if brief.mind_quality != "meaningful_synthesis":
        return None

    frontier = brief.active_frontier
    selected = list(frontier.selected) if frontier is not None else []
    selected = selected[:max_items]
    attention_frontier = [
        {
            "label": _clip(m.label),
            "summary": _clip(m.summary),
            "score": round(float(m.score), 4),
        }
        for m in selected
    ]
    curiosity_threads = [_clip(m.summary) for m in selected if str(m.summary).strip()][:max_items]

    stance_payload = brief.stance_payload if isinstance(brief.stance_payload, dict) else {}
    reflective_themes = _str_list(stance_payload.get("reflective_themes"), max_items=max_items)
    self_relevance = _clip(stance_payload.get("self_relevance")) if stance_payload.get("self_relevance") else None
    identity_salience = stance_payload.get("identity_salience") or None
    juniper_relevance = _clip(stance_payload.get("juniper_relevance")) if stance_payload.get("juniper_relevance") else None

    # No empty-shell cognition: require at least one substantive signal.
    has_substance = bool(
        attention_frontier or reflective_themes or curiosity_threads
        or self_relevance or juniper_relevance
    )
    if not has_substance:
        return None

    return {
        "attention_frontier": attention_frontier,
        "reflective_themes": reflective_themes,
        "curiosity_threads": curiosity_threads,
        "self_relevance": self_relevance,
        "identity_salience": identity_salience,
        "juniper_relevance": juniper_relevance,
        "mind_quality": brief.mind_quality,
        "mind_run_id": str(result.mind_run_id),
        "snapshot_hash": result.snapshot_hash,
    }
