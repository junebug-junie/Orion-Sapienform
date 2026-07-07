"""orion-thought → orion-mind advisory enrichment (unified turn coloring).

The unified turn computes stance cold via the ``stance_react`` verb. This module
optionally runs Mind first and selects a strict, mode-agnostic self/attention
subset as an *advisory* prompt prior. ``stance_react`` remains the sole author of
ThoughtEventV1 and reconciles this coloring. Everything fails open.
"""
from __future__ import annotations

import logging
from typing import Any

from orion.mind.v1 import MindRunPolicyV1, MindRunRequestV1, MindRunResultV1
from orion.schemas.thought import StanceReactRequestV1

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
_MAX_USER_TEXT_CHARS = 20_000
_MAX_OPEN_LOOPS = 6


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


def _situation_compact_from_broadcast(request: StanceReactRequestV1) -> dict[str, Any] | None:
    """Fold real open-loop / selected-description text into the accepted
    situation_compact facet. Returns None when there is no usable text.
    """
    broadcast = request.association.broadcast
    if broadcast is None:
        return None
    loops: list[dict[str, str]] = []
    for loop in (broadcast.frame.open_loops or [])[:_MAX_OPEN_LOOPS]:
        description = (loop.description or "").strip()
        if not description:
            continue
        entry: dict[str, str] = {"description": description[:_MAX_STR_CHARS]}
        why = (loop.why_it_matters or "").strip()
        if why:
            entry["why_it_matters"] = why[:_MAX_STR_CHARS]
        loops.append(entry)
    selected = (broadcast.selected_description or "").strip()
    if not loops and not selected:
        return None
    compact: dict[str, Any] = {"attention_situation": True}
    if selected:
        compact["selected_focus"] = selected[:_MAX_STR_CHARS]
    if loops:
        compact["open_loops"] = loops
    return compact


def build_light_mind_request(
    request: StanceReactRequestV1,
    *,
    wall_time_ms: int,
    router_profile: str,
) -> MindRunRequestV1:
    """Build a bounded Mind request with NO cognitive-projection cold rebuild.

    Evidence in v1 is the current user turn (as a single current_turn item),
    plus recall_bundle (only if already threaded on stance_inputs) and a
    situation_compact facet derived from the attention broadcast.
    """
    user_text = (request.user_message or "").strip()[:_MAX_USER_TEXT_CHARS]
    snapshot: dict[str, Any] = {"user_text": user_text, "messages_tail": []}

    facets: dict[str, Any] = {}
    stance_inputs = request.stance_inputs if isinstance(request.stance_inputs, dict) else {}
    recall_bundle = stance_inputs.get("recall_bundle")
    if isinstance(recall_bundle, dict) and recall_bundle:
        facets["recall_bundle"] = recall_bundle
    situation = _situation_compact_from_broadcast(request)
    if situation:
        facets["situation_compact"] = situation
    if facets:
        snapshot["facets"] = facets

    return MindRunRequestV1(
        correlation_id=request.correlation_id,
        session_id=request.session_id,
        trigger="user_turn",
        snapshot_inputs=snapshot,
        policy=MindRunPolicyV1(
            n_loops_max=1,
            wall_time_ms_max=max(1, int(wall_time_ms)),
            router_profile_id=router_profile or "default",
        ),
    )
