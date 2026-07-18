from __future__ import annotations

from typing import Any

from orion.schemas.recall_pcr import RetrievalIntentV1

_COLLECTOR_PLANS: dict[str, dict[str, bool]] = {
    "relational": {"cards": True, "rdf": True, "rdf_chat": True, "active_packet": True, "concept_region": True},
    "semantic": {"cards": True, "rdf": True, "rdf_chat": True, "active_packet": True, "concept_region": True},
    "procedural": {"cards": True, "active_packet": True, "concept_region": True},
    "open_loop": {"cards": True, "rdf_chat": True, "active_packet": True, "concept_region": True},
    "contradiction": {"cards": True, "rdf": True, "active_packet": True, "graphiti": True, "concept_region": True},
}


def collectors_for_intent(intent: RetrievalIntentV1 | str | None) -> dict[str, bool]:
    key = str(intent or "").strip()
    return dict(_COLLECTOR_PLANS.get(key, {}))


def apply_collector_plan(profile: dict[str, Any], plan: dict[str, bool]) -> dict[str, Any]:
    """Return a profile copy with backends disabled when not in the collector plan."""
    if not plan:
        return profile
    narrowed = dict(profile)
    if not plan.get("cards"):
        narrowed["cards_top_k"] = 0
        rel = dict(narrowed.get("relevance") or {})
        bw = dict(rel.get("backend_weights") or {})
        bw["cards"] = 0.0
        rel["backend_weights"] = bw
        narrowed["relevance"] = rel
    if not plan.get("rdf"):
        narrowed["rdf_top_k"] = 0
        narrowed["enable_rdf"] = False
    if not plan.get("rdf_chat"):
        # "rdf_chat" in this plan's vocabulary means "chat-turn text
        # content", not "RDF specifically" -- falkor_chat
        # (storage/falkor_chat_adapter.py, RECALL_FALKOR_IN_CHAT) is the
        # same content on a different backend and needs the same
        # suppression, or intents like "procedural"/"contradiction" (which
        # deliberately exclude chat-turn history) leak it back in via
        # whichever backend is active. Also disables the fetch itself (not
        # just its fusion weight) via enable_falkor_chat -- see worker.py's
        # falkor_chat_enabled, which reads this same profile field.
        rel = dict(narrowed.get("relevance") or {})
        bw = dict(rel.get("backend_weights") or {})
        bw["rdf_chat"] = 0.0
        bw["falkor_chat"] = 0.0
        rel["backend_weights"] = bw
        narrowed["relevance"] = rel
        narrowed["enable_falkor_chat"] = False
    narrowed["enable_sql_chat"] = False
    narrowed["sql_chat_top_k"] = 0
    narrowed["enable_sql_timeline"] = False
    return narrowed
