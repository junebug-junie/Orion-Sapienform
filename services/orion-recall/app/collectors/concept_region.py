"""Turn-scoped concept-region collector (Phase 6, concept-atlas pipeline).

Mirrors the salience/intent-gated shape of `active_packet.py`, not the
always-on producer shape of `chat_stance.py`. This module never runs
unconditionally and never registers a standing stance: it is a pure,
per-call function that, only when the current turn's text cheaply and
deterministically matches a concept label already materialized in the
substrate graph, returns a bounded neighborhood fragment scoped to that
single turn. If nothing matches, it returns empty without doing a store
read at all.

Matching is intentionally dumb: case-insensitive substring matching of
stored concept labels against the turn text, with a minimum label length
to avoid noise-matching on very short labels (e.g. "Go", "AI"). No
embedding or LLM call belongs in this module -- that is a separate,
heavier design decision explicitly deferred out of this phase.

Wiring this collector into the live turn-assembly pipeline (deciding
when/how it gets invoked during a real chat turn, and whether its output
feeds `chat_stance.py` or something else) is out of scope here. This file
only builds and tests the collector function in isolation.
"""

from __future__ import annotations

import logging
from typing import Any

from orion.core.schemas.cognitive_substrate import BaseSubstrateNodeV1, SubstrateEdgeV1
from orion.substrate.store import SubstrateGraphStore

logger = logging.getLogger(__name__)

# Search breadth for the label-match pass. `read_concept_region` ranks and
# truncates by salience/confidence, so a low-salience concept could be cut
# before we ever get to compare its label against the turn text unless we
# ask for a generous slice up front. This is still a single bounded call
# into the store's existing region-read API -- no new traversal method.
_DEFAULT_SEARCH_LIMIT_NODES = 500
_DEFAULT_SEARCH_LIMIT_EDGES = 500

# Minimum concept-label length eligible for substring matching. Labels
# shorter than this match too much incidental text (e.g. "Go", "It", "A")
# to be a useful cheap-and-deterministic signal. Documented judgment call
# for this phase, not a tuned threshold.
_MIN_LABEL_CHARS = 3


def _turn_text(query: Any) -> str:
    """Extract the current turn's text from `query`.

    Mirrors `active_packet.py`'s `_intent`/fragment-extraction convention:
    `query` is typed loosely (`Any`) and the turn text lives on a
    `.fragment` attribute. A bare string is also accepted directly so
    callers that already have raw turn text don't need to wrap it.
    """

    if query is None:
        return ""
    if isinstance(query, str):
        return query.strip()
    return str(getattr(query, "fragment", None) or "").strip()


def _node_label(node: Any) -> str:
    return str(getattr(node, "label", "") or "")


def _label_matches(label: str, turn_text_lower: str) -> bool:
    label_norm = label.strip().lower()
    if len(label_norm) < _MIN_LABEL_CHARS:
        return False
    return label_norm in turn_text_lower


def _node_to_fragment(node: BaseSubstrateNodeV1) -> dict[str, Any]:
    label = _node_label(node)
    definition = str(getattr(node, "definition", "") or "").strip()
    snippet = f"{label}: {definition}".strip(": ").strip() if definition else label
    promotion_state = str(getattr(node, "promotion_state", "") or "")
    try:
        salience = float(getattr(getattr(node, "signals", None), "salience", 0.0) or 0.0)
    except Exception:
        salience = 0.0
    return {
        "id": f"concept_region:node:{getattr(node, 'node_id', '')}",
        "source": "concept_region",
        "snippet": snippet,
        "text": snippet,
        "score": max(0.0, min(1.0, salience)),
        "tags": ["kind:concept", f"promotion_state:{promotion_state}"],
    }


def _edge_to_fragment(edge: SubstrateEdgeV1) -> dict[str, Any]:
    predicate = str(getattr(edge, "predicate", "") or "associated_with")
    source_id = str(getattr(getattr(edge, "source", None), "node_id", "") or "")
    target_id = str(getattr(getattr(edge, "target", None), "node_id", "") or "")
    snippet = f"{source_id} {predicate} {target_id}".strip()
    try:
        salience = float(getattr(edge, "salience", 0.0) or 0.0)
    except Exception:
        salience = 0.0
    return {
        "id": f"concept_region:edge:{getattr(edge, 'edge_id', '')}",
        "source": "concept_region",
        "snippet": snippet,
        "text": snippet,
        "score": max(0.0, min(1.0, salience)),
        "tags": ["kind:concept_edge", f"predicate:{predicate}"],
    }


def fetch_concept_region_fragment(
    query: Any,
    *,
    store: SubstrateGraphStore | None,
    limit_nodes: int = _DEFAULT_SEARCH_LIMIT_NODES,
    limit_edges: int = _DEFAULT_SEARCH_LIMIT_EDGES,
) -> list[dict[str, Any]]:
    """Return a turn-scoped concept-region fragment, or [] if nothing matched.

    Never persists anything, never raises, and never runs a store read
    unless there is turn text to match against. This is a per-call
    function only -- it must not be registered as a standing/always-on
    stance producer (see `chat_stance.py::_build_unification_registry`
    for the pattern this deliberately does NOT follow).
    """

    turn_text = _turn_text(query)
    if not turn_text:
        return []
    if store is None:
        return []

    turn_text_lower = turn_text.lower()

    try:
        region = store.read_concept_region(limit_nodes=limit_nodes, limit_edges=limit_edges)
    except Exception as exc:  # noqa: BLE001 - collector must degrade, never raise
        logger.debug("concept_region collector: store read failed: %s", exc)
        return []

    nodes = list(getattr(region, "nodes", None) or [])
    edges = list(getattr(region, "edges", None) or [])

    matched_nodes = [node for node in nodes if _label_matches(_node_label(node), turn_text_lower)]
    if not matched_nodes:
        return []

    matched_node_ids = {getattr(node, "node_id", None) for node in matched_nodes}
    matched_edges = [
        edge
        for edge in edges
        if getattr(getattr(edge, "source", None), "node_id", None) in matched_node_ids
        or getattr(getattr(edge, "target", None), "node_id", None) in matched_node_ids
    ]

    fragments: list[dict[str, Any]] = [_node_to_fragment(node) for node in matched_nodes]
    fragments.extend(_edge_to_fragment(edge) for edge in matched_edges)
    return fragments


__all__ = ["fetch_concept_region_fragment"]
