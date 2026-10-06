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

`fetch_concept_region_fragment()` above never persists anything -- that
promise is unchanged. `reinforce_matched_concepts()` and
`fetch_concept_region_fragment_and_reinforce()` below are a separate,
explicitly-named write path added later (see CONCEPT_REINFORCEMENT_DESIGN.md
in this same folder for the full design conversation): being surfaced in a
live turn is treated as a real "this concept is still relevant" signal that
nudges `signals.activation` back up, the mirror image of the decay this
concept-atlas system already applies. Only activation moves -- never
confidence, salience, recency_score, or temporal.observed_at; see the design
doc for why each of those is deliberately left alone.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Iterable

from orion.core.schemas.cognitive_substrate import BaseSubstrateNodeV1, SubstrateEdgeV1
from orion.substrate.falkor_codec import EXTERNALLY_OWNED_METADATA_KEYS
from orion.substrate.store import SubstrateGraphStore

logger = logging.getLogger(__name__)

# Mirrors orion.memory.crystallization.dynamics.recall_boost()'s boost
# constant and math exactly -- a structurally identical "this was retrieved"
# signal, reusing an already-validated value instead of inventing a new one.
_RECALL_REINFORCEMENT_BOOST = 0.08

_NODE_FRAGMENT_ID_PREFIX = "concept_region:node:"

# Search breadth for the label-match pass. `read_concept_region` ranks and
# truncates by salience/confidence, so a low-salience concept could be cut
# before we ever get to compare its label against the turn text unless we
# ask for a generous slice up front. On Falkor this ranking runs in the
# database (orion/substrate/falkor_direct.py); only matched rows come back.
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
        "id": f"{_NODE_FRAGMENT_ID_PREFIX}{getattr(node, 'node_id', '')}",
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
        # A store that can apply the label filter itself (the hydration-free
        # Falkor handle, orion/substrate/falkor_direct.py) only fetches full
        # rows for what matches. Same result as filtering the full slice
        # below -- the filter below is idempotent on it.
        matching = getattr(store, "read_concept_region_matching", None)
        if callable(matching):
            region = matching(
                keep_label=lambda label: _label_matches(label, turn_text_lower),
                limit_nodes=limit_nodes,
                limit_edges=limit_edges,
            )
        else:
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


def reinforce_matched_concepts(
    matched_node_ids: Iterable[str],
    *,
    store: SubstrateGraphStore | None,
    boost: float = _RECALL_REINFORCEMENT_BOOST,
    abandoned: "threading.Event | None" = None,
) -> int:
    """Bump activation for concept nodes that were actually surfaced in a live turn.

    Being recalled is a real but weaker reinforcement signal than recurrence
    (see CONCEPT_REINFORCEMENT_DESIGN.md in this folder) -- mirrors
    `orion.memory.crystallization.dynamics.recall_boost()`'s exact math:
    `activation = current + (1 - current) * boost`, asymptotic toward 1.0,
    never overshoots. Only `signals.activation.activation` moves; confidence,
    salience, recency_score, and temporal.observed_at are never touched here
    (see the design doc for why each is deliberately left alone).

    Never raises: a missing store, a missing/non-concept node, or a failed
    write for one node degrades to skipping that node, matching this
    module's read-side conventions. Returns the count of nodes actually
    reinforced.

    Never calls `store.snapshot()` (a complete graph read). In the live
    service `store` is `FalkorDirectConceptStore`
    (`orion/substrate/falkor_direct.py`), which has no snapshot at all:
    `get_node_and_identity_key()` is one single-node `GRAPH.RO_QUERY` and
    `upsert_node()` is one `MERGE ... SET`. Stores without the combined read
    (the in-memory store) use `get_node_by_id()` + `get_identity_key_by_node_id()`.

    ``abandoned``: checked before every node, so once the recall stops waiting
    for this call (deadline passed mid-loop) no further node is read or
    written. Nodes already written before that moment stay written.
    """
    node_ids = {str(node_id) for node_id in matched_node_ids if node_id}
    if not node_ids or store is None:
        return 0

    clamped_boost = min(1.0, max(0.0, boost))

    read_both = getattr(store, "get_node_and_identity_key", None)

    reinforced = 0
    for node_id in sorted(node_ids):
        if abandoned is not None and abandoned.is_set():
            logger.debug(
                "concept_region reinforcement stopped: recall abandoned this call reinforced=%s remaining=%s",
                reinforced,
                len(node_ids) - reinforced,
            )
            break
        try:
            if callable(read_both):
                node, identity_key = read_both(node_id)
            else:
                node = store.get_node_by_id(node_id)
                identity_key = (
                    store.get_identity_key_by_node_id(node_id)
                    if node is not None and getattr(node, "node_kind", None) == "concept"
                    else None
                )
        except Exception as exc:  # noqa: BLE001 - must degrade, never raise
            logger.debug("concept_region reinforcement: read of %s failed: %s", node_id, exc)
            continue
        if node is None or getattr(node, "node_kind", None) != "concept":
            continue
        if not identity_key:
            # Falkor's codec writes `identity_key or ""` unconditionally on
            # every upsert -- a missing identity here would durably clobber
            # this node's real identity_key rather than leaving it alone.
            # Every current producer registers a real identity_key at
            # creation, so this should be rare; skip rather than risk it.
            logger.debug("concept_region reinforcement: node %s has no known identity_key, skipping", node_id)
            continue
        try:
            activation_signal = node.signals.activation
            current = min(1.0, max(0.0, activation_signal.activation))
            boosted = min(1.0, current + (1.0 - current) * clamped_boost)
            updated_signal = activation_signal.model_copy(update={"activation": boosted})
            updated_signals = node.signals.model_copy(update={"activation": updated_signal})
            updated_node = node.model_copy(update={"signals": updated_signals})
            if abandoned is not None and abandoned.is_set():
                break
            # skip_metadata_keys: updated_node's metadata is node.metadata,
            # read moments earlier by get_node_by_id() above -- only
            # signals.activation is actually recomputed here, but
            # re-persisting the whole node re-writes reducer-owned metadata
            # fields (prediction_error, contributing_turn_ids) from that
            # stale read too. Same protection already proven for
            # SubstrateDynamicsEngine.tick() and (2026-07-30)
            # SubstrateGraphMaterializer.apply_record()'s merge branch,
            # concept_induction's materialize_concept_profile_to_falkor(),
            # and orion-hub's concept decay scheduler -- see
            # falkor_codec.EXTERNALLY_OWNED_METADATA_KEYS's docstring.
            store.upsert_node(
                identity_key=identity_key,
                node=updated_node,
                skip_metadata_keys=EXTERNALLY_OWNED_METADATA_KEYS,
            )
            reinforced += 1
        except Exception as exc:  # noqa: BLE001 - one node's failure must not block the rest
            logger.debug("concept_region reinforcement: node %s failed: %s", node_id, exc)

    return reinforced


def fetch_concept_region_fragment_and_reinforce(
    query: Any,
    *,
    store: SubstrateGraphStore | None,
    limit_nodes: int = _DEFAULT_SEARCH_LIMIT_NODES,
    limit_edges: int = _DEFAULT_SEARCH_LIMIT_EDGES,
    abandoned: "threading.Event | None" = None,
) -> list[dict[str, Any]]:
    """`fetch_concept_region_fragment()` plus reinforcement for whatever it matched.

    Composes the two functions above rather than modifying
    `fetch_concept_region_fragment()` itself, which keeps its own
    never-persists contract intact for any other caller. This is the
    function the live turn-assembly pipeline should call.

    ``abandoned``: set by the caller when it stops waiting for this call (the
    recall deadline passed while this ran in a worker thread). Checked before
    reinforcement starts and again before every node's write (inside
    `reinforce_matched_concepts`), so fragments the recall dropped are not
    reinforced as if they had been surfaced in the turn.
    """
    fragments = fetch_concept_region_fragment(
        query, store=store, limit_nodes=limit_nodes, limit_edges=limit_edges
    )
    if not fragments or store is None:
        return fragments

    matched_node_ids = [
        fragment["id"][len(_NODE_FRAGMENT_ID_PREFIX):]
        for fragment in fragments
        if str(fragment.get("id", "")).startswith(_NODE_FRAGMENT_ID_PREFIX)
    ]
    if matched_node_ids:
        if abandoned is not None and abandoned.is_set():
            logger.debug(
                "concept_region reinforcement skipped: recall abandoned this call matched=%s",
                len(matched_node_ids),
            )
            return fragments
        reinforce_matched_concepts(matched_node_ids, store=store, abandoned=abandoned)

    return fragments


__all__ = [
    "fetch_concept_region_fragment",
    "fetch_concept_region_fragment_and_reinforce",
    "reinforce_matched_concepts",
]
