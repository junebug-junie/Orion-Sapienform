"""Concept induction adapter — concept_induced tier.

Reads live ``ConceptNodeV1`` nodes from the substrate store's concept region
(``SubstrateGraphStore.query_concept_region``) — the concept graph populated
by golden seed concepts (``orion/substrate/seed.py``) and topic-foundry
derived concepts (``orion/substrate/adapters/topic_foundry.py``) — and maps
them into a ``SubstrateGraphRecordV1`` with concept_induced tier nodes.

This replaces the previous data source, the old spaCy-based
``orion.spark.concept_induction.profile_repository`` pipeline, which is dead
(``CONCEPT_AUTONOMOUS_TRIGGER_ENABLED=false``) and always yielded an empty
repository, so this adapter always returned ``None`` and nothing from the
concept substrate ever reached a live chat turn.

Filters to the four subjects ``chat_stance`` cares about (orion,
relationship, juniper, claude — matching this producer's own
``anchor_scopes`` registration) and derives ``metadata["concept_type"]``
bucketing from each node's ``anchor_scope`` when not already set, mirroring
the exact fallback precedent already established in
``chat_stance.py::_concept_summary_from_store``: only ``anchor_scope ==
"relationship"`` auto-routes to the relationship bucket downstream in
``_project_concept_from_beliefs`` (via ``anchor_key == "relationship"``); both
``orion`` and ``juniper`` subjects fall into the "self" bucket by default in
that same legacy function, so this adapter reproduces that mapping rather
than inventing a new one. ``claude`` (added 2026-08-22, once the golden seed
fixture's ``anchor_scope="claude"`` landed -- see
``orion/substrate/seed_concepts.yaml``) maps to "relationship" rather than
"self": Claude is a collaborator in Orion's Hub, not part of Orion's own
self-identity, and "relationship" is the closest fit among
``_project_concept_from_beliefs``'s fixed 4 buckets
(self/relationship/growth/tension) for "a collaborative dynamic with another
party" -- reused rather than inventing a 5th bucket for one subject.
Confirmed live 2026-08-22: without this, Claude's node was silently dropped
entirely (not merely misbucketed) by the ``_SUBJECTS`` filter below, so it
never reached chat_stance's concept summary at all.

Returns ``None`` for an empty result. Raises ``ProducerUnavailableError`` on a
store connectivity failure or degraded read (2026-10-06): the unification
layer now counts ``None`` as a fresh pull, so a failure must not look like one.
A malformed individual node is still skipped.
"""

from __future__ import annotations

import logging
from typing import Any

from orion.core.schemas.cognitive_substrate import SubstrateGraphRecordV1
from orion.substrate import build_substrate_store_from_env, select_concept_nodes_by_anchor_scope
from orion.substrate.relational.registry import ProducerUnavailableError
from orion.substrate.store import SubstrateGraphStore

logger = logging.getLogger("orion.substrate.relational.adapters.concept_induction_ctx")

_TIER_RANK = 3  # concept_induced
_SUBJECTS = ("orion", "relationship", "juniper", "claude")

# Fallback concept_type derived from a node's anchor_scope when the node has
# no explicit metadata["concept_type"]. Mirrors the subject-based bucketing
# precedent in chat_stance.py::_concept_summary_from_store: only
# "relationship" auto-routes to the relationship bucket; "orion" and
# "juniper" both default to "self". "claude" defaults to "relationship" --
# see this module's own docstring for why.
_ANCHOR_TO_CONCEPT_TYPE: dict[str, str] = {
    "orion": "self",
    "relationship": "relationship",
    "juniper": "self",
    "claude": "relationship",
}

_STORE: SubstrateGraphStore | None = None


def _get_store() -> SubstrateGraphStore:
    """Return (or lazily initialise) this module's own fallback substrate store.

    Only used when the caller did not bind a store (``store=None``) -- e.g.
    orion-cortex-orch's cold build, whose layer store is a fresh in-memory
    store with no concepts in it. The live stance layers bind their own store
    (see ``build_projection_unification_registry(concept_store=...)``), so
    this second Falkor connection is never opened on the chat path. A
    construction failure raises ``ProducerUnavailableError``.
    """
    global _STORE
    if _STORE is None:
        try:
            _STORE = build_substrate_store_from_env()
        except Exception as exc:
            raise ProducerUnavailableError(f"concept_induction store init failed: {exc}") from exc
    return _STORE


def map_concept_induction_ctx_to_substrate(
    ctx: dict[str, Any],  # noqa: ARG001
    *,
    store: SubstrateGraphStore | None = None,
) -> SubstrateGraphRecordV1 | None:
    """Fetch live concept-region nodes from the substrate store (concept_induced tier).

    ``store`` is the unification layer's own durable store, bound by the
    registry builder. ``query_concept_region`` reads that store's in-process
    cache, which the layer has just refreshed with its own ``snapshot()`` at
    the top of ``beliefs_for_stance`` -- so this costs no extra Falkor round
    trip and sees concepts written since boot.

    Before 2026-10-06 (unified-turn latency L6 step 2) this always used a
    private store that hydrated once at process boot and never called
    ``snapshot()`` again: every turn read boot-time concepts, and the
    write-through tier then re-saved those boot-time copies over the live
    graph. The unbound fallback now calls ``snapshot()`` first, which refreshes
    when a write landed or the store's refresh ceiling elapsed.
    """
    if store is None:
        store = _get_store()
        try:
            store.snapshot()
        except Exception as exc:
            raise ProducerUnavailableError(f"concept_induction snapshot failed: {exc}") from exc

    # Transient failures raise (the layer marks the producer degraded and
    # retries next turn); None means "reached the store, nothing to add".
    try:
        result = store.query_concept_region(limit_nodes=64, limit_edges=64)
    except Exception as exc:
        raise ProducerUnavailableError(f"concept_induction query failed: {exc}") from exc

    if result is not None and getattr(result, "degraded", False):
        raise ProducerUnavailableError(
            f"concept_induction query degraded: {getattr(result, 'error', None)}"
        )
    if result is None:
        return None

    region_slice = getattr(result, "slice", None)
    raw_nodes = list(getattr(region_slice, "nodes", None) or [])
    if not raw_nodes:
        return None

    by_subject = select_concept_nodes_by_anchor_scope(raw_nodes, list(_SUBJECTS))

    all_nodes: list[Any] = []
    for anchor, nodes_for_anchor in by_subject.items():
        for node in nodes_for_anchor:
            try:
                metadata = dict(node.metadata or {})
                if not str(metadata.get("concept_type") or "").strip():
                    metadata["concept_type"] = _ANCHOR_TO_CONCEPT_TYPE.get(anchor, "self")

                patched_prov = node.provenance.model_copy(update={"tier_rank": _TIER_RANK})
                all_nodes.append(node.model_copy(update={"provenance": patched_prov, "metadata": metadata}))
            except Exception as exc:
                logger.debug(
                    "concept_induction_ctx_node_map_failed node_id=%s error=%s",
                    getattr(node, "node_id", "?"),
                    exc,
                )
                continue

    return SubstrateGraphRecordV1(anchor_scope="orion", nodes=all_nodes) if all_nodes else None
