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

Filters to the three subjects ``chat_stance`` cares about (orion,
relationship, juniper — matching this producer's own ``anchor_scopes``
registration) and derives ``metadata["concept_type"]`` bucketing from each
node's ``anchor_scope`` when not already set, mirroring the exact fallback
precedent already established in
``chat_stance.py::_concept_summary_from_store``: only ``anchor_scope ==
"relationship"`` auto-routes to the relationship bucket downstream in
``_project_concept_from_beliefs`` (via ``anchor_key == "relationship"``); both
``orion`` and ``juniper`` subjects fall into the "self" bucket by default in
that same legacy function, so this adapter reproduces that mapping rather
than inventing a new one.

Never raises: degrades to ``None`` on any store connectivity failure,
malformed data, or empty result.
"""

from __future__ import annotations

import logging
from typing import Any

from orion.core.schemas.cognitive_substrate import SubstrateGraphRecordV1
from orion.substrate import build_substrate_store_from_env
from orion.substrate.store import SubstrateGraphStore

logger = logging.getLogger("orion.substrate.relational.adapters.concept_induction_ctx")

_TIER_RANK = 3  # concept_induced
_SUBJECTS = ("orion", "relationship", "juniper")

# Fallback concept_type derived from a node's anchor_scope when the node has
# no explicit metadata["concept_type"]. Mirrors the subject-based bucketing
# precedent in chat_stance.py::_concept_summary_from_store: only
# "relationship" auto-routes to the relationship bucket; "orion" and
# "juniper" both default to "self".
_ANCHOR_TO_CONCEPT_TYPE: dict[str, str] = {
    "orion": "self",
    "relationship": "relationship",
    "juniper": "self",
}

_STORE: SubstrateGraphStore | None = None


def _get_store() -> SubstrateGraphStore | None:
    """Return (or lazily initialise) the process-level substrate store.

    Mirrors the singleton-caching pattern in
    ``services/orion-cortex-exec/app/chat_stance.py::_get_unification_layer``
    without importing from that module — ``chat_stance.py`` lazily imports
    this adapter inside ``_build_unification_registry``, so importing back
    from here would be circular. Never raises: a construction failure is
    logged and the caller degrades to ``None``.
    """
    global _STORE
    if _STORE is None:
        try:
            _STORE = build_substrate_store_from_env()
        except Exception as exc:
            logger.debug("concept_induction_ctx_store_init_failed error=%s", exc)
            return None
    return _STORE


def map_concept_induction_ctx_to_substrate(ctx: dict[str, Any]) -> SubstrateGraphRecordV1 | None:  # noqa: ARG001
    """Fetch live concept-region nodes from the substrate store (concept_induced tier)."""
    store = _get_store()
    if store is None:
        return None

    try:
        result = store.query_concept_region(limit_nodes=64, limit_edges=64)
    except Exception as exc:
        logger.debug("concept_induction_ctx_query_failed error=%s", exc)
        return None

    if result is None or getattr(result, "degraded", False):
        return None

    region_slice = getattr(result, "slice", None)
    raw_nodes = list(getattr(region_slice, "nodes", None) or [])
    if not raw_nodes:
        return None

    all_nodes: list[Any] = []
    for node in raw_nodes:
        try:
            if getattr(node, "node_kind", None) != "concept":
                continue
            anchor = getattr(node, "anchor_scope", None)
            if anchor not in _SUBJECTS:
                continue

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
