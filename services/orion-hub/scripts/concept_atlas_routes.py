"""Concept Atlas — read-only interpretability routes for the substrate concept graph.

Phase 8 of the concept-graph-pipeline design
(``docs/superpowers/specs/2026-07-15-concept-atlas-graph-pipeline-design.md``).

This module owns exactly two GET routes plus the standalone page route for the
iframe-embedded Concept Atlas Hub tab. It deliberately does not construct its
own substrate store instance: ``scripts.api_routes`` already builds one shared
``SUBSTRATE_SEMANTIC_STORE`` at import time via ``build_substrate_store_from_env()``
(see ``services/orion-hub/scripts/api_routes.py``), and ``memory_graph_routes.py``
already establishes the convention of importing ``scripts.api_routes`` inside
route functions (deferred import) rather than at module load time to reuse that
shared instance. This module follows the same convention.

If that store is ever not attached / not importable, every route here degrades
to an honest "unavailable" response instead of fabricating data or raising a
500 — this is a debug/interpretability surface, not a critical path.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse

from .settings import settings
from .topic_foundry_client import TopicFoundryClientError, fetch_run_topics_and_keywords

logger = logging.getLogger("orion-hub.concept_atlas")

router = APIRouter(tags=["concept-atlas"])

# Top-N nodes by (activation-weighted) degree flagged as "god nodes" in the
# network response. The store is small and in-memory (per the design spec's
# explicit non-goal of no precomputed ranking job), so this is computed fresh
# on every request rather than cached.
_GOD_NODE_TOP_N = 5

_VALID_ANCHOR_SCOPES = {"orion", "juniper", "relationship", "world", "session"}
# promotion_state is not a network-route query param per the design spec's
# route contract (scope/min_activation/focus only); the Concept Atlas UI
# applies its promotion_state global filter client-side against this
# endpoint's response instead of round-tripping it server-side.

# Concept nodes sitting within this margin of their own decay_floor are
# treated as "at risk" in the summary response. This is only meaningful once
# a node's activation has moved off its schema default of 0.0 — see the
# comment in ``_at_risk_concepts`` below.
_AT_RISK_MARGIN = 0.05


def _get_substrate_store() -> Any:
    """Best-effort resolution of the shared substrate store, never raises.

    Deferred import mirrors ``memory_graph_routes.py``'s established pattern
    (``import scripts.api_routes as api_mod`` inside route functions) so this
    module does not force a heavy import of ``scripts.api_routes`` at module
    load time -- useful for isolated router tests, and consistent with the
    existing convention rather than inventing a new one.
    """
    try:
        import scripts.api_routes as api_mod
    except Exception as exc:  # pragma: no cover - defensive, mirrors route-level degrade
        logger.warning("concept_atlas_store_import_failed error=%s", exc)
        return None
    store = getattr(api_mod, "SUBSTRATE_SEMANTIC_STORE", None)
    if store is None:
        logger.info("concept_atlas_store_not_attached")
    return store


def _unavailable(reason: str, error: Optional[str] = None, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"available": False, "reason": reason}
    if error:
        payload["error"] = error
    payload.update(extra)
    return payload


class _CountingSubstrateStore:
    """Thin store wrapper that counts successful concept/evidence/edge upserts.

    ``SubstrateGraphMaterializer.apply_record`` writes incrementally and raises
    without returning a partial ``MaterializationResultV1``. Delegating through
    this wrapper lets the ingest route report precise successful counts on
    mid-record failure, while preserving all other store operations (including
    identity-resolver reads) via ``__getattr__``.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.concepts_written = 0
        self.evidence_nodes_written = 0
        self.edges_written = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def upsert_node(self, *, identity_key: str | None, node: Any) -> None:
        self._inner.upsert_node(identity_key=identity_key, node=node)
        kind = getattr(node, "node_kind", None)
        if kind == "concept":
            self.concepts_written += 1
        elif kind == "evidence":
            self.evidence_nodes_written += 1

    def upsert_edge(self, *, identity_key: str, edge: Any) -> None:
        self._inner.upsert_edge(identity_key=identity_key, edge=edge)
        self.edges_written += 1


def _concept_nodes(store: Any) -> list[Any]:
    """All concept-kind nodes from the store's snapshot, or [] on any failure."""
    try:
        snapshot = store.snapshot()
    except Exception as exc:
        logger.warning("concept_atlas_snapshot_failed error=%s", exc)
        return []
    return [n for n in snapshot.nodes.values() if getattr(n, "node_kind", None) == "concept"]


def _at_risk_concepts(concept_nodes: list[Any]) -> tuple[list[dict[str, Any]], Optional[str]]:
    """Concepts whose activation is decaying toward their decay_floor.

    Honesty note: ``SubstrateActivationV1.activation`` defaults to 0.0 and no
    live writer currently drives concept-node decay in the default
    ``InMemorySubstrateGraphStore`` path (per the design spec's Current
    architecture section: "present in schema, no confirmed live writer").
    Returning every node at the default as "at risk" would be a fabricated
    signal, not a real one. So: if every concept node's activation is exactly
    the same value (almost always 0.0 today), this returns an empty list and
    an explanatory note instead of pretending the field carries real
    information yet.
    """
    if not concept_nodes:
        return [], None
    activations = {float(n.signals.activation.activation) for n in concept_nodes}
    if len(activations) <= 1:
        return [], (
            "activation shows no variance across concept nodes yet (no live decay "
            "writer wired to this store today) -- at_risk is intentionally empty "
            "rather than fabricated"
        )
    at_risk: list[dict[str, Any]] = []
    for n in concept_nodes:
        act = n.signals.activation
        if float(act.activation) <= float(act.decay_floor) + _AT_RISK_MARGIN:
            at_risk.append(
                {
                    "node_id": n.node_id,
                    "label": getattr(n, "label", n.node_id),
                    "activation": float(act.activation),
                    "decay_floor": float(act.decay_floor),
                    "promotion_state": n.promotion_state,
                }
            )
    at_risk.sort(key=lambda row: row["activation"])
    return at_risk[:20], None


@router.get("/api/substrate/concepts/summary")
async def concept_atlas_summary() -> dict[str, Any]:
    store = _get_substrate_store()
    if store is None:
        return _unavailable(
            "substrate_store_unavailable",
            total_concepts=0,
            by_promotion_state={},
            by_anchor_scope={},
            edge_counts_by_predicate={},
            at_risk=[],
        )

    concept_nodes = _concept_nodes(store)

    by_promotion_state: dict[str, int] = {}
    by_anchor_scope: dict[str, int] = {}
    for node in concept_nodes:
        by_promotion_state[node.promotion_state] = by_promotion_state.get(node.promotion_state, 0) + 1
        by_anchor_scope[node.anchor_scope] = by_anchor_scope.get(node.anchor_scope, 0) + 1

    edge_counts_by_predicate: dict[str, int] = {}
    try:
        snapshot = store.snapshot()
        concept_ids = {n.node_id for n in concept_nodes}
        for edge in snapshot.edges.values():
            if edge.source.node_id in concept_ids or edge.target.node_id in concept_ids:
                edge_counts_by_predicate[edge.predicate] = edge_counts_by_predicate.get(edge.predicate, 0) + 1
    except Exception as exc:
        logger.warning("concept_atlas_summary_edge_scan_failed error=%s", exc)

    at_risk, at_risk_note = _at_risk_concepts(concept_nodes)

    return {
        "available": True,
        "total_concepts": len(concept_nodes),
        "by_promotion_state": by_promotion_state,
        "by_anchor_scope": by_anchor_scope,
        "edge_counts_by_predicate": edge_counts_by_predicate,
        "at_risk": at_risk,
        "at_risk_note": at_risk_note,
    }


@router.get("/api/substrate/concepts/network")
async def concept_atlas_network(
    scope: Optional[str] = Query(None),
    min_activation: Optional[str] = Query(None),
    focus: Optional[str] = Query(None),
) -> dict[str, Any]:
    store = _get_substrate_store()
    if store is None:
        return _unavailable("substrate_store_unavailable", nodes=[], edges=[], god_node_count=0)

    try:
        result = store.query_concept_region(limit_nodes=300, limit_edges=600)
    except Exception as exc:
        logger.warning("concept_atlas_network_query_failed error=%s", exc)
        return _unavailable("substrate_store_error", str(exc), nodes=[], edges=[], god_node_count=0)

    nodes = list(result.slice.nodes)
    edges = list(result.slice.edges)

    # --- filters: degrade sanely on malformed input, never 500 ---
    scope_norm = scope.strip() if isinstance(scope, str) else ""
    if scope_norm and scope_norm in _VALID_ANCHOR_SCOPES:
        nodes = [n for n in nodes if n.anchor_scope == scope_norm]
    elif scope_norm:
        logger.info("concept_atlas_network_ignored_bad_scope value=%s", scope_norm)

    min_act_value: Optional[float] = None
    if isinstance(min_activation, str) and min_activation.strip():
        try:
            parsed_min_act = float(min_activation)
        except ValueError:
            parsed_min_act = None
        # float("nan")/float("inf") parse without raising ValueError but are
        # not a meaningful activation threshold -- nan comparisons are always
        # False (silently empties the graph) and treating them as malformed
        # (i.e. ignored) is more honest than a confusing empty result.
        if parsed_min_act is None or not (0.0 <= parsed_min_act <= 1.0):
            logger.info("concept_atlas_network_ignored_bad_min_activation value=%s", min_activation)
        else:
            min_act_value = parsed_min_act
    if min_act_value is not None:
        nodes = [n for n in nodes if float(n.signals.activation.activation) >= min_act_value]

    node_ids = {n.node_id for n in nodes}
    edges = [e for e in edges if e.source.node_id in node_ids and e.target.node_id in node_ids]

    focus_norm = focus.strip() if isinstance(focus, str) else ""
    if focus_norm:
        focus_lower = focus_norm.lower()
        focus_id = next(
            (
                n.node_id
                for n in nodes
                if n.node_id == focus_norm or str(getattr(n, "label", "") or "").lower() == focus_lower
            ),
            None,
        )
        if focus_id is not None:
            neighbor_ids = {focus_id}
            for e in edges:
                if e.source.node_id == focus_id:
                    neighbor_ids.add(e.target.node_id)
                if e.target.node_id == focus_id:
                    neighbor_ids.add(e.source.node_id)
            nodes = [n for n in nodes if n.node_id in neighbor_ids]
            node_ids = {n.node_id for n in nodes}
            edges = [e for e in edges if e.source.node_id in node_ids and e.target.node_id in node_ids]
        else:
            logger.info("concept_atlas_network_focus_not_found value=%s", focus_norm)

    # --- degree (activation-weighted) computed fresh at request time ---
    degree: dict[str, float] = {n.node_id: 0.0 for n in nodes}
    for e in edges:
        weight = 1.0 + float(e.salience or 0.0)
        if e.source.node_id in degree:
            degree[e.source.node_id] += weight
        if e.target.node_id in degree:
            degree[e.target.node_id] += weight

    ranked = sorted((nid for nid in degree if degree[nid] > 0.0), key=lambda nid: degree[nid], reverse=True)
    god_ids = set(ranked[:_GOD_NODE_TOP_N])

    node_payload = [
        {
            "id": n.node_id,
            "label": getattr(n, "label", n.node_id),
            "node_kind": n.node_kind,
            "anchor_scope": n.anchor_scope,
            "promotion_state": n.promotion_state,
            "activation": float(n.signals.activation.activation),
            "salience": float(n.signals.salience),
            "confidence": float(n.signals.confidence),
            "degree": degree.get(n.node_id, 0.0),
            "god_node": n.node_id in god_ids,
        }
        for n in nodes
    ]
    edge_payload = [
        {
            "id": e.edge_id,
            "source": e.source.node_id,
            "target": e.target.node_id,
            "predicate": e.predicate,
            "confidence": float(e.confidence),
            "salience": float(e.salience),
        }
        for e in edges
    ]

    # Non-default backends (e.g. GraphDBSubstrateStore under
    # SUBSTRATE_STORE_BACKEND=graphdb) can return a non-raising result that is
    # nonetheless backed by a stale fallback snapshot after an upstream query
    # failure (see query_concept_region()'s own degraded/error fields). The
    # default in-memory store never sets these, but surface them honestly
    # when a backend does -- "available: true" alone would otherwise hide a
    # real degraded-data condition (runtime truth beats config truth).
    degraded = bool(getattr(result, "degraded", False))
    return {
        "available": True,
        "nodes": node_payload,
        "edges": edge_payload,
        "god_node_count": len(god_ids),
        "truncated": bool(result.truncated),
        "degraded": degraded,
        "degraded_error": getattr(result, "error", None) if degraded else None,
    }


@router.post("/api/substrate/concepts/ingest-topic-foundry")
def concept_atlas_ingest_topic_foundry() -> dict[str, Any]:
    """Operator-triggered, on-demand ingestion of topic-foundry's latest completed run.

    Deliberately a sync ``def`` (not ``async def``): the underlying HTTP calls
    use the blocking ``requests`` library (see ``topic_foundry_client.py``),
    and FastAPI runs sync route handlers in a worker thread pool automatically
    -- an ``async def`` wrapper here would block the event loop for the
    duration of every topic-foundry round trip instead.

    Fetches the latest completed run + its topics + per-topic keywords from
    topic-foundry over HTTP, converts them via
    ``orion.substrate.adapters.topic_foundry.map_topic_foundry_run_to_substrate``,
    and writes the resulting concept/evidence nodes and edges into the shared
    ``SUBSTRATE_SEMANTIC_STORE``.

    This is a manual trigger only -- not wired into Hub's startup event or any
    scheduler, mirroring ``/api/substrate/review-runtime/debug-run``'s shape
    (operator-triggered multi-step pipeline, not an automatic background job).
    Every failure mode below degrades to an honest, non-500 response rather
    than fabricating a success result.

    Scoping note: ``segment_topic_map`` is passed as empty in this first cut,
    so no ``co_occurs_with`` co-occurrence edges are produced yet -- only
    concept nodes (+ backing evidence nodes/edges) per real topic. See the PR
    report for why this was scoped out rather than attempted.
    """
    store = _get_substrate_store()
    if store is None:
        return _unavailable("substrate_store_unavailable", concepts_written=0, edges_written=0)

    base_url = str(getattr(settings, "TOPIC_FOUNDRY_BASE_URL", "") or "").strip()
    if not base_url:
        return _unavailable(
            "topic_foundry_base_url_not_configured",
            concepts_written=0,
            edges_written=0,
        )

    try:
        fetched = fetch_run_topics_and_keywords(base_url)
    except TopicFoundryClientError as exc:
        logger.warning("concept_atlas_ingest_topic_foundry_fetch_failed error=%s", exc)
        return _unavailable(
            "topic_foundry_fetch_failed",
            str(exc),
            concepts_written=0,
            edges_written=0,
        )
    except Exception as exc:  # pragma: no cover - defensive, never let a client bug 500 this route
        logger.warning("concept_atlas_ingest_topic_foundry_unexpected_fetch_error error=%s", exc)
        return _unavailable(
            "topic_foundry_unexpected_error",
            str(exc),
            concepts_written=0,
            edges_written=0,
        )

    run_id = fetched["run_id"]
    topics = fetched["topics"]
    keywords_by_topic = fetched["keywords_by_topic"]

    try:
        from orion.substrate.adapters.topic_foundry import map_topic_foundry_run_to_substrate

        record = map_topic_foundry_run_to_substrate(
            run_id=run_id,
            topics=topics,
            keywords_by_topic=keywords_by_topic,
            segment_topic_map={},
        )
    except Exception as exc:  # pragma: no cover - the adapter itself never raises, but don't trust across the boundary
        logger.warning("concept_atlas_ingest_topic_foundry_adapter_failed run_id=%s error=%s", run_id, exc)
        return _unavailable(
            "topic_foundry_adapter_failed",
            str(exc),
            run_id=run_id,
            concepts_written=0,
            edges_written=0,
        )

    if not record.nodes:
        return _unavailable(
            "topic_foundry_no_usable_topics",
            run_id=run_id,
            topics_fetched=len(topics),
            concepts_written=0,
            edges_written=0,
        )

    counting_store = _CountingSubstrateStore(store)
    try:
        from orion.substrate.materializer import SubstrateGraphMaterializer
        from orion.substrate.reconcile import SubstrateIdentityResolver

        # Exact-label identity works through the resolver's legacy key path;
        # explicitly wiring the store also activates embedding matches whenever
        # a caller supplies metadata['concept_embedding'].
        materializer = SubstrateGraphMaterializer(
            store=counting_store,
            identity_resolver=SubstrateIdentityResolver(store=counting_store),
        )
        result = materializer.apply_record(record)
    except Exception as exc:
        logger.warning("concept_atlas_ingest_topic_foundry_store_write_failed run_id=%s error=%s", run_id, exc)
        return _unavailable(
            "substrate_store_write_failed",
            str(exc),
            run_id=run_id,
            concepts_written=counting_store.concepts_written,
            evidence_nodes_written=counting_store.evidence_nodes_written,
            edges_written=counting_store.edges_written,
        )

    # Response fields remain total record items processed/written (not unique
    # durable nodes after merge). Materializer edges_seen matches that meaning.
    concept_count = sum(1 for n in record.nodes if n.node_kind == "concept")
    evidence_count = sum(1 for n in record.nodes if n.node_kind == "evidence")
    edge_count = result.edges_seen

    return {
        "available": True,
        "run_id": run_id,
        "topics_fetched": len(topics),
        "concepts_written": concept_count,
        "evidence_nodes_written": evidence_count,
        "edges_written": edge_count,
        "co_occurrence_edges_skipped": "segment_topic_map not supplied in this ingestion route (empty by design)",
    }


@router.get("/concept-atlas")
async def concept_atlas_page() -> HTMLResponse:
    from .main import TEMPLATES_DIR, build_hub_ui_asset_version

    template_path = TEMPLATES_DIR / "concept_atlas.html"
    if not template_path.is_file():
        raise HTTPException(status_code=404, detail="concept_atlas_template_missing")
    template = template_path.read_text(encoding="utf-8")
    rendered = template.replace("{{HUB_UI_ASSET_VERSION}}", build_hub_ui_asset_version())
    return HTMLResponse(
        content=rendered,
        status_code=200,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )
