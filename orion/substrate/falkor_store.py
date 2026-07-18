"""FalkorDB-backed SubstrateGraphStore (write-through cache + Cypher-native properties).

Queries and reads are served from the in-process cache (same shape as a warm
GraphDBSubstrateStore). Durable writes go through an injectable sync client so
unit tests never need a live FalkorDB.

Durable support is Concept + Evidence + SubstrateEdge. Cold-start hydration
prefers native scalar properties and falls back to legacy ``payload_json``
rows, rewriting them to native properties (and removing the blob) on
successful concept/edge decode.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from dataclasses import dataclass, replace
from typing import Any
from urllib.parse import urlparse

from pydantic import TypeAdapter

from orion.core.schemas.cognitive_substrate import (
    BaseSubstrateNodeV1,
    SubstrateEdgeV1,
    SubstrateNodeV1,
)
from orion.graph.falkor_client import (
    FalkorGraphClient,
    RecordingFalkorClient,
    RedisGraphQueryClient,
    _header_field_names,
    _rows_from_query_result,
)
from orion.graph.property_guard import sanitize_metadata
from orion.substrate.falkor_codec import (
    decode_edge,
    decode_node,
    encode_edge_properties,
    encode_node_properties,
    node_label_for_kind,
)
from orion.substrate.store import (
    InMemorySubstrateGraphStore,
    MaterializedSubstrateGraphState,
    SubstrateNeighborhoodSliceV1,
    SubstrateQueryResultV1,
)
from orion.substrate.graphdb_store import _resolve_snapshot_force_refresh_ceiling_sec

logger = logging.getLogger("orion.substrate.falkor_store")

NODE_ADAPTER = TypeAdapter(SubstrateNodeV1)

# FalkorGraphClient, RecordingFalkorClient, RedisGraphQueryClient moved to
# orion.graph.falkor_client (2026-07-18, zero substrate coupling) -- imported
# above and re-exported here so existing `from orion.substrate.falkor_store
# import RecordingFalkorClient, RedisGraphQueryClient` call sites (this
# module's own test suite) keep working unchanged.

__all__ = [
    "FalkorGraphClient",
    "RecordingFalkorClient",
    "RedisGraphQueryClient",
    "FalkorSubstrateStoreConfig",
    "FalkorSubstrateStore",
]


@dataclass(frozen=True)
class FalkorSubstrateStoreConfig:
    uri: str
    graph_name: str = "orion_substrate"
    # Same knob name and same generation+ceiling GATING logic as
    # GraphDBSubstrateStoreConfig's field of this name (see that class's
    # docstring) -- real change detection via a same-process write-
    # generation counter, with this as the periodic safety-net ceiling
    # bounding staleness from writes this process can't see (a different
    # process's write, or a direct external mutation like an operator
    # running Cypher DELETE by hand). Without any periodic refresh, a direct
    # external deletion is invisible to this cache forever, AND the decay
    # scheduler (services/orion-hub/scripts/api_routes.py::
    # decay_concept_activations) durably re-upserts every node in every
    # snapshot() it reads on every tick -- so a stale cache doesn't just
    # show old data, it actively resurrects deleted durable data on the next
    # tick. Confirmed live: this exact resurrection loop was observed and
    # root-caused in production.
    #
    # NOT the same as GraphDB in two respects, both worth knowing:
    # (1) GraphDBSubstrateStore's own generation+ceiling refresh is upsert-
    #     only (never removes cache entries for durably-deleted nodes) and
    #     it never does an eager hydration at construction -- so GraphDB
    #     likely still has an equivalent resurrection exposure of its own;
    #     porting this fix here does not imply GraphDB is already immune.
    #     The fresh-cache-swap idea (see _hydrate_from_durable) is what
    #     actually fixes deletion-visibility and is new to this store, not
    #     ported from GraphDB.
    # (2) This store's hydrate queries have no LIMIT (GraphDB's are capped
    #     at 500 nodes / 1000 edges), and same_generation is invalidated by
    #     ANY write anywhere -- so a caller that both reads and writes every
    #     tick (the decay scheduler above is exactly this shape: one
    #     snapshot() read, then N upsert_node() writes) will see a
    #     generation mismatch on its OWN next tick's read, forcing a full
    #     unbounded re-hydration on very close to every tick regardless of
    #     this ceiling. This is an accepted, understood tradeoff at current
    #     graph sizes (tens of nodes) -- correctness over cache efficiency
    #     for this caller shape -- not a bug, but worth revisiting (a LIMIT,
    #     or excluding a caller's own immediately-prior writes from
    #     generation invalidation) if it ever shows up as real cost.
    #
    # See also FALKOR_SNAPSHOT_FORCE_REFRESH_CEILING_SEC (falls back to this
    # shared SUBSTRATE_SNAPSHOT_FORCE_REFRESH_CEILING_SEC when unset) for an
    # independent override -- RoutedSubstrateGraphStore can run both a
    # GraphDB-backed and a Falkor-backed store concurrently in one process
    # (primary + shadow), and given point (2) above, the two backends' real
    # refresh costs are not symmetric.
    snapshot_force_refresh_ceiling_sec: float = 30.0


NATIVE_NODE_RETURN_FIELDS: tuple[str, ...] = (
    "node_id",
    "node_kind",
    "identity_key",
    "label",
    "definition",
    "taxonomy_path_json",
    "evidence_type",
    "content_ref",
    "anchor_scope",
    "subject_ref",
    "promotion_state",
    "risk_tier",
    "confidence",
    "salience",
    "activation",
    "recency_score",
    "decay_half_life_seconds",
    "decay_floor",
    "observed_at",
    "valid_from",
    "valid_to",
    "provenance_authority",
    "provenance_source_kind",
    "provenance_source_channel",
    "provenance_producer",
    "provenance_model_name",
    "provenance_correlation_id",
    "provenance_trace_id",
    "provenance_tier_rank",
    "evidence_refs_json",
    "dynamic_pressure",
    "dynamic_pressure_reason",
    "dormant",
    "dormancy_updated_at",
    "prediction_error",
)

NATIVE_EDGE_RETURN_FIELDS: tuple[str, ...] = (
    "edge_id",
    "identity_key",
    "source_id",
    "source_kind",
    "target_id",
    "target_kind",
    "predicate",
    "substrate_edge",
    "confidence",
    "salience",
    "observed_at",
    "valid_from",
    "valid_to",
    "provenance_authority",
    "provenance_source_kind",
    "provenance_source_channel",
    "provenance_producer",
    "provenance_model_name",
    "provenance_correlation_id",
    "provenance_trace_id",
    "provenance_tier_rank",
    "evidence_refs_json",
)


def _return_clause(alias: str, fields: tuple[str, ...]) -> str:
    return ", ".join(f"{alias}.{field} AS {field}" for field in fields)


def _set_assignments(alias: str, params: dict[str, Any], *, skip: set[str]) -> str:
    keys = sorted(k for k in params if k not in skip)
    return ", ".join(f"{alias}.{key} = ${key}" for key in keys)


### RecordingFalkorClient / RedisGraphQueryClient / _header_field_names /
### _rows_from_query_result moved to orion.graph.falkor_client (2026-07-18),
### imported at the top of this file and re-exported via __all__.


def _with_sanitized_metadata(model: Any) -> Any:
    cleaned, _rejected = sanitize_metadata(getattr(model, "metadata", None) or {})
    if cleaned == (getattr(model, "metadata", None) or {}):
        return model
    return model.model_copy(update={"metadata": cleaned})


class FalkorSubstrateStore:
    """SubstrateGraphStore with Falkor write-through and in-memory read cache.

    Durable persistence is Concept + Evidence + SubstrateEdge only. Node
    upserts for any other node kind raise ``ValueError`` rather than writing
    incomplete native rows.
    """

    def __init__(
        self,
        cfg: FalkorSubstrateStoreConfig,
        *,
        client: FalkorGraphClient | None = None,
        hydrate: bool = True,
    ) -> None:
        self._cfg = cfg
        self._client: FalkorGraphClient = client or RedisGraphQueryClient(
            uri=cfg.uri, graph_name=cfg.graph_name
        )
        self._cache = InMemorySubstrateGraphStore()
        self._result_source_kind = "falkor"
        # See FalkorSubstrateStoreConfig.snapshot_force_refresh_ceiling_sec's
        # docstring for the full reasoning -- mirrors GraphDBSubstrateStore's
        # already-proven write-generation + ceiling mechanism exactly.
        self._write_generation = 0
        self._last_snapshot_at: float | None = None
        # -1 sentinel: guarantees the first snapshot() call always sees
        # same_generation=False (0 == -1 is never true), so no separate
        # "first call" branch is needed in snapshot() itself.
        self._last_snapshot_generation = -1
        self._snapshot_lock = threading.Lock()
        if hydrate:
            self._hydrate_from_durable()
            # Record that construction already did a full hydration, so the
            # first real snapshot() call doesn't immediately redo it. Without
            # this, the -1 sentinel above would force a second, wasted
            # refresh on literally the next call (GraphDBSubstrateStore
            # doesn't have this wrinkle because it never hydrates eagerly at
            # construction -- its first snapshot() call is genuinely the
            # first fetch, not a redundant second one).
            self._last_snapshot_at = time.monotonic()
            self._last_snapshot_generation = self._write_generation

    def _hydrate_from_durable(self) -> None:
        """(Re)populate the in-process cache from durable Falkor state.

        Called once at construction, and again periodically by snapshot()'s
        refresh path (see FalkorSubstrateStoreConfig.snapshot_force_refresh_ceiling_sec).
        Builds a FRESH InMemorySubstrateGraphStore and swaps it in wholesale
        rather than upserting into the existing self._cache in place -- this
        is the actual fix for the resurrection bug: an in-place upsert can
        only ever add/update nodes that still exist durably, it can never
        remove a node that was deleted directly in Falkor (bypassing this
        process). A fresh cache, containing only what's durably present
        right now, correctly reflects deletions on every refresh.

        On query failure, returns without touching self._cache at all --
        the stale-but-still-valid existing cache is a strictly better
        fallback than an empty one.
        """
        try:
            node_rows = self._client.graph_query(
                "MATCH (n:SubstrateNode) RETURN " + _return_clause("n", NATIVE_NODE_RETURN_FIELDS)
            )
            edge_rows = self._client.graph_query(
                "MATCH (source:SubstrateNode)-[e]->(target:SubstrateNode) "
                "WHERE e.substrate_edge = true "
                "RETURN " + _return_clause("e", NATIVE_EDGE_RETURN_FIELDS)
            )
            legacy_node_rows = self._client.graph_query(
                "MATCH (n:SubstrateNode) WHERE n.payload_json IS NOT NULL "
                "RETURN n.payload_json AS payload_json, n.identity_key AS identity_key"
            )
            legacy_edge_rows = self._client.graph_query(
                "MATCH ()-[e]->() WHERE e.payload_json IS NOT NULL "
                "RETURN e.payload_json AS payload_json, e.identity_key AS identity_key"
            )
        except Exception as exc:
            logger.warning("falkor_substrate_hydrate_failed error=%s", exc)
            return

        fresh_cache = InMemorySubstrateGraphStore()
        for row in _normalize_rows(node_rows, fields=NATIVE_NODE_RETURN_FIELDS):
            try:
                node = decode_node(row)
            except Exception:
                logger.warning("falkor_substrate_hydrate_node_invalid")
                continue
            if node is None:
                continue
            identity = row.get("identity_key")
            fresh_cache.upsert_node(identity_key=str(identity) if identity else None, node=node)

        for row in _normalize_rows(edge_rows, fields=NATIVE_EDGE_RETURN_FIELDS):
            try:
                edge = decode_edge(row)
            except Exception:
                logger.warning("falkor_substrate_hydrate_edge_invalid")
                continue
            if edge is None:
                continue
            identity = row.get("identity_key") or self._edge_identity(edge)
            fresh_cache.upsert_edge(identity_key=str(identity), edge=edge)

        # Swap now, before legacy migration -- _migrate_legacy_payload_*
        # below call self.upsert_node()/self.upsert_edge() (the full public
        # write path, which also bumps self._write_generation), and must
        # operate against the current cache, not a soon-to-be-discarded one.
        #
        # Known, bounded race: upsert_node()/upsert_edge() are deliberately
        # not protected by _snapshot_lock (matching GraphDBSubstrateStore's
        # own precedent), so a concurrent upsert_node() call whose
        # `self._cache.upsert_node(...)` step interleaves with this swap
        # could land its write on the old (about to be discarded) cache
        # object instead of `fresh_cache` -- that write would be briefly
        # invisible to snapshot(). It self-heals: the write still bumped
        # self._write_generation, so the very next snapshot() call detects
        # the mismatch and re-fetches, which then sees the write durably
        # reflected in Falkor. Not permanent loss, just a narrow visibility
        # delay -- and this window scales with fresh_cache's build time,
        # which is currently unbounded (see the LIMIT tradeoff noted on
        # FalkorSubstrateStoreConfig.snapshot_force_refresh_ceiling_sec).
        self._cache = fresh_cache

        self._migrate_legacy_payload_nodes(
            _normalize_rows(legacy_node_rows, fields=("payload_json", "identity_key"))
        )
        self._migrate_legacy_payload_edges(
            _normalize_rows(legacy_edge_rows, fields=("payload_json", "identity_key"))
        )

    def _migrate_legacy_payload_nodes(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            payload = row.get("payload_json")
            if not payload:
                continue
            try:
                node = NODE_ADAPTER.validate_json(payload)
            except Exception:
                logger.warning("falkor_substrate_legacy_node_invalid")
                continue
            if getattr(node, "node_kind", None) not in ("concept", "evidence"):
                logger.warning(
                    "falkor_substrate_legacy_node_skipped node_kind=%s node_id=%s",
                    getattr(node, "node_kind", None),
                    getattr(node, "node_id", None),
                )
                continue
            identity = row.get("identity_key")
            identity_key = str(identity) if identity else None
            # Seed cache first so a transient rewrite failure cannot empty Atlas.
            self._cache.upsert_node(identity_key=identity_key, node=node)
            try:
                # Rewrite to native properties. upsert_node()'s MERGE keys on
                # (SubstrateNode:<type-label> {node_id}) -- a label pattern
                # this legacy row (labeled SubstrateNode only, no type label
                # yet) can never itself satisfy. So the write always lands on
                # a *different*, already-migrated node (or creates a fresh
                # one) instead of converting this row in place. Confirmed
                # live (2026-07-18): this left a permanent orphaned duplicate
                # for every node this path ever touched -- the orphaned row's
                # payload_json got re-parsed and re-clobbered the canonical
                # node's real data on every subsequent hydrate, forever
                # (this is what silently reverted PR #1173's golden-concept
                # salience fix within one restart cycle). It also cascaded
                # into duplicate edges: an edge's own MERGE binds its
                # source/target via the bare `:SubstrateNode` label, which is
                # ambiguous between the legacy and canonical node rows until
                # the legacy one is gone -- confirmed live, one relationship
                # existed as 4 near-identical copies (one per source/target
                # duplicate-row combination).
                self.upsert_node(identity_key=identity_key, node=node)
                self._delete_orphaned_legacy_node_duplicate(node.node_id)
                logger.info(
                    "falkor_substrate_legacy_node_migrated node_id=%s identity_key=%s",
                    node.node_id,
                    identity_key or "",
                )
            except Exception as exc:
                logger.warning(
                    "falkor_substrate_legacy_node_migrate_failed node_id=%s error=%s",
                    getattr(node, "node_id", None),
                    exc,
                )

    def _delete_orphaned_legacy_node_duplicate(self, node_id: str) -> None:
        """Remove any *other* SubstrateNode sharing `node_id` that still
        carries a legacy payload_json, after the canonical node has just
        been migrated (see the comment in `_migrate_legacy_payload_nodes`).

        `upsert_node()`'s type-labeled MERGE can never match the un-migrated
        legacy row itself, so that row would otherwise never be touched by
        the migration write and would persist forever. The canonical node
        this method runs after has already had `payload_json` removed by
        `upsert_node()`'s own SET clause, so this can never match/delete it
        -- only a genuinely orphaned duplicate matches
        `payload_json IS NOT NULL` at this point. `DETACH DELETE` also
        removes any relationships still attached to the orphaned row, which
        is what cleans up the cascaded duplicate-edge side effect described
        above without needing separate edge-dedup logic.

        Deliberately does not catch its own exceptions: a failed cleanup
        reproduces exactly the bug this method exists to fix (an orphaned
        row that keeps re-clobbering the canonical node on every future
        hydrate), so it must be indistinguishable from any other migration
        failure to the caller -- letting it propagate into
        `_migrate_legacy_payload_nodes`'s existing `except` block means it's
        logged as `falkor_substrate_legacy_node_migrate_failed`, not
        silently swallowed under a `..._migrated` success log line.
        """
        self._client.graph_query(
            "MATCH (n:SubstrateNode {node_id: $node_id}) "
            "WHERE n.payload_json IS NOT NULL "
            "DETACH DELETE n",
            {"node_id": node_id},
        )

    def _migrate_legacy_payload_edges(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            payload = row.get("payload_json")
            if not payload:
                continue
            try:
                edge = SubstrateEdgeV1.model_validate_json(payload)
            except Exception:
                logger.warning("falkor_substrate_legacy_edge_invalid")
                continue
            identity = row.get("identity_key") or self._edge_identity(edge)
            self._cache.upsert_edge(identity_key=str(identity), edge=edge)
            try:
                self.upsert_edge(identity_key=str(identity), edge=edge)
                logger.info(
                    "falkor_substrate_legacy_edge_migrated edge_id=%s identity_key=%s",
                    edge.edge_id,
                    identity,
                )
            except Exception as exc:
                logger.warning(
                    "falkor_substrate_legacy_edge_migrate_failed edge_id=%s error=%s",
                    edge.edge_id,
                    exc,
                )

    @staticmethod
    def _edge_identity(edge: SubstrateEdgeV1) -> str:
        return f"{edge.source.node_id}|{edge.predicate}|{edge.target.node_id}"

    def get_node_by_id(self, node_id: str) -> BaseSubstrateNodeV1 | None:
        return self._cache.get_node_by_id(node_id)

    def get_edge_by_id(self, edge_id: str) -> SubstrateEdgeV1 | None:
        return self._cache.get_edge_by_id(edge_id)

    def get_node_id_by_identity(self, identity_key: str) -> str | None:
        return self._cache.get_node_id_by_identity(identity_key)

    def get_identity_key_by_node_id(self, node_id: str) -> str | None:
        return self._cache.get_identity_key_by_node_id(node_id)

    def get_edge_id_by_identity(self, identity_key: str) -> str | None:
        return self._cache.get_edge_id_by_identity(identity_key)

    def upsert_node(self, *, identity_key: str | None, node: BaseSubstrateNodeV1) -> None:
        if getattr(node, "node_kind", None) not in ("concept", "evidence"):
            raise ValueError(
                "FalkorSubstrateStore durable writes support concept and evidence nodes only; "
                f"got node_kind={getattr(node, 'node_kind', None)!r}"
            )
        node = _with_sanitized_metadata(node)
        params = encode_node_properties(node, identity_key)
        label = node_label_for_kind(str(node.node_kind))
        assignments = _set_assignments("n", params, skip={"node_id"})
        cypher = (
            f"MERGE (n:SubstrateNode:{label} {{node_id: $node_id}}) "
            f"SET {assignments} "
            "REMOVE n.payload_json"
        )
        try:
            self._client.graph_query(cypher, params)
        except Exception as exc:
            logger.error("falkor_substrate_upsert_node_failed node_id=%s error=%s", node.node_id, exc)
            raise
        self._cache.upsert_node(identity_key=identity_key, node=node)
        self._write_generation += 1

    def upsert_edge(self, *, identity_key: str, edge: SubstrateEdgeV1) -> None:
        edge = _with_sanitized_metadata(edge)
        params = encode_edge_properties(edge, identity_key)
        relationship_type = edge.predicate
        assignments = _set_assignments("e", params, skip={"edge_id", "source_id", "target_id"})
        cypher = (
            "MERGE (source:SubstrateNode {node_id: $source_id}) "
            "MERGE (target:SubstrateNode {node_id: $target_id}) "
            f"MERGE (source)-[e:`{relationship_type}` {{edge_id: $edge_id}}]->(target) "
            f"SET {assignments} "
            "REMOVE e.payload_json"
        )
        try:
            self._client.graph_query(cypher, params)
        except Exception as exc:
            logger.error("falkor_substrate_upsert_edge_failed edge_id=%s error=%s", edge.edge_id, exc)
            raise
        self._cache.upsert_edge(identity_key=identity_key, edge=edge)
        self._write_generation += 1

    def snapshot(self) -> MaterializedSubstrateGraphState:
        with self._snapshot_lock:
            now_mono = time.monotonic()
            elapsed = (now_mono - self._last_snapshot_at) if self._last_snapshot_at is not None else None
            ceiling = float(self._cfg.snapshot_force_refresh_ceiling_sec)

            # A write since our last refresh is a KNOWN, certain change -- the
            # cache is never reused in that case, regardless of ceiling.
            same_generation = self._last_snapshot_generation == self._write_generation
            # ceiling <= 0 means "trust same_generation forever" (no periodic
            # forced refresh); otherwise the ceiling forces a real refresh once
            # elapsed even though same_generation is still true -- the safety
            # net that bounds staleness from writes THIS process can't see:
            # another process's write, or a direct external mutation (e.g. an
            # operator running Cypher DELETE by hand against Falkor directly).
            within_ceiling = ceiling <= 0.0 or elapsed is None or elapsed < ceiling

            if same_generation and within_ceiling:
                return self._cache.snapshot()

            # Capture the generation BEFORE refreshing (which does network
            # I/O), not after -- if a write races in while the refresh is in
            # flight, capturing after would credit that write to data fetched
            # before it happened, silently masking it until the next
            # unrelated write or the ceiling fires again. Capturing before
            # means such a race just costs one possibly-redundant extra
            # refresh on the next call, never a falsely-trusted stale cache.
            # Same reasoning as GraphDBSubstrateStore.snapshot() (already
            # proven in production for the SPARQL backend); this mirrors it
            # for Falkor to fix the equivalent gap here.
            generation_at_fetch_start = self._write_generation
            self._hydrate_from_durable()
            self._last_snapshot_at = now_mono
            self._last_snapshot_generation = generation_at_fetch_start
            return self._cache.snapshot()

    def query_focal_slice(self, *, node_ids: list[str], max_edges: int = 64) -> SubstrateQueryResultV1:
        return _retag_source(
            self._cache.query_focal_slice(node_ids=node_ids, max_edges=max_edges),
            self._result_source_kind,
        )

    def query_hotspot_region(
        self, *, min_salience: float = 0.6, limit_nodes: int = 32, limit_edges: int = 64
    ) -> SubstrateQueryResultV1:
        result = self._cache.query_hotspot_region(
            min_salience=min_salience, limit_nodes=limit_nodes, limit_edges=limit_edges
        )
        return _retag_source(result, self._result_source_kind)

    def query_contradiction_region(
        self, *, limit_nodes: int = 32, limit_edges: int = 64
    ) -> SubstrateQueryResultV1:
        return _retag_source(
            self._cache.query_contradiction_region(limit_nodes=limit_nodes, limit_edges=limit_edges),
            self._result_source_kind,
        )

    def query_concept_region(
        self, *, limit_nodes: int = 32, limit_edges: int = 64
    ) -> SubstrateQueryResultV1:
        return _retag_source(
            self._cache.query_concept_region(limit_nodes=limit_nodes, limit_edges=limit_edges),
            self._result_source_kind,
        )

    def query_provenance_neighborhood(
        self, *, evidence_ref: str, limit_nodes: int = 32, limit_edges: int = 64
    ) -> SubstrateQueryResultV1:
        return _retag_source(
            self._cache.query_provenance_neighborhood(
                evidence_ref=evidence_ref, limit_nodes=limit_nodes, limit_edges=limit_edges
            ),
            self._result_source_kind,
        )

    def read_focal_slice(self, *, node_ids: list[str], max_edges: int = 64) -> SubstrateNeighborhoodSliceV1:
        return self._cache.read_focal_slice(node_ids=node_ids, max_edges=max_edges)

    def read_hotspot_region(
        self, *, min_salience: float = 0.6, limit_nodes: int = 32, limit_edges: int = 64
    ) -> SubstrateNeighborhoodSliceV1:
        return self._cache.read_hotspot_region(
            min_salience=min_salience, limit_nodes=limit_nodes, limit_edges=limit_edges
        )

    def read_contradiction_region(
        self, *, limit_nodes: int = 32, limit_edges: int = 64
    ) -> SubstrateNeighborhoodSliceV1:
        return self._cache.read_contradiction_region(limit_nodes=limit_nodes, limit_edges=limit_edges)

    def read_concept_region(
        self, *, limit_nodes: int = 32, limit_edges: int = 64
    ) -> SubstrateNeighborhoodSliceV1:
        return self._cache.read_concept_region(limit_nodes=limit_nodes, limit_edges=limit_edges)

    def read_provenance_neighborhood(
        self, *, evidence_ref: str, limit_nodes: int = 32, limit_edges: int = 64
    ) -> SubstrateNeighborhoodSliceV1:
        return self._cache.read_provenance_neighborhood(
            evidence_ref=evidence_ref, limit_nodes=limit_nodes, limit_edges=limit_edges
        )


def _retag_source(result: SubstrateQueryResultV1, source_kind: str) -> SubstrateQueryResultV1:
    return replace(result, source_kind=source_kind)


def _normalize_rows(
    raw: Any, *, fields: tuple[str, ...] | list[str] | None = None
) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, list):
        # Raw GRAPH.QUERY response: [header, records, statistics]. redis-py's
        # Graph.query normally strips this shape, but retain compatibility
        # with injected clients that return the wire response.
        if (
            len(raw) == 3
            and isinstance(raw[0], list)
            and isinstance(raw[1], list)
            and raw[0]
            and all(isinstance(column, (list, tuple)) for column in raw[0])
        ):
            names = _header_field_names(raw[0])
            return [
                dict(zip(names, record))
                for record in raw[1]
                if isinstance(record, (list, tuple)) and len(record) == len(names)
            ]
        field_names = list(fields) if fields else []
        out: list[dict[str, Any]] = []
        for item in raw:
            if isinstance(item, dict):
                if "_positional" in item and field_names:
                    values = item.get("_positional")
                    if isinstance(values, list) and len(values) == len(field_names):
                        out.append(dict(zip(field_names, values)))
                        continue
                out.append(item)
            elif isinstance(item, (list, tuple)):
                if field_names and len(item) == len(field_names):
                    out.append(dict(zip(field_names, item)))
                elif field_names:
                    logger.warning(
                        "falkor_substrate_normalize_row_width_mismatch expected=%s got=%s",
                        len(field_names),
                        len(item),
                    )
                elif len(item) >= 2:
                    # Legacy two-column fallback only when caller omitted fields.
                    out.append({"node_id": item[0], "identity_key": item[1]})
                elif item:
                    out.append({"node_id": item[0], "identity_key": ""})
        return out
    return []


def _resolve_falkor_snapshot_force_refresh_ceiling_sec() -> float:
    """FALKOR_SNAPSHOT_FORCE_REFRESH_CEILING_SEC, falling back to the shared
    SUBSTRATE_SNAPSHOT_FORCE_REFRESH_CEILING_SEC (same knob GraphDB reads)
    when unset. A Falkor-specific override exists because
    RoutedSubstrateGraphStore can run a GraphDB-backed store and a
    Falkor-backed store concurrently in the same process (primary + shadow),
    and the two backends' refresh costs are not symmetric -- Falkor's own
    hydrate queries are currently unbounded (no LIMIT, unlike GraphDB's
    capped _query_nodes/_query_edges_for_node_ids), so an operator running
    both may want a longer Falkor ceiling without also having to change
    GraphDB's. Without this override, both backends would be forced to share
    one ceiling value with no way to tune them independently.
    """
    raw = str(os.getenv("FALKOR_SNAPSHOT_FORCE_REFRESH_CEILING_SEC", "")).strip()
    if not raw:
        return _resolve_snapshot_force_refresh_ceiling_sec()
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "falkor_snapshot_force_refresh_ceiling_invalid value=%r; falling back to shared setting", raw
        )
        return _resolve_snapshot_force_refresh_ceiling_sec()
    if not math.isfinite(value):
        logger.warning(
            "falkor_snapshot_force_refresh_ceiling_invalid value=%r (non-finite); falling back to shared setting",
            raw,
        )
        return _resolve_snapshot_force_refresh_ceiling_sec()
    return value


def build_falkor_substrate_store_from_env() -> FalkorSubstrateStore | InMemorySubstrateGraphStore:
    uri = str(os.getenv("FALKORDB_URI", "")).strip()
    if not uri:
        logger.warning("SUBSTRATE_STORE_BACKEND=falkor but FALKORDB_URI missing; falling back to in-memory")
        return InMemorySubstrateGraphStore()
    graph_name = str(os.getenv("FALKORDB_SUBSTRATE_GRAPH", "orion_substrate")).strip() or "orion_substrate"
    logger.info(
        "substrate_store_backend_selected backend=falkor uri_host=%s graph=%s",
        urlparse(uri).hostname or "",
        graph_name,
    )
    return FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(
            uri=uri,
            graph_name=graph_name,
            snapshot_force_refresh_ceiling_sec=_resolve_falkor_snapshot_force_refresh_ceiling_sec(),
        )
    )
