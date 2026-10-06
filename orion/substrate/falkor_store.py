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

from .neighborhood import NeighborhoodRequestV1, NeighborhoodResultV1

import logging
import math
import os
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
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
    set_assignments as _set_assignments,
)
from orion.graph.property_guard import sanitize_metadata
from orion.substrate.falkor_codec import (
    DURABLE_NODE_KINDS,
    JSON_SUFFIXED_EXTERNALLY_OWNED_METADATA_KEYS,
    decode_edge,
    decode_node,
    encode_edge_properties,
    encode_node_properties,
    node_label_for_kind,
)
from orion.substrate.store import (
    CompleteScanReceipt,
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
    # (2) Falkor hydrates all supported durable nodes/edges with bounded
    # keyset pages. A write/read loop still requests a complete refresh on
    # every generation change; local exploration should use read_neighborhood.
    snapshot_force_refresh_ceiling_sec: float = 30.0
    # Optional redis socket timeouts (seconds) for the default
    # RedisGraphQueryClient. None = redis-py default (no timeout), which is
    # what every caller got before these existed. orion-recall sets them so a
    # hung FalkorDB cannot pin a request thread (or the boot warmup) forever.
    client_socket_timeout_s: float | None = None
    client_socket_connect_timeout_s: float | None = None
    hydration_page_size: int = 1000

    def __post_init__(self) -> None:
        if not 1 <= self.hydration_page_size <= 10000:
            raise ValueError("hydration_page_size must be between 1 and 10000")


NATIVE_NODE_RETURN_FIELDS: tuple[str, ...] = (
    "node_id",
    "node_kind",
    "identity_key",
    "label",
    "definition",
    "taxonomy_path_json",
    "evidence_type",
    "content_ref",
    # entity nodes; without these the generic MATCH (n:SubstrateNode)
    # hydration returns them with every entity column NULL and
    # decode_entity_node falls back to "unknown"/[] for real stored values.
    "entity_type",
    "aliases_json",
    # topic_id has had a complete encode/decode pair since the topic-foundry
    # work (_topic_foundry_properties_from_metadata /
    # _topic_foundry_metadata_from_row) and was simply never listed here, so
    # the decoder could only ever see NULL and drop it. Every hydrated node
    # therefore lost its cluster id -- and the atlas colours nodes by it.
    "topic_id",
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
    # Without these, falkor_codec's decode branches for them can never fire:
    # this tuple is the only row source for hydration, so after any rehydrate
    # get_node_by_id() reports both as absent even though the durable Cypher
    # properties hold real values -- including inside
    # _write_prediction_error_node()'s own carry-forward read.
    # NOTE: contributing_turn_ids_json and
    # prediction_error_evidence_event_ids_json have the same gap and are left
    # as-is here; that is pre-existing and out of scope for this patch.
    "perception_staleness",
    "perception_yield",
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


# _set_assignments moved to orion.graph.falkor_client.set_assignments
# (2026-07-18, zero substrate coupling), imported above under its old name.


def _edge_hydrate_return_clause(fields: tuple[str, ...]) -> str:
    """Like `_return_clause("e", fields)`, except `source_id`/`target_id`
    are derived from the matched `source`/`target` node variables
    (`source.node_id`, `target.node_id`) instead of read as properties on
    the edge itself.

    `upsert_edge()` deliberately never writes `source_id`/`target_id` onto
    the edge (`skip={"edge_id", "source_id", "target_id"}` in its
    `_set_assignments()` call) -- the real linkage lives in the graph
    topology the MATCH pattern itself encodes, not a redundant edge
    property. Reading `e.source_id`/`e.target_id` therefore always returned
    NULL, which `decode_edge()` then `str()`-coerced into the literal
    string `"None"` for every hydrated edge's source/target node_id --
    confirmed live (2026-07-18): every edge in the running graph lost its
    real source/target linkage on every hydrate, silently, since nothing
    downstream raised on a `"None"`-string node_id.

    Column order and field names must stay identical to `_return_clause`'s
    output for the same `fields` tuple -- `_normalize_rows()` zips
    positional (list/tuple) query results against that same tuple by
    index, and `decode_edge()` looks fields up by name from the result.

    Tied to this file's one edge-hydration call site specifically: `source`
    and `target` are hardcoded literal Cypher variable names matching that
    query's own `MATCH (source:SubstrateNode)-[e]->(target:SubstrateNode)`
    pattern, not a general parameter. Reusing this for a query that binds
    those variables under different names would produce Cypher referencing
    undefined identifiers -- a loud parse/execution error, not a silent
    wrong-value bug, but not a drop-in generic helper either.
    """
    parts = []
    for field in fields:
        if field == "source_id":
            parts.append("source.node_id AS source_id")
        elif field == "target_id":
            parts.append("target.node_id AS target_id")
        else:
            parts.append(f"e.{field} AS {field}")
    return ", ".join(parts)


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
        if client is None:
            client_kwargs: dict[str, float] = {}
            if cfg.client_socket_timeout_s is not None:
                client_kwargs["socket_timeout"] = cfg.client_socket_timeout_s
            if cfg.client_socket_connect_timeout_s is not None:
                client_kwargs["socket_connect_timeout"] = cfg.client_socket_connect_timeout_s
            client = RedisGraphQueryClient(uri=cfg.uri, graph_name=cfg.graph_name, **client_kwargs)
        self._client: FalkorGraphClient = client
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
        # Guards the in-process cache and ``_write_generation`` against
        # concurrent writers. ``_snapshot_lock`` only serializes snapshot()
        # callers; writes never took it. Before cortex-exec moved the stance
        # build off the event loop (2026-10-06 turn-latency L2) a write could not
        # overlap a snapshot because the build blocked the whole loop. Now it
        # can, and the unified layer's cold fan-out already writes from pool
        # threads. Held only for in-memory work (never across a Falkor round
        # trip), so a 4 s rehydrate never blocks a writer for 4 s.
        self._cache_lock = threading.RLock()
        self.last_scan_receipt: CompleteScanReceipt | None = None
        self._last_successful_refresh_at: str | None = None
        if hydrate:
            self._hydrate_from_durable()

    last_hydrate_ok: bool | None = None
    last_hydrate_node_count: int = 0

    def _scan_pages(self, *, match: str, where: str, alias: str,
                    returns: str, fields: tuple[str, ...], progress: dict[str, int]):
        """A short page may be the server cap. Only an empty page ends a scan."""
        cursor = -1
        while True:
            raw = self._client.graph_query(
                f"{match} WHERE {where} AND id({alias}) > $after_id "
                f"RETURN {returns}, id({alias}) AS object_id "
                "ORDER BY object_id LIMIT $page_size",
                {"after_id": cursor, "page_size": self._cfg.hydration_page_size},
            )
            rows = _normalize_rows(raw, fields=(*fields, "object_id"), strict=True)
            progress["pages"] += 1
            if not rows:
                return
            for row in rows:
                object_id = row.get("object_id")
                if type(object_id) is not int or object_id <= cursor:
                    raise ValueError("nonadvancing or invalid object cursor")
                cursor = object_id
                yield row

    def _hydrate_from_durable(self) -> None:
        """Stage and validate all pages before replacing the last good cache.

        Object IDs are scan cursors, never business identities. Concurrent
        external mutation can produce a mixed-time view; detected missing
        endpoints or local writes reject the scan. Legacy rewrite remains a
        best-effort compatibility step after validation, never in read-only mode.
        """
        started = datetime.now(timezone.utc).isoformat()
        generation = self._write_generation
        progress = {"pages": 0, "edge_identity_aliases": 0}
        fresh = InMemorySubstrateGraphStore()
        legacy_nodes, legacy_edges = [], []
        node_ids, edge_ids = set(), set()

        def add_node(row, node):
            if node is None or not node.node_id or node.node_id == "None":
                raise ValueError("invalid or unsupported durable node")
            if node.node_id in node_ids:
                raise ValueError(f"duplicate node_id: {node.node_id}")
            identity = row.get("identity_key") or None
            if identity and fresh.get_node_id_by_identity(str(identity)) is not None:
                raise ValueError(f"duplicate node identity: {identity}")
            node_ids.add(node.node_id)
            fresh.upsert_node(identity_key=str(identity) if identity else None, node=node)

        def add_edge(row, edge):
            if edge is None:
                raise ValueError("invalid durable edge")
            if edge.edge_id in edge_ids:
                raise ValueError(f"duplicate edge_id: {edge.edge_id}")
            for endpoint in (edge.source, edge.target):
                node = fresh.get_node_by_id(endpoint.node_id)
                if node is None or node.node_kind != endpoint.node_kind:
                    raise ValueError(f"missing or mismatched endpoint: {endpoint.node_id}")
            identity = str(row.get("identity_key") or self._edge_identity(edge))
            previous_id = fresh.get_edge_id_by_identity(identity)
            if previous_id is not None:
                previous = fresh.get_edge_by_id(previous_id)
                if (previous.source != edge.source or previous.target != edge.target
                        or previous.predicate != edge.predicate):
                    raise ValueError(f"incompatible edge identity: {identity}")
                progress["edge_identity_aliases"] += 1
            edge_ids.add(edge.edge_id)
            fresh.upsert_edge(identity_key=identity, edge=edge)
            if previous_id is not None and previous_id < edge.edge_id:
                # Preserve all parallel edges, with a stable representative for
                # the historical one-ID lookup API (not a uniqueness claim).
                fresh.upsert_edge(identity_key=identity, edge=previous)

        try:
            for row in self._scan_pages(
                match="MATCH (n:SubstrateNode)", where="n.payload_json IS NULL",
                alias="n", returns=_return_clause("n", NATIVE_NODE_RETURN_FIELDS),
                fields=NATIVE_NODE_RETURN_FIELDS, progress=progress,
            ):
                add_node(row, decode_node(row))
            for row in self._scan_pages(
                match="MATCH (n:SubstrateNode)", where="n.payload_json IS NOT NULL",
                alias="n", returns="n.payload_json AS payload_json, n.identity_key AS identity_key",
                fields=("payload_json", "identity_key"), progress=progress,
            ):
                node = NODE_ADAPTER.validate_json(row["payload_json"])
                if node.node_kind not in DURABLE_NODE_KINDS:
                    raise ValueError("unsupported legacy node kind")
                add_node(row, node)
                legacy_nodes.append(row)
            for row in self._scan_pages(
                match="MATCH (source:SubstrateNode)-[e]->(target:SubstrateNode)",
                where="e.substrate_edge = true AND e.payload_json IS NULL", alias="e",
                returns=_edge_hydrate_return_clause(NATIVE_EDGE_RETURN_FIELDS),
                fields=NATIVE_EDGE_RETURN_FIELDS, progress=progress,
            ):
                add_edge(row, decode_edge(row))
            for row in self._scan_pages(
                match="MATCH ()-[e]->()", where="e.payload_json IS NOT NULL", alias="e",
                returns="e.payload_json AS payload_json, e.identity_key AS identity_key",
                fields=("payload_json", "identity_key"), progress=progress,
            ):
                add_edge(row, SubstrateEdgeV1.model_validate_json(row["payload_json"]))
                legacy_edges.append(row)
            # Generation check and cache swap are one atomic step under
            # _cache_lock: a local write landing between a bare check and the
            # swap would go into the old cache and then be dropped by the swap,
            # handing the caller a snapshot missing that write.
            with self._cache_lock:
                if self._write_generation != generation:
                    raise ValueError("local mutation during scan; retry required")
                self._cache = fresh
                self._last_snapshot_generation = generation
        except Exception as exc:
            logger.warning("falkor_substrate_hydrate_failed error=%s", exc)
            self.last_hydrate_ok = False
            self.last_scan_receipt = CompleteScanReceipt(
                started_at=started, finished_at=datetime.now(timezone.utc).isoformat(),
                complete=False, stale=True, node_count=len(node_ids), edge_count=len(edge_ids),
                pages_read=progress["pages"],
                last_successful_refresh_at=self._last_successful_refresh_at,
                reason=str(exc), edge_identity_aliases=progress["edge_identity_aliases"],
            )
            return

        self.last_hydrate_ok = True
        self.last_hydrate_node_count = len(node_ids)
        finished = datetime.now(timezone.utc).isoformat()
        self._last_snapshot_at = time.monotonic()
        self._last_successful_refresh_at = finished
        self.last_scan_receipt = CompleteScanReceipt(
            started_at=started, finished_at=finished, complete=True, stale=False,
            node_count=len(node_ids), edge_count=len(edge_ids), pages_read=progress["pages"],
            last_successful_refresh_at=finished,
            edge_identity_aliases=progress["edge_identity_aliases"],
        )
        if not getattr(self._client, "read_only", False):
            self._migrate_legacy_payload_nodes(legacy_nodes)
            self._migrate_legacy_payload_edges(legacy_edges)

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
            if getattr(node, "node_kind", None) not in DURABLE_NODE_KINDS:
                logger.warning(
                    "falkor_substrate_legacy_node_skipped node_kind=%s node_id=%s",
                    getattr(node, "node_kind", None),
                    getattr(node, "node_id", None),
                )
                continue
            identity = row.get("identity_key")
            identity_key = str(identity) if identity else None
            # Seed cache first so a transient rewrite failure cannot empty Atlas.
            with self._cache_lock:
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
            with self._cache_lock:
                representative_id = self._cache.get_edge_id_by_identity(str(identity))
                representative = self._cache.get_edge_by_id(representative_id) if representative_id else None
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

            finally:
                # Native rewriting must not change the lookup selected by the
                # validated staging scan, including when rewriting fails.
                if representative is not None and representative.edge_id < edge.edge_id:
                    with self._cache_lock:
                        self._cache.upsert_edge(identity_key=str(identity), edge=representative)

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

    def upsert_node(
        self,
        *,
        identity_key: str | None,
        node: BaseSubstrateNodeV1,
        skip_metadata_keys: frozenset[str] | None = None,
    ) -> None:
        """``skip_metadata_keys`` (raw ``node.metadata`` key names, e.g.
        ``falkor_codec.EXTERNALLY_OWNED_METADATA_KEYS``) excludes those keys
        from BOTH the Cypher SET clause (so an existing value on the graph is
        left untouched, not overwritten with this call's copy) AND the local
        in-process cache update (so the cache doesn't diverge from the graph
        by caching this call's stale copy instead of the real value already
        there). Confirmed live 2026-07-29 as a real bug, not theoretical: a
        caller with no reason to know the current value of a field it
        doesn't own (SubstrateDynamicsEngine.tick() re-persisting
        `prediction_error` purely because activation decay triggered its
        write guard) durably clobbered a different writer's fresh values --
        `node:substrate.bus_synaptic` frozen at `prediction_error=1.0` for
        3+ hours despite the real writer succeeding every 30s the whole
        time, which independently caused false "Bus Anomaly Detected"
        alerts via orion-equilibrium-service's own poll of the same frozen
        value. See falkor_codec.EXTERNALLY_OWNED_METADATA_KEYS's docstring
        for the full trace.

        The skip-preserving merge read, the cache write and the
        ``_write_generation`` bump run together under ``_cache_lock`` after the
        durable write, so a concurrent writer or a snapshot cache swap cannot
        interleave between them (2026-10-06, turn-latency L2).
        """
        # Derived from the codec's DURABLE_NODE_KINDS rather than repeating the
        # tuple: these two guards were separate hardcoded copies of the same
        # list, so widening one without the other would swap a clear rejection
        # for a confusing encode error one frame deeper.
        if getattr(node, "node_kind", None) not in DURABLE_NODE_KINDS:
            raise ValueError(
                f"FalkorSubstrateStore durable writes support {', '.join(DURABLE_NODE_KINDS)} nodes only; "
                f"got node_kind={getattr(node, 'node_kind', None)!r}"
            )
        node = _with_sanitized_metadata(node)

        skip_encoded_keys: set[str] = set()
        if skip_metadata_keys:
            # Translate raw metadata keys into the encoded Cypher property
            # names actually present in the SET clause -- contributing_turn_ids
            # becomes contributing_turn_ids_json only once
            # encode_node_properties() runs (see _dynamics_properties_from_metadata,
            # and test_externally_owned_metadata_keys_translation_matches_real_encoding
            # for the drift guard on this translation). List-typed keys get a
            # _json suffix (JSON_SUFFIXED_EXTERNALLY_OWNED_METADATA_KEYS,
            # falkor_codec.py -- the single source of truth, not duplicated
            # here as a hardcoded ternary, after that exact drift already
            # happened once: 2026-08-11, adding prediction_error_evidence_
            # event_ids to EXTERNALLY_OWNED_METADATA_KEYS without updating a
            # then-hardcoded ternary here reproduced the stale-clobber bug
            # this whole mechanism exists to prevent, caught only by the
            # drift-guard test below, not by inspection). Scalar-typed keys
            # (e.g. prediction_error) keep their raw name.
            for key in skip_metadata_keys:
                skip_encoded_keys.add(
                    f"{key}_json"
                    if key in JSON_SUFFIXED_EXTERNALLY_OWNED_METADATA_KEYS
                    else key
                )

        params = encode_node_properties(node, identity_key)
        label = node_label_for_kind(str(node.node_kind))
        assignments = _set_assignments("n", params, skip={"node_id"} | skip_encoded_keys)
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
        with self._cache_lock:
            cache_node = node
            if skip_metadata_keys:
                existing_cached = self._cache.get_node_by_id(node.node_id)
                merged_metadata = dict(node.metadata or {})
                existing_metadata = (existing_cached.metadata or {}) if existing_cached is not None else {}
                for key in skip_metadata_keys:
                    if key in existing_metadata:
                        merged_metadata[key] = existing_metadata[key]
                    else:
                        # No cached copy to fall back on -- an honest "unknown"
                        # (key absent) beats caching this caller's own unverified
                        # guess for a field it doesn't own. Review finding
                        # 2026-07-29: without this, a cache miss (e.g. mid-
                        # rehydrate) would leave the LOCAL cache holding the
                        # caller's copy for a field the durable Cypher write
                        # deliberately skipped -- cache/durable divergence until
                        # the next generation-triggered rehydrate.
                        merged_metadata.pop(key, None)
                cache_node = node.model_copy(update={"metadata": merged_metadata})
            self._cache.upsert_node(identity_key=identity_key, node=cache_node)
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
        with self._cache_lock:
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

            if not (same_generation and within_ceiling and self.last_hydrate_ok):
                self._hydrate_from_durable()
            receipt = self.last_scan_receipt
            with self._cache_lock:
                if receipt and self._last_snapshot_generation != self._write_generation:
                    receipt = replace(receipt, stale=True)
                state = self._cache.snapshot()
            return replace(state, scan_receipt=receipt)

    def read_neighborhood(self, request: NeighborhoodRequestV1) -> NeighborhoodResultV1:
        from .neighborhood_backends import read_falkor_neighborhood
        return read_falkor_neighborhood(self, request)

    # Region reads iterate the in-memory cache's dicts; InMemorySubstrateGraphStore
    # has no lock of its own, so they share _cache_lock with writers.
    def query_focal_slice(self, *, node_ids: list[str], max_edges: int = 64) -> SubstrateQueryResultV1:
        with self._cache_lock:
            result = self._cache.query_focal_slice(node_ids=node_ids, max_edges=max_edges)
        return _retag_source(result, self._result_source_kind)

    def query_hotspot_region(
        self, *, min_salience: float = 0.6, limit_nodes: int = 32, limit_edges: int = 64
    ) -> SubstrateQueryResultV1:
        with self._cache_lock:
            result = self._cache.query_hotspot_region(
                min_salience=min_salience, limit_nodes=limit_nodes, limit_edges=limit_edges
            )
        return _retag_source(result, self._result_source_kind)

    def query_contradiction_region(
        self, *, limit_nodes: int = 32, limit_edges: int = 64
    ) -> SubstrateQueryResultV1:
        with self._cache_lock:
            result = self._cache.query_contradiction_region(limit_nodes=limit_nodes, limit_edges=limit_edges)
        return _retag_source(result, self._result_source_kind)

    def query_concept_region(
        self, *, limit_nodes: int = 32, limit_edges: int = 64
    ) -> SubstrateQueryResultV1:
        with self._cache_lock:
            result = self._cache.query_concept_region(limit_nodes=limit_nodes, limit_edges=limit_edges)
        return _retag_source(result, self._result_source_kind)

    def query_provenance_neighborhood(
        self, *, evidence_ref: str, limit_nodes: int = 32, limit_edges: int = 64
    ) -> SubstrateQueryResultV1:
        with self._cache_lock:
            result = self._cache.query_provenance_neighborhood(
                evidence_ref=evidence_ref, limit_nodes=limit_nodes, limit_edges=limit_edges
            )
        return _retag_source(result, self._result_source_kind)

    def read_focal_slice(self, *, node_ids: list[str], max_edges: int = 64) -> SubstrateNeighborhoodSliceV1:
        with self._cache_lock:
            return self._cache.read_focal_slice(node_ids=node_ids, max_edges=max_edges)

    def read_hotspot_region(
        self, *, min_salience: float = 0.6, limit_nodes: int = 32, limit_edges: int = 64
    ) -> SubstrateNeighborhoodSliceV1:
        with self._cache_lock:
            return self._cache.read_hotspot_region(
                min_salience=min_salience, limit_nodes=limit_nodes, limit_edges=limit_edges
            )

    def read_contradiction_region(
        self, *, limit_nodes: int = 32, limit_edges: int = 64
    ) -> SubstrateNeighborhoodSliceV1:
        with self._cache_lock:
            return self._cache.read_contradiction_region(limit_nodes=limit_nodes, limit_edges=limit_edges)

    def read_concept_region(
        self, *, limit_nodes: int = 32, limit_edges: int = 64
    ) -> SubstrateNeighborhoodSliceV1:
        with self._cache_lock:
            return self._cache.read_concept_region(limit_nodes=limit_nodes, limit_edges=limit_edges)

    def read_provenance_neighborhood(
        self, *, evidence_ref: str, limit_nodes: int = 32, limit_edges: int = 64
    ) -> SubstrateNeighborhoodSliceV1:
        with self._cache_lock:
            return self._cache.read_provenance_neighborhood(
                evidence_ref=evidence_ref, limit_nodes=limit_nodes, limit_edges=limit_edges
            )


def _retag_source(result: SubstrateQueryResultV1, source_kind: str) -> SubstrateQueryResultV1:
    return replace(result, source_kind=source_kind)


def _normalize_rows(
    raw: Any, *, fields: tuple[str, ...] | list[str] | None = None, strict: bool = False
) -> list[dict[str, Any]]:
    if raw is None:
        if strict:
            raise ValueError("missing query result")
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
            if strict and any(not isinstance(r, (list, tuple)) or len(r) != len(names) for r in raw[1]):
                raise ValueError("malformed wire query row")
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
                if strict and "_positional" in item:
                    raise ValueError("malformed positional query row")
                out.append(item)
            elif isinstance(item, (list, tuple)):
                if field_names and len(item) == len(field_names):
                    out.append(dict(zip(field_names, item)))
                elif field_names:
                    if strict:
                        raise ValueError("query row width mismatch")
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
            elif strict:
                raise ValueError("malformed query row")
        return out
    if strict:
        raise ValueError("malformed query result")
    return []


def _resolve_falkor_snapshot_force_refresh_ceiling_sec() -> float:
    """FALKOR_SNAPSHOT_FORCE_REFRESH_CEILING_SEC, falling back to the shared
    SUBSTRATE_SNAPSHOT_FORCE_REFRESH_CEILING_SEC (same knob GraphDB reads)
    when unset. A Falkor-specific override exists because
    RoutedSubstrateGraphStore can run a GraphDB-backed store and a
    Falkor-backed store concurrently in the same process (primary + shadow),
    and the two backends' refresh costs are not symmetric -- Falkor's own
    hydrate traverses all pages (unlike GraphDB's
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


def build_falkor_substrate_store_from_env(
    *,
    graph_name_env: str = "FALKORDB_SUBSTRATE_GRAPH",
    graph_name_default: str = "orion_substrate",
    client_socket_timeout_s: float | None = None,
    client_socket_connect_timeout_s: float | None = None,
) -> FalkorSubstrateStore | InMemorySubstrateGraphStore:
    """Build a FalkorSubstrateStore from env, targeting a single named graph.

    ``client_socket_timeout_s``/``client_socket_connect_timeout_s`` are
    optional redis socket timeouts for the store's client (None = no timeout,
    the prior behaviour).

    ``graph_name_env``/``graph_name_default`` let a second call site build a
    second, independently-named graph on the same FalkorDB instance (FalkorDB
    holds multiple graphs at no extra infra cost -- a graph name is just a
    string) without duplicating this function's URI-resolution/logging/
    fallback logic. Both existing zero-arg call sites (graphdb_store.py,
    routed_store.py) keep resolving ``FALKORDB_SUBSTRATE_GRAPH`` exactly as
    before -- these are keyword-only with defaults matching prior behavior,
    not a breaking change. See build_aitown_falkor_substrate_store_from_env()
    below for the second call site this was added for.
    """
    uri = str(os.getenv("FALKORDB_URI", "")).strip()
    if not uri:
        logger.warning("SUBSTRATE_STORE_BACKEND=falkor but FALKORDB_URI missing; falling back to in-memory")
        return InMemorySubstrateGraphStore()
    graph_name = str(os.getenv(graph_name_env, graph_name_default)).strip() or graph_name_default
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
            client_socket_timeout_s=client_socket_timeout_s,
            client_socket_connect_timeout_s=client_socket_connect_timeout_s,
        )
    )


def build_aitown_falkor_substrate_store_from_env() -> FalkorSubstrateStore | InMemorySubstrateGraphStore:
    """Second, independently-named FalkorDB graph for AI Town's own
    organically-clustered concept graph (design spec:
    docs/superpowers/specs/2026-08-18-aitown-concept-graph-split-and-atlas-
    readability-design.md, "AI Town's own concept graph"). Interpretability-
    only -- explicitly not fed into concept_induced/chat_stance or any other
    Orion cognition consumer (same spec's Non-goals).

    Deliberately only wired for the falkor backend (unlike the generic
    multi-backend build_substrate_store_from_env() dispatcher this doesn't
    go through) -- the live orion_substrate deployment is Falkor-backed
    today and there is no real second consumer yet motivating a full
    graph-name override threaded through every other backend
    (routed/sparql/graphdb) too. Falls back to in-memory + warning the same
    way the primary graph does when FALKORDB_URI is unset, same reasoning:
    never raise, degrade honestly (per CLAUDE.md's "no keyword cathedral"
    gate -- build only what has a real consumer today).
    """
    return build_falkor_substrate_store_from_env(
        graph_name_env="FALKORDB_AITOWN_SUBSTRATE_GRAPH",
        graph_name_default="orion_substrate_aitown",
    )


def build_self_falkor_substrate_store_from_env() -> FalkorSubstrateStore | InMemorySubstrateGraphStore:
    """Third, independently-named FalkorDB graph -- the Self Atlas
    (self-model rebuild arc, Patch 3, 2026-09-05). Same shape and same
    Falkor-only scoping rationale as
    build_aitown_falkor_substrate_store_from_env() above."""
    return build_falkor_substrate_store_from_env(
        graph_name_env="FALKORDB_SELF_SUBSTRATE_GRAPH",
        graph_name_default="orion_substrate_self",
    )
