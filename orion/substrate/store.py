from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

from orion.core.schemas.cognitive_substrate import BaseSubstrateNodeV1, SubstrateEdgeV1
from .neighborhood import NeighborhoodRequestV1, NeighborhoodResultV1, read_memory_neighborhood


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class CompleteScanReceipt:
    """Operational coverage receipt; a paged read is never snapshot isolation."""

    started_at: str
    finished_at: str
    complete: bool
    stale: bool
    node_count: int
    edge_count: int
    pages_read: int
    last_successful_refresh_at: str | None
    reason: str | None = None
    consistency: str = "non_atomic_keyset"
    edge_identity_aliases: int = 0
    # Identities held by more than one durable node (hydrated, not fatal --
    # see FalkorSubstrateStore._hydrate_from_durable). Nonzero = repair due.
    duplicate_node_identities: int = 0


@dataclass(frozen=True)
class MaterializedSubstrateGraphState:
    nodes: dict[str, BaseSubstrateNodeV1]
    edges: dict[str, SubstrateEdgeV1]
    node_identity_index: dict[str, str]
    edge_identity_index: dict[str, str]
    scan_receipt: CompleteScanReceipt | None = None


@dataclass(frozen=True)
class SubstrateNeighborhoodSliceV1:
    nodes: list[BaseSubstrateNodeV1]
    edges: list[SubstrateEdgeV1]


@dataclass(frozen=True)
class SubstrateQueryResultV1:
    query_kind: str
    slice: SubstrateNeighborhoodSliceV1
    source_kind: str
    degraded: bool = False
    error: str | None = None
    truncated: bool = False
    limits: dict[str, int] = field(default_factory=dict)
    generated_at: str = field(default_factory=_utc_now_iso)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NodeDynamicsWriteV1:
    """A targeted write of ONLY the fields SubstrateDynamicsEngine.tick() owns.

    Why this exists (confirmed live 2026-10-11, wp-read timestamp repair
    PR #2608): the tick used to re-upsert the WHOLE node from its start-of-tick
    snapshot, so any write to any other property (observed_at, label,
    provenance, ...) landing between the tick's read and its write was
    reverted -- 238 corrected observed_at values within two minutes.

    Two groups, with different guards:

    - Pressure (``dynamic_pressure``/``dynamic_pressure_reason``): computed
      from the graph's prediction-error/contradiction state, owned only by the
      tick, written unconditionally when set (None = leave untouched).
    - Activation (``activation``, ``recency_score``, ``activation_decayed_at``,
      and dormancy, which is derived from activation): computed by decaying
      the value the tick READ, so it is written compare-and-set -- only if the
      node still holds ``expected_activation`` and ``expected_decay_stamp``.
      A fresh activation written by anyone else in between wins; the tick
      re-reads and decays it next tick.

    ``expected_decay_stamp`` is the stamp in its stored (encoded string) form,
    or None when the node carried none.
    """

    node_id: str
    node_kind: str
    expected_activation: float
    expected_decay_stamp: str | None
    activation: float
    recency_score: float
    activation_decayed_at: str
    dynamic_pressure: float | None = None
    dynamic_pressure_reason: str | None = None
    dormant: bool | None = None
    dormancy_updated_at: str | None = None


# Activation compare-and-set tolerance (absolute; activation is in [0, 1]).
# FalkorDB returns doubles rounded to ~15 significant digits, so the value read
# back never equals the stored one exactly; see FalkorSubstrateStore.
# update_node_dynamics. Used by every store so they agree.
ACTIVATION_CAS_TOLERANCE = 1e-12


def encode_decay_stamp(raw: object) -> str | None:
    """The stored string form of an ``activation_decayed_at`` value (mirrors
    falkor_codec's encoding, kept here so the in-memory store compares the
    same representation the durable store does)."""
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw.isoformat()
    return str(raw)


def apply_node_dynamics_write(
    node: BaseSubstrateNodeV1,
    write: NodeDynamicsWriteV1,
    *,
    include_activation: bool,
    include_concept_only: bool = True,
) -> BaseSubstrateNodeV1:
    """Return ``node`` with only the write's owned fields replaced. Every
    other field -- observed_at, label, provenance, foreign metadata -- is the
    node's own, untouched."""
    from .activation import ACTIVATION_DECAYED_AT_KEY

    metadata = dict(node.metadata or {})
    if include_concept_only and write.dynamic_pressure is not None:
        metadata["dynamic_pressure"] = write.dynamic_pressure
        metadata["dynamic_pressure_reason"] = write.dynamic_pressure_reason
    update: dict[str, Any] = {}
    if include_activation:
        metadata[ACTIVATION_DECAYED_AT_KEY] = write.activation_decayed_at
        if include_concept_only and write.dormant is not None:
            metadata["dormant"] = write.dormant
            metadata["dormancy_updated_at"] = write.dormancy_updated_at
        bundle = node.signals.activation.model_copy(
            update={"activation": write.activation, "recency_score": write.recency_score}
        )
        update["signals"] = node.signals.model_copy(update={"activation": bundle})
    update["metadata"] = metadata
    return node.model_copy(update=update)


_SIGNAL_ACTIVATION_KEYS = ("activation", "recency_score")


def preserve_skipped_keys(
    node: BaseSubstrateNodeV1, existing: BaseSubstrateNodeV1, skip_metadata_keys: frozenset[str]
) -> BaseSubstrateNodeV1:
    """``node`` with ``existing``'s values for every skipped key: metadata keys
    are copied (or dropped when ``existing`` lacks them), and the signal-level
    ``activation``/``recency_score`` (falkor_codec.ACTIVATION_STATE_KEYS) are
    copied from ``existing.signals.activation``. The full-replace stores' half
    of upsert_node()'s ``skip_metadata_keys`` contract."""
    merged_metadata = dict(node.metadata or {})
    existing_metadata = existing.metadata or {}
    for key in skip_metadata_keys:
        if key in _SIGNAL_ACTIVATION_KEYS:
            continue
        if key in existing_metadata:
            merged_metadata[key] = existing_metadata[key]
        else:
            merged_metadata.pop(key, None)
    update: dict[str, Any] = {"metadata": merged_metadata}
    skipped_signals = [key for key in _SIGNAL_ACTIVATION_KEYS if key in skip_metadata_keys]
    if skipped_signals:
        bundle = node.signals.activation.model_copy(
            update={key: getattr(existing.signals.activation, key) for key in skipped_signals}
        )
        update["signals"] = node.signals.model_copy(update={"activation": bundle})
    return node.model_copy(update=update)


def node_dynamics_cas_matches(node: BaseSubstrateNodeV1, write: NodeDynamicsWriteV1) -> bool:
    from .activation import ACTIVATION_DECAYED_AT_KEY

    return (
        abs(node.signals.activation.activation - write.expected_activation) <= ACTIVATION_CAS_TOLERANCE
        and encode_decay_stamp((node.metadata or {}).get(ACTIVATION_DECAYED_AT_KEY)) == write.expected_decay_stamp
    )


def node_dynamics_writer(store: object):
    """The store's targeted ``update_node_dynamics`` if it has one, else None.

    Looked up on the class, not via the Protocol: SubstrateGraphStore is a
    Protocol some stores subclass explicitly (GraphDBSubstrateStore), and a
    Protocol stub would be inherited as a silent no-op returning None. A
    wrapper (RoutedSubstrateGraphStore) reports its primary's capability via
    ``supports_node_dynamics_write``.
    """
    if not callable(getattr(type(store), "update_node_dynamics", None)):
        return None
    if not getattr(store, "supports_node_dynamics_write", True):
        return None
    return store.update_node_dynamics  # type: ignore[attr-defined]


class SubstrateGraphStore(Protocol):
    def get_node_by_id(self, node_id: str) -> BaseSubstrateNodeV1 | None: ...
    def get_edge_by_id(self, edge_id: str) -> SubstrateEdgeV1 | None: ...
    def get_node_id_by_identity(self, identity_key: str) -> str | None: ...
    def get_edge_id_by_identity(self, identity_key: str) -> str | None: ...
    def get_identity_key_by_node_id(self, node_id: str) -> str | None: ...
    def upsert_node(
        self,
        *,
        identity_key: str | None,
        node: BaseSubstrateNodeV1,
        skip_metadata_keys: frozenset[str] | None = None,
    ) -> None: ...
    def upsert_edge(self, *, identity_key: str, edge: SubstrateEdgeV1) -> None: ...
    def snapshot(self) -> MaterializedSubstrateGraphState: ...

    def read_neighborhood(self, request: NeighborhoodRequestV1) -> NeighborhoodResultV1: ...

    def query_focal_slice(self, *, node_ids: list[str], max_edges: int = 64) -> SubstrateQueryResultV1: ...
    def query_hotspot_region(self, *, min_salience: float = 0.6, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateQueryResultV1: ...
    def query_contradiction_region(self, *, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateQueryResultV1: ...
    def query_concept_region(self, *, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateQueryResultV1: ...
    def query_provenance_neighborhood(self, *, evidence_ref: str, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateQueryResultV1: ...

    def read_focal_slice(self, *, node_ids: list[str], max_edges: int = 64) -> SubstrateNeighborhoodSliceV1: ...
    def read_hotspot_region(self, *, min_salience: float = 0.6, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateNeighborhoodSliceV1: ...
    def read_contradiction_region(self, *, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateNeighborhoodSliceV1: ...
    def read_concept_region(self, *, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateNeighborhoodSliceV1: ...
    def read_provenance_neighborhood(self, *, evidence_ref: str, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateNeighborhoodSliceV1: ...


class InMemorySubstrateGraphStore:
    """Bounded persistent in-memory materialized substrate graph state."""

    def __init__(self) -> None:
        self._nodes: dict[str, BaseSubstrateNodeV1] = {}
        self._edges: dict[str, SubstrateEdgeV1] = {}
        self._node_identity_index: dict[str, str] = {}
        self._edge_identity_index: dict[str, str] = {}
        # Reverse of _node_identity_index, maintained alongside it. Exists so
        # a caller that already has a node_id (e.g. from a prior read) can
        # recover its identity_key for a subsequent upsert_node() call
        # without a full snapshot() -- FalkorSubstrateStore's codec writes
        # `identity_key or ""` unconditionally on every upsert, so passing a
        # missing/None identity_key durably clobbers the node's real one.
        self._identity_key_by_node_id: dict[str, str] = {}

    def get_node_by_id(self, node_id: str) -> BaseSubstrateNodeV1 | None:
        return self._nodes.get(node_id)

    def get_edge_by_id(self, edge_id: str) -> SubstrateEdgeV1 | None:
        return self._edges.get(edge_id)

    def get_node_id_by_identity(self, identity_key: str) -> str | None:
        return self._node_identity_index.get(identity_key)

    def get_edge_id_by_identity(self, identity_key: str) -> str | None:
        return self._edge_identity_index.get(identity_key)

    def get_identity_key_by_node_id(self, node_id: str) -> str | None:
        return self._identity_key_by_node_id.get(node_id)

    def upsert_node(
        self,
        *,
        identity_key: str | None,
        node: BaseSubstrateNodeV1,
        skip_metadata_keys: frozenset[str] | None = None,
    ) -> None:
        """``skip_metadata_keys``: same contract as FalkorSubstrateStore's
        (see that implementation's docstring) -- preserve whatever's already
        stored for these metadata keys instead of overwriting with this
        call's copy, so a test double backed by this in-memory store
        exercises the same "don't clobber a field you don't own" behavior."""
        if skip_metadata_keys:
            existing = self._nodes.get(node.node_id)
            if existing is not None:
                node = preserve_skipped_keys(node, existing, skip_metadata_keys)
        self._nodes[node.node_id] = node
        if identity_key:
            self._node_identity_index[identity_key] = node.node_id
            self._identity_key_by_node_id[node.node_id] = identity_key

    def upsert_edge(self, *, identity_key: str, edge: SubstrateEdgeV1) -> None:
        self._edges[edge.edge_id] = edge
        self._edge_identity_index[identity_key] = edge.edge_id

    def set_identity_canonical(self, identity_key: str, node_id: str) -> None:
        """Point ``identity_key`` at ``node_id`` without touching any node.

        For a duplicated identity (several stored nodes, one key): every
        duplicate keeps its own reverse mapping, so a later write to it still
        carries its real identity_key, while lookups resolve to one canonical
        node. Only an already-cached node can become canonical."""
        if node_id in self._nodes:
            self._node_identity_index[identity_key] = node_id
            self._identity_key_by_node_id[node_id] = identity_key

    def update_node_dynamics(self, write: NodeDynamicsWriteV1) -> bool:
        """Targeted write of the dynamics tick's owned fields (see
        NodeDynamicsWriteV1). Returns whether the activation group applied
        (False = the node is gone, or its activation/stamp moved since the
        tick read it). Never creates a node."""
        existing = self._nodes.get(write.node_id)
        if existing is None:
            return False
        applied = node_dynamics_cas_matches(existing, write)
        self._nodes[write.node_id] = apply_node_dynamics_write(existing, write, include_activation=applied)
        return applied

    def snapshot(self) -> MaterializedSubstrateGraphState:
        return MaterializedSubstrateGraphState(
            nodes=dict(self._nodes),
            edges=dict(self._edges),
            node_identity_index=dict(self._node_identity_index),
            edge_identity_index=dict(self._edge_identity_index),
        )

    def read_neighborhood(self, request: NeighborhoodRequestV1) -> NeighborhoodResultV1:
        return read_memory_neighborhood(self, request)

    def query_focal_slice(self, *, node_ids: list[str], max_edges: int = 64) -> SubstrateQueryResultV1:
        edges_limit = max(1, int(max_edges))
        slice_value = self.read_focal_slice(node_ids=node_ids, max_edges=edges_limit)
        return SubstrateQueryResultV1(
            query_kind="focal_slice",
            slice=slice_value,
            source_kind="cache",
            limits={"max_edges": edges_limit, "node_ids": len(node_ids)},
        )

    def query_hotspot_region(self, *, min_salience: float = 0.6, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateQueryResultV1:
        nodes_limit = max(1, int(limit_nodes))
        edges_limit = max(1, int(limit_edges))
        slice_value = self.read_hotspot_region(min_salience=min_salience, limit_nodes=nodes_limit, limit_edges=edges_limit)
        return SubstrateQueryResultV1(
            query_kind="hotspot_region",
            slice=slice_value,
            source_kind="cache",
            limits={"limit_nodes": nodes_limit, "limit_edges": edges_limit},
            details={"min_salience": float(min_salience)},
            truncated=len(slice_value.nodes) >= nodes_limit or len(slice_value.edges) >= edges_limit,
        )

    def query_contradiction_region(self, *, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateQueryResultV1:
        nodes_limit = max(1, int(limit_nodes))
        edges_limit = max(1, int(limit_edges))
        slice_value = self.read_contradiction_region(limit_nodes=nodes_limit, limit_edges=edges_limit)
        return SubstrateQueryResultV1(
            query_kind="contradiction_region",
            slice=slice_value,
            source_kind="cache",
            limits={"limit_nodes": nodes_limit, "limit_edges": edges_limit},
            truncated=len(slice_value.nodes) >= nodes_limit or len(slice_value.edges) >= edges_limit,
        )

    def query_concept_region(self, *, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateQueryResultV1:
        nodes_limit = max(1, int(limit_nodes))
        edges_limit = max(1, int(limit_edges))
        slice_value = self.read_concept_region(limit_nodes=nodes_limit, limit_edges=edges_limit)
        return SubstrateQueryResultV1(
            query_kind="concept_region",
            slice=slice_value,
            source_kind="cache",
            limits={"limit_nodes": nodes_limit, "limit_edges": edges_limit},
            truncated=len(slice_value.nodes) >= nodes_limit or len(slice_value.edges) >= edges_limit,
        )

    def query_provenance_neighborhood(self, *, evidence_ref: str, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateQueryResultV1:
        nodes_limit = max(1, int(limit_nodes))
        edges_limit = max(1, int(limit_edges))
        slice_value = self.read_provenance_neighborhood(evidence_ref=evidence_ref, limit_nodes=nodes_limit, limit_edges=edges_limit)
        return SubstrateQueryResultV1(
            query_kind="provenance_neighborhood",
            slice=slice_value,
            source_kind="cache",
            limits={"limit_nodes": nodes_limit, "limit_edges": edges_limit},
            details={"evidence_ref": str(evidence_ref or "")},
            truncated=len(slice_value.nodes) >= nodes_limit or len(slice_value.edges) >= edges_limit,
        )

    def read_focal_slice(self, *, node_ids: list[str], max_edges: int = 64) -> SubstrateNeighborhoodSliceV1:
        node_set = {str(node_id) for node_id in node_ids if str(node_id)}
        nodes = [self._nodes[node_id] for node_id in node_set if node_id in self._nodes]
        edge_candidates = [
            edge
            for edge in self._edges.values()
            if (edge.source.node_id in node_set or edge.target.node_id in node_set) and self._walkable(edge)
        ]
        edge_candidates.sort(key=lambda edge: (edge.salience, edge.confidence), reverse=True)
        return SubstrateNeighborhoodSliceV1(nodes=nodes, edges=edge_candidates[: max(1, int(max_edges))])

    def _walkable(self, edge: SubstrateEdgeV1) -> bool:
        """The neighborhood's edge-role gate, for every region read (recall's concept region
        included): no structure/provenance edges, no unaccepted projections."""
        from .neighborhood import walkable_edge

        return walkable_edge(edge, self._nodes.get(edge.assertion_id) if edge.assertion_id else None)

    def _read_by_node_predicate(self, *, node_predicate, limit_nodes: int, limit_edges: int) -> SubstrateNeighborhoodSliceV1:
        bounded_nodes = max(1, int(limit_nodes))
        bounded_edges = max(1, int(limit_edges))
        nodes = [node for node in self._nodes.values() if node_predicate(node)]
        nodes.sort(key=lambda node: (node.signals.salience, node.signals.confidence), reverse=True)
        selected_nodes = nodes[:bounded_nodes]
        node_ids = {node.node_id for node in selected_nodes}
        edges = [
            edge
            for edge in self._edges.values()
            if (edge.source.node_id in node_ids or edge.target.node_id in node_ids)
            and edge.edge_role in ("legacy_unreviewed", "semantic_projection")
        ]
        edges.sort(key=lambda edge: (edge.salience, edge.confidence), reverse=True)
        # Same order as falkor_direct's concept region: role filter before the cut, the
        # assertion check on what survives it (the two must stay equivalent).
        return SubstrateNeighborhoodSliceV1(nodes=selected_nodes,
                                            edges=[e for e in edges[:bounded_edges] if self._walkable(e)])

    def read_hotspot_region(self, *, min_salience: float = 0.6, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateNeighborhoodSliceV1:
        threshold = max(0.0, min(1.0, float(min_salience)))
        return self._read_by_node_predicate(
            node_predicate=lambda node: float(node.signals.salience) >= threshold,
            limit_nodes=limit_nodes,
            limit_edges=limit_edges,
        )

    def read_contradiction_region(self, *, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateNeighborhoodSliceV1:
        return self._read_by_node_predicate(
            node_predicate=lambda node: node.node_kind == "contradiction",
            limit_nodes=limit_nodes,
            limit_edges=limit_edges,
        )

    def read_concept_region(self, *, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateNeighborhoodSliceV1:
        return self._read_by_node_predicate(
            node_predicate=lambda node: node.node_kind == "concept",
            limit_nodes=limit_nodes,
            limit_edges=limit_edges,
        )

    def read_provenance_neighborhood(self, *, evidence_ref: str, limit_nodes: int = 32, limit_edges: int = 64) -> SubstrateNeighborhoodSliceV1:
        needle = str(evidence_ref or "").strip()
        if not needle:
            return SubstrateNeighborhoodSliceV1(nodes=[], edges=[])
        return self._read_by_node_predicate(
            node_predicate=lambda node: needle in (node.provenance.evidence_refs or []),
            limit_nodes=limit_nodes,
            limit_edges=limit_edges,
        )


def select_concept_nodes_by_anchor_scope(
    nodes: "list[BaseSubstrateNodeV1]", subjects: "list[str]"
) -> dict[str, list]:
    """Group concept-region query results into `node_kind == "concept"` nodes
    bucketed by `anchor_scope`, restricted to `subjects`.

    Shared by the two live readers of `query_concept_region()`'s concept
    nodes: `orion/substrate/relational/adapters/concept_induction_ctx.py`
    (chat_stance's `concept_induced` tier) and
    `orion/spark/concept_induction/substrate_repository.py`
    (`concept_induction_pass`'s `substrate` backend). Both need the exact
    same node_kind/anchor_scope filter; pulling it out here means a future
    change to that filter (a new node_kind, a new anchor scope) only has to
    be made once.
    """
    wanted = set(subjects)
    by_subject: dict[str, list] = {subject: [] for subject in subjects}
    for node in nodes:
        if getattr(node, "node_kind", None) != "concept":
            continue
        anchor = getattr(node, "anchor_scope", None)
        if anchor not in wanted:
            continue
        by_subject[anchor].append(node)
    return by_subject
