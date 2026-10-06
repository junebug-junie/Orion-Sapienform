"""Bounded evidence-handle reads: which Evidence nodes back a Concept/Entity.

A handle is an id plus a ``content_ref`` (``<table>:<id>``). Text stays in
Postgres; nothing here hydrates it. One rule for every backend:

* an Evidence node is linked to node N by ``N -observed_in-> Evidence`` or
  ``Evidence -supports-> N`` (the two provenance shapes the codec persists);
* the edge must be valid at ``at``: its start (``valid_from``, or
  ``observed_at`` when unset, i.e. when Orion first saw it) <= at, and
  ``valid_to`` unset or > at;
* optional ``evidence_types`` filter on the Evidence node;
* newest first by ``valid_from`` (``observed_at`` when unset), ties by edge_id;
* at most ``per_node_limit`` handles per node; a node with more is listed in
  ``truncated_node_ids``.

Requested ids that are absent, or not a concept/entity, are reported in
``missing_node_ids`` and make the read ``degraded`` (not an empty success).
Any backend error fails the whole read closed, without cache fallback.
Times are compared as UTC; the codec stores ISO-8601 strings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from orion.core.schemas.cognitive_substrate import BaseSubstrateNodeV1, SubstrateEdgeV1

SEMANTIC_KINDS = ("concept", "entity")
# (predicate, which endpoint is the semantic node). Both already exist in
# SubstrateEdgePredicateV1 and in the live graph (`supports`: 3,310 edges).
PROVENANCE_SHAPES = (("observed_in", "source"), ("supports", "target"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EvidenceHandleRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    node_ids: tuple[str, ...] = Field(min_length=1, max_length=16)
    evidence_types: tuple[str, ...] | None = Field(default=None, min_length=1, max_length=16)
    per_node_limit: int = Field(default=6, ge=1, le=16)
    at: datetime | None = None

    @field_validator("at")
    @classmethod
    def _utc(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("at must be timezone-aware")
        return value.astimezone(timezone.utc)

    def at_or_now(self) -> datetime:
        return self.at or datetime.now(timezone.utc)


@dataclass(frozen=True)
class EvidenceHandleV1:
    node_id: str
    edge_id: str
    predicate: str
    evidence_node_id: str
    evidence_type: str
    content_ref: str
    observed_at: datetime
    valid_from: datetime | None = None
    valid_to: datetime | None = None


@dataclass(frozen=True)
class EvidenceHandleResultV1:
    handles: tuple[EvidenceHandleV1, ...] = ()
    source_kind: str = "cache"
    complete_for_request: bool = False
    truncated: bool = False
    degraded: bool = False
    reason: str | None = None
    missing_node_ids: tuple[str, ...] = ()
    truncated_node_ids: tuple[str, ...] = ()
    read_started_at: str = field(default_factory=_now)
    read_finished_at: str = field(default_factory=_now)


def _utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _sort_key(handle: EvidenceHandleV1):
    return (-(_utc(handle.valid_from or handle.observed_at)).timestamp(), handle.edge_id)


def _result(request, *, source_kind, started, found, per_node) -> EvidenceHandleResultV1:
    requested = sorted(set(request.node_ids))
    missing = tuple(node_id for node_id in requested if node_id not in found)
    handles: list[EvidenceHandleV1] = []
    truncated_ids: list[str] = []
    for node_id in requested:
        if node_id not in found:
            continue
        items = per_node.get(node_id, [])
        if len(items) > request.per_node_limit:
            truncated_ids.append(node_id)
        handles.extend(items[:request.per_node_limit])
    return EvidenceHandleResultV1(
        handles=tuple(handles), source_kind=source_kind, truncated=bool(truncated_ids),
        complete_for_request=not truncated_ids and not missing, degraded=bool(missing),
        reason="node_unavailable_or_not_semantic" if missing else "budget_exhausted" if truncated_ids else None,
        missing_node_ids=missing, truncated_node_ids=tuple(truncated_ids),
        read_started_at=started, read_finished_at=_now())


def _failed(source_kind: str, started: str, exc: Exception) -> EvidenceHandleResultV1:
    # Error text can carry backend credentials; expose only the type.
    return EvidenceHandleResultV1(source_kind=source_kind, read_started_at=started,
                                  read_finished_at=_now(), degraded=True,
                                  reason=f"unavailable:{type(exc).__name__}")


def select_handles(request: EvidenceHandleRequestV1, node_id: str,
                   pairs: Iterable[tuple[SubstrateEdgeV1, BaseSubstrateNodeV1]]) -> list[EvidenceHandleV1]:
    """Reference rule over (edge, evidence node) candidates of one node.

    Returns up to per_node_limit + 1 handles so the caller can see truncation.
    """
    at = request.at_or_now()
    types = set(request.evidence_types) if request.evidence_types is not None else None
    shapes = dict(PROVENANCE_SHAPES)
    out = []
    for edge, evidence in pairs:
        side = shapes.get(edge.predicate)
        if side is None or evidence.node_kind != "evidence":
            continue
        semantic, other = (edge.source, edge.target) if side == "source" else (edge.target, edge.source)
        if semantic.node_id != node_id or other.node_id != evidence.node_id:
            continue
        evidence_type = getattr(evidence, "evidence_type", None)
        if types is not None and evidence_type not in types:
            continue
        valid_from, valid_to = _utc(edge.temporal.valid_from), _utc(edge.temporal.valid_to)
        start = valid_from or _utc(edge.temporal.observed_at)
        if start > at or (valid_to is not None and valid_to <= at):
            continue
        out.append(EvidenceHandleV1(
            node_id=node_id, edge_id=edge.edge_id, predicate=edge.predicate,
            evidence_node_id=evidence.node_id, evidence_type=str(evidence_type),
            content_ref=str(getattr(evidence, "content_ref", "")),
            observed_at=_utc(edge.temporal.observed_at), valid_from=valid_from, valid_to=valid_to))
    out.sort(key=_sort_key)
    return out[:request.per_node_limit + 1]


def read_memory_evidence_handles(store, request: EvidenceHandleRequestV1,
                                 *, source_kind: str = "cache") -> EvidenceHandleResultV1:
    started = _now()
    try:
        nodes, edges = store._nodes, store._edges
        found = {node_id for node_id in request.node_ids
                 if node_id in nodes and nodes[node_id].node_kind in SEMANTIC_KINDS}
        pairs: dict[str, list] = {node_id: [] for node_id in found}
        for edge in edges.values():
            for here, there in ((edge.source.node_id, edge.target.node_id),
                                (edge.target.node_id, edge.source.node_id)):
                if here in pairs and there in nodes:
                    pairs[here].append((edge, nodes[there]))
        per_node = {node_id: select_handles(request, node_id, items) for node_id, items in pairs.items()}
        return _result(request, source_kind=source_kind, started=started, found=found, per_node=per_node)
    except Exception as exc:  # noqa: BLE001 - fail closed, typed reason only
        return _failed(source_kind, started, exc)


# One Cypher read for every requested node. The WITH barrier binds the
# indexed node_id seek before the expand (see neighborhood_backends.match).
EVIDENCE_HANDLES_CYPHER = (
    "MATCH (n:SubstrateNode) WHERE n.node_id IN $node_ids "
    "AND n.node_kind IN ['concept', 'entity'] WITH n "
    "OPTIONAL MATCH (n)-[e]-(ev:SubstrateNode) WHERE e.substrate_edge = true "
    "AND ev.node_kind = 'evidence' "
    "AND ((e.predicate = 'observed_in' AND startNode(e) = n) "
    "OR (e.predicate = 'supports' AND endNode(e) = n)) "
    "AND coalesce(e.valid_from, e.observed_at) <= $at "
    "AND (e.valid_to IS NULL OR e.valid_to > $at) "
    "{type_filter}"
    "WITH n, e, ev ORDER BY n.node_id, coalesce(e.valid_from, e.observed_at) DESC, e.edge_id "
    "WITH n, collect(CASE WHEN e IS NULL THEN NULL ELSE [e.edge_id, e.predicate, ev.node_id, "
    "ev.evidence_type, ev.content_ref, e.observed_at, e.valid_from, e.valid_to] END) AS items "
    "RETURN n.node_id AS node_id, items[0..$take] AS items"
)


def _dt(value) -> datetime | None:
    if value in (None, ""):
        return None
    return _utc(datetime.fromisoformat(str(value)))


def read_falkor_evidence_handles(store, request: EvidenceHandleRequestV1) -> EvidenceHandleResultV1:
    from .falkor_store import _normalize_rows

    started = _now()
    try:
        params = {"node_ids": sorted(set(request.node_ids)), "at": request.at_or_now().isoformat(),
                  "take": request.per_node_limit + 1}
        type_filter = ""
        if request.evidence_types is not None:
            type_filter = "AND ev.evidence_type IN $evidence_types "
            params["evidence_types"] = list(request.evidence_types)
        rows = _normalize_rows(
            store._client.graph_query(EVIDENCE_HANDLES_CYPHER.format(type_filter=type_filter), params=params),
            fields=("node_id", "items"))
        found: set[str] = set()
        per_node: dict[str, list[EvidenceHandleV1]] = {}
        for row in rows:
            node_id = str(row["node_id"])
            if node_id in found or node_id not in params["node_ids"]:
                raise ValueError("duplicate_or_unexpected_node")
            found.add(node_id)
            items = row.get("items") or []
            if len(items) > params["take"]:
                raise ValueError("handle_page_exceeds_limit")
            per_node[node_id] = [EvidenceHandleV1(
                node_id=node_id, edge_id=str(item[0]), predicate=str(item[1]),
                evidence_node_id=str(item[2]), evidence_type=str(item[3]), content_ref=str(item[4]),
                observed_at=_dt(item[5]), valid_from=_dt(item[6]), valid_to=_dt(item[7]))
                for item in items]
        return _result(request, source_kind="falkor", started=started, found=found, per_node=per_node)
    except Exception as exc:  # noqa: BLE001 - fail closed, typed reason only
        return _failed("falkor", started, exc)


# Per-node SPARQL candidate scan cap. Exceeding it fails closed rather than
# silently dropping candidates the reference rule would have ranked.
SPARQL_CANDIDATE_CAP = 4096


def read_sparql_evidence_handles(store, request: EvidenceHandleRequestV1) -> EvidenceHandleResultV1:
    from .graphdb_store import NODE_ADAPTER, ORION_SUBSTRATE_NS
    from .neighborhood_backends import sparql_nodes

    source_kind = store._result_source_kind
    started = _now()
    prefix = f"PREFIX orion: <{ORION_SUBSTRATE_NS}>\n"
    graph = f"GRAPH <{store._cfg.graph_uri}>"
    lit = store._lit
    try:
        ids = sorted(set(request.node_ids))
        nodes = sparql_nodes(store, ids)
        if len({node.node_id for node in nodes}) != len(nodes):
            raise ValueError("duplicate_node_id")
        found = {node.node_id for node in nodes if node.node_kind in SEMANTIC_KINDS and node.node_id in ids}
        per_node = {}
        for node_id in sorted(found):
            node = lit(node_id)
            rows = store._select(prefix + f"SELECT ?edge_json ?evidence_json WHERE {{ {graph} {{ "
                "?edge a orion:SubstrateEdge ; orion:sourceNodeId ?source_id ; "
                "orion:targetNodeId ?target_id ; orion:predicate ?predicate ; orion:payloadJson ?edge_json . "
                "?ev orion:nodeId ?evidence_id ; orion:nodeKind \"evidence\" ; orion:payloadJson ?evidence_json . "
                f"FILTER((?predicate = \"observed_in\" && ?source_id = {node} && ?target_id = ?evidence_id) || "
                f"(?predicate = \"supports\" && ?target_id = {node} && ?source_id = ?evidence_id)) "
                f"}} }} LIMIT {SPARQL_CANDIDATE_CAP + 1}")
            if len(rows) > SPARQL_CANDIDATE_CAP:
                raise ValueError("evidence_candidate_cap_exceeded")
            pairs = [(SubstrateEdgeV1.model_validate_json(store._binding_str(row, "edge_json")),
                      NODE_ADAPTER.validate_json(store._binding_str(row, "evidence_json"))) for row in rows]
            per_node[node_id] = select_handles(request, node_id, pairs)
        return _result(request, source_kind=source_kind, started=started, found=found, per_node=per_node)
    except Exception as exc:  # noqa: BLE001 - fail closed, typed reason only
        return _failed(source_kind, started, exc)
