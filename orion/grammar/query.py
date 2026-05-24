"""Read-only grammar trace queries and graph materialization (spec §11)."""

from __future__ import annotations

from collections import deque
from typing import Any

from app.models.grammar_trace import (
    GrammarAtomSQL,
    GrammarCompactionSQL,
    GrammarEdgeSQL,
    GrammarEventSQL,
    GrammarProjectionSQL,
    GrammarTemporalHopSQL,
    GrammarTraceSQL,
)
from orion.grammar import graph_view


def _trace_to_dict(row: GrammarTraceSQL) -> dict[str, Any]:
    return {
        "trace_id": row.trace_id,
        "trace_type": row.trace_type,
        "session_id": row.session_id,
        "turn_id": row.turn_id,
        "root_event_id": row.root_event_id,
        "started_at": row.started_at,
        "ended_at": row.ended_at,
        "status": row.status,
        "summary": row.summary,
        "created_at": row.created_at,
    }


def _atom_to_dict(row: GrammarAtomSQL) -> dict[str, Any]:
    return {
        "atom_id": row.atom_id,
        "trace_id": row.trace_id,
        "atom_type": row.atom_type,
        "semantic_role": row.semantic_role,
        "layer": row.layer,
        "dimensions": list(row.dimensions or []),
        "summary": row.summary,
        "text_value": row.text_value,
        "confidence": row.confidence,
        "salience": row.salience,
        "uncertainty": row.uncertainty,
        "time_start": row.time_start,
        "time_end": row.time_end,
        "source_event_id": row.source_event_id,
        "payload_ref": row.payload_ref,
        "renderer_hint": row.renderer_hint,
        "atom_json": row.atom_json,
        "created_at": row.created_at,
    }


def _edge_to_dict(row: GrammarEdgeSQL) -> dict[str, Any]:
    return {
        "edge_id": row.edge_id,
        "trace_id": row.trace_id,
        "from_atom_id": row.from_atom_id,
        "to_atom_id": row.to_atom_id,
        "relation_type": row.relation_type,
        "confidence": row.confidence,
        "salience": row.salience,
        "layer_from": row.layer_from,
        "layer_to": row.layer_to,
        "temporal_relation": row.temporal_relation,
        "evidence_event_ids": list(row.evidence_event_ids or []),
        "edge_json": row.edge_json,
        "created_at": row.created_at,
    }


def _hop_to_dict(row: GrammarTemporalHopSQL) -> dict[str, Any]:
    return {
        "hop_id": row.hop_id,
        "trace_id": row.trace_id,
        "from_atom_id": row.from_atom_id,
        "to_atom_id": row.to_atom_id,
        "hop_type": row.hop_type,
        "direction": row.direction,
        "reason": row.reason,
        "confidence": row.confidence,
        "turn_distance": row.turn_distance,
        "session_distance": row.session_distance,
        "target_time_start": row.target_time_start,
        "target_time_end": row.target_time_end,
        "hop_json": row.hop_json,
        "created_at": row.created_at,
    }


def _compaction_to_dict(row: GrammarCompactionSQL) -> dict[str, Any]:
    return {
        "compaction_id": row.compaction_id,
        "trace_id": row.trace_id,
        "source_atom_ids": list(row.source_atom_ids or []),
        "output_atom_id": row.output_atom_id,
        "compaction_type": row.compaction_type,
        "method": row.method,
        "summary": row.summary,
        "preserves": list(row.preserves or []),
        "drops": list(row.drops or []),
        "confidence": row.confidence,
        "compaction_json": row.compaction_json,
        "created_at": row.created_at,
    }


def _projection_to_dict(row: GrammarProjectionSQL) -> dict[str, Any]:
    return {
        "projection_id": row.projection_id,
        "trace_id": row.trace_id,
        "source_atom_ids": list(row.source_atom_ids or []),
        "projection_type": row.projection_type,
        "summary": row.summary,
        "confidence": row.confidence,
        "expires_at": row.expires_at,
        "projected_atom_id": row.projected_atom_id,
        "projection_json": row.projection_json,
        "created_at": row.created_at,
    }


def _event_to_dict(row: GrammarEventSQL) -> dict[str, Any]:
    return {
        "event_id": row.event_id,
        "trace_id": row.trace_id,
        "event_kind": row.event_kind,
        "session_id": row.session_id,
        "turn_id": row.turn_id,
        "layer": row.layer,
        "dimensions": list(row.dimensions or []),
        "emitted_at": row.emitted_at,
        "source_service": row.source_service,
        "source_component": row.source_component,
        "source_event_id": row.source_event_id,
        "payload_ref": row.payload_ref,
        "event_json": row.event_json,
        "created_at": row.created_at,
    }


def _count_atoms(session: Any, trace_id: str) -> int:
    return (
        session.query(GrammarAtomSQL)
        .filter(GrammarAtomSQL.trace_id == trace_id)
        .count()
    )


def _count_edges(session: Any, trace_id: str) -> int:
    return (
        session.query(GrammarEdgeSQL)
        .filter(GrammarEdgeSQL.trace_id == trace_id)
        .count()
    )


def _distinct_layers(session: Any, trace_id: str) -> int:
    rows = (
        session.query(GrammarAtomSQL.layer)
        .filter(GrammarAtomSQL.trace_id == trace_id)
        .distinct()
        .all()
    )
    return len(rows)


def _distinct_dimensions(session: Any, trace_id: str) -> int:
    atoms = (
        session.query(GrammarAtomSQL.dimensions)
        .filter(GrammarAtomSQL.trace_id == trace_id)
        .all()
    )
    dims: set[str] = set()
    for (dimensions,) in atoms:
        dims.update(dimensions or [])
    return len(dims)


def list_traces(
    session: Any,
    session_id: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    q = session.query(GrammarTraceSQL)
    if session_id is not None:
        q = q.filter(GrammarTraceSQL.session_id == session_id)
    rows = q.order_by(GrammarTraceSQL.started_at.desc()).limit(limit).all()

    items: list[dict[str, Any]] = []
    for row in rows:
        trace_id = row.trace_id
        items.append(
            {
                **_trace_to_dict(row),
                "atom_count": _count_atoms(session, trace_id),
                "edge_count": _count_edges(session, trace_id),
                "layer_count": _distinct_layers(session, trace_id),
                "dimension_count": _distinct_dimensions(session, trace_id),
            }
        )
    return items


def get_trace(session: Any, trace_id: str) -> dict[str, Any] | None:
    trace_row = (
        session.query(GrammarTraceSQL)
        .filter(GrammarTraceSQL.trace_id == trace_id)
        .first()
    )
    if trace_row is None:
        return None

    atoms = (
        session.query(GrammarAtomSQL)
        .filter(GrammarAtomSQL.trace_id == trace_id)
        .order_by(GrammarAtomSQL.created_at)
        .all()
    )
    edges = (
        session.query(GrammarEdgeSQL)
        .filter(GrammarEdgeSQL.trace_id == trace_id)
        .order_by(GrammarEdgeSQL.created_at)
        .all()
    )
    hops = (
        session.query(GrammarTemporalHopSQL)
        .filter(GrammarTemporalHopSQL.trace_id == trace_id)
        .order_by(GrammarTemporalHopSQL.created_at)
        .all()
    )
    compactions = (
        session.query(GrammarCompactionSQL)
        .filter(GrammarCompactionSQL.trace_id == trace_id)
        .order_by(GrammarCompactionSQL.created_at)
        .all()
    )
    projections = (
        session.query(GrammarProjectionSQL)
        .filter(GrammarProjectionSQL.trace_id == trace_id)
        .order_by(GrammarProjectionSQL.created_at)
        .all()
    )

    return {
        "trace": _trace_to_dict(trace_row),
        "atoms": [_atom_to_dict(a) for a in atoms],
        "edges": [_edge_to_dict(e) for e in edges],
        "temporal_hops": [_hop_to_dict(h) for h in hops],
        "compactions": [_compaction_to_dict(c) for c in compactions],
        "projections": [_projection_to_dict(p) for p in projections],
        "layer_summary": graph_view.build_layer_summary(atoms),
        "dimension_summary": graph_view.build_dimension_summary(atoms),
    }


def _bfs_atom_ids(
    edges: list[GrammarEdgeSQL],
    *,
    seed_ids: set[str],
    depth: int,
    direction: str = "both",
) -> set[str]:
    if depth < 0:
        return set()

    adj_out: dict[str, set[str]] = {}
    adj_in: dict[str, set[str]] = {}
    for edge in edges:
        adj_out.setdefault(edge.from_atom_id, set()).add(edge.to_atom_id)
        adj_in.setdefault(edge.to_atom_id, set()).add(edge.from_atom_id)

    visited = set(seed_ids)
    frontier = set(seed_ids)
    for _ in range(depth):
        if not frontier:
            break
        nxt: set[str] = set()
        for atom_id in frontier:
            if direction in ("both", "out"):
                nxt.update(adj_out.get(atom_id, set()) - visited)
            if direction in ("both", "in"):
                nxt.update(adj_in.get(atom_id, set()) - visited)
        visited.update(nxt)
        frontier = nxt
    return visited


def _trace_graph_atom_ids(
    atoms: list[GrammarAtomSQL],
    edges: list[GrammarEdgeSQL],
    *,
    depth: int,
) -> set[str]:
    all_ids = {a.atom_id for a in atoms}
    if not all_ids:
        return all_ids

    targets = {e.to_atom_id for e in edges}
    roots = {a.atom_id for a in atoms if a.atom_id not in targets}
    if not roots:
        roots = all_ids

    if depth <= 0:
        return roots

    reachable = _bfs_atom_ids(edges, seed_ids=roots, depth=depth, direction="out")
    reachable.update(roots)
    nodes_in_edges = {e.from_atom_id for e in edges} | {e.to_atom_id for e in edges}
    isolated = all_ids - nodes_in_edges
    reachable |= isolated
    return reachable & all_ids


def get_trace_graph(
    session: Any,
    trace_id: str,
    layout: str = "layered",
    depth: int = 2,
) -> dict[str, Any] | None:
    trace_row = (
        session.query(GrammarTraceSQL)
        .filter(GrammarTraceSQL.trace_id == trace_id)
        .first()
    )
    if trace_row is None:
        return None

    atoms = (
        session.query(GrammarAtomSQL)
        .filter(GrammarAtomSQL.trace_id == trace_id)
        .all()
    )
    edges = (
        session.query(GrammarEdgeSQL)
        .filter(GrammarEdgeSQL.trace_id == trace_id)
        .all()
    )

    atom_ids = _trace_graph_atom_ids(atoms, edges, depth=depth)
    selected_atoms = [a for a in atoms if a.atom_id in atom_ids]
    selected_edges = [
        e
        for e in edges
        if e.from_atom_id in atom_ids and e.to_atom_id in atom_ids
    ]

    return {
        "nodes": [
            graph_view.atom_row_to_node(a, layout=layout) for a in selected_atoms
        ],
        "edges": [graph_view.edge_row_to_edge(e) for e in selected_edges],
        "groups": {
            "layers": graph_view.build_layer_groups(selected_atoms),
            "dimensions": graph_view.build_dimension_groups(selected_atoms),
        },
    }


def get_atom_neighborhood(
    session: Any,
    atom_id: str,
    depth: int = 2,
    direction: str = "both",
) -> dict[str, Any] | None:
    center = (
        session.query(GrammarAtomSQL)
        .filter(GrammarAtomSQL.atom_id == atom_id)
        .first()
    )
    if center is None:
        return None

    trace_id = center.trace_id
    edges = (
        session.query(GrammarEdgeSQL)
        .filter(GrammarEdgeSQL.trace_id == trace_id)
        .all()
    )
    atom_ids = _bfs_atom_ids(
        edges,
        seed_ids={atom_id},
        depth=depth,
        direction=direction,
    )
    atoms = (
        session.query(GrammarAtomSQL)
        .filter(GrammarAtomSQL.atom_id.in_(atom_ids))
        .all()
    )
    selected_edges = [
        e
        for e in edges
        if e.from_atom_id in atom_ids and e.to_atom_id in atom_ids
    ]

    return {
        "center_atom_id": atom_id,
        "trace_id": trace_id,
        "depth": depth,
        "direction": direction,
        "nodes": [graph_view.atom_row_to_node(a, layout="layered") for a in atoms],
        "edges": [graph_view.edge_row_to_edge(e) for e in selected_edges],
    }


def get_atom_provenance(session: Any, atom_id: str) -> dict[str, Any] | None:
    atom_row = (
        session.query(GrammarAtomSQL)
        .filter(GrammarAtomSQL.atom_id == atom_id)
        .first()
    )
    if atom_row is None:
        return None

    trace_id = atom_row.trace_id
    source_event = None
    if atom_row.source_event_id:
        ev = (
            session.query(GrammarEventSQL)
            .filter(GrammarEventSQL.event_id == atom_row.source_event_id)
            .first()
        )
        if ev is not None:
            source_event = _event_to_dict(ev)

    incoming = (
        session.query(GrammarEdgeSQL)
        .filter(
            GrammarEdgeSQL.trace_id == trace_id,
            GrammarEdgeSQL.to_atom_id == atom_id,
        )
        .all()
    )
    outgoing = (
        session.query(GrammarEdgeSQL)
        .filter(
            GrammarEdgeSQL.trace_id == trace_id,
            GrammarEdgeSQL.from_atom_id == atom_id,
        )
        .all()
    )
    parent_atom_ids = sorted({e.from_atom_id for e in incoming})
    child_atom_ids = sorted({e.to_atom_id for e in outgoing})

    compactions_in = (
        session.query(GrammarCompactionSQL)
        .filter(
            GrammarCompactionSQL.trace_id == trace_id,
            GrammarCompactionSQL.output_atom_id == atom_id,
        )
        .all()
    )
    compactions_from = (
        session.query(GrammarCompactionSQL)
        .filter(GrammarCompactionSQL.trace_id == trace_id)
        .all()
    )
    compactions_sourcing = [
        c
        for c in compactions_from
        if atom_id in (c.source_atom_ids or [])
    ]

    hops_in = (
        session.query(GrammarTemporalHopSQL)
        .filter(
            GrammarTemporalHopSQL.trace_id == trace_id,
            GrammarTemporalHopSQL.to_atom_id == atom_id,
        )
        .all()
    )
    hops_out = (
        session.query(GrammarTemporalHopSQL)
        .filter(
            GrammarTemporalHopSQL.trace_id == trace_id,
            GrammarTemporalHopSQL.from_atom_id == atom_id,
        )
        .all()
    )

    projections = (
        session.query(GrammarProjectionSQL)
        .filter(GrammarProjectionSQL.trace_id == trace_id)
        .all()
    )
    projections_related = [
        p
        for p in projections
        if atom_id in (p.source_atom_ids or [])
        or p.projected_atom_id == atom_id
    ]

    return {
        "atom": _atom_to_dict(atom_row),
        "source_event": source_event,
        "parent_atom_ids": parent_atom_ids,
        "child_atom_ids": child_atom_ids,
        "incoming_edges": [_edge_to_dict(e) for e in incoming],
        "outgoing_edges": [_edge_to_dict(e) for e in outgoing],
        "compactions_into": [_compaction_to_dict(c) for c in compactions_in],
        "compactions_from_sources": [_compaction_to_dict(c) for c in compactions_sourcing],
        "temporal_hops_in": [_hop_to_dict(h) for h in hops_in],
        "temporal_hops_out": [_hop_to_dict(h) for h in hops_out],
        "projections": [_projection_to_dict(p) for p in projections_related],
    }


def get_temporal_path(
    session: Any,
    atom_id: str,
    direction: str = "backward",
    limit: int = 25,
) -> dict[str, Any] | None:
    atom_row = (
        session.query(GrammarAtomSQL)
        .filter(GrammarAtomSQL.atom_id == atom_id)
        .first()
    )
    if atom_row is None:
        return None

    trace_id = atom_row.trace_id
    hops = (
        session.query(GrammarTemporalHopSQL)
        .filter(GrammarTemporalHopSQL.trace_id == trace_id)
        .all()
    )

    backward: dict[str, list[GrammarTemporalHopSQL]] = {}
    forward: dict[str, list[GrammarTemporalHopSQL]] = {}
    lateral: dict[str, list[GrammarTemporalHopSQL]] = {}
    for hop in hops:
        if hop.direction == "backward":
            backward.setdefault(hop.from_atom_id, []).append(hop)
        elif hop.direction == "forward":
            forward.setdefault(hop.from_atom_id, []).append(hop)
        else:
            lateral.setdefault(hop.from_atom_id, []).append(hop)

    path_hops: list[GrammarTemporalHopSQL] = []
    visited_hops: set[str] = set()
    queue: deque[str] = deque([atom_id])

    while queue and len(path_hops) < limit:
        current = queue.popleft()
        bucket = backward if direction == "backward" else forward if direction == "forward" else lateral
        for hop in bucket.get(current, []):
            if hop.hop_id in visited_hops:
                continue
            visited_hops.add(hop.hop_id)
            path_hops.append(hop)
            if hop.to_atom_id and hop.to_atom_id not in visited_hops:
                queue.append(hop.to_atom_id)
            if len(path_hops) >= limit:
                break

    atom_ids: set[str] = {atom_id}
    for hop in path_hops:
        atom_ids.add(hop.from_atom_id)
        if hop.to_atom_id:
            atom_ids.add(hop.to_atom_id)

    atoms = (
        session.query(GrammarAtomSQL)
        .filter(GrammarAtomSQL.atom_id.in_(atom_ids))
        .all()
    )

    return {
        "start_atom_id": atom_id,
        "trace_id": trace_id,
        "direction": direction,
        "hops": [_hop_to_dict(h) for h in path_hops],
        "atoms": [_atom_to_dict(a) for a in atoms],
    }
