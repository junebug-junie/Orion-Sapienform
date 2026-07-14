"""Append-only grammar ledger: trace upsert, event insert, derived child rows."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.models.grammar_trace import (
    GrammarAtomSQL,
    GrammarCompactionSQL,
    GrammarEdgeSQL,
    GrammarEventSQL,
    GrammarProjectionSQL,
    GrammarTemporalHopSQL,
    GrammarTraceSQL,
)
from orion.schemas.grammar import (
    GrammarAtomV1,
    GrammarCompactionV1,
    GrammarEdgeV1,
    GrammarEventV1,
    GrammarProjectionV1,
    TemporalHopV1,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _created_at(event: GrammarEventV1) -> datetime:
    return event.observed_at or event.emitted_at or _utc_now()


def _trace_type_from_event(event: GrammarEventV1) -> str:
    prov = event.provenance
    if prov.source_component:
        return prov.source_component
    if event.layer:
        return event.layer
    return "unknown"


def _upsert_trace(session: Any, event: GrammarEventV1) -> None:
    """Upsert grammar_traces without a read-before-write SELECT."""
    now = _utc_now()
    trace_type = _trace_type_from_event(event)
    root_event_id = event.root_event_id or event.event_id
    table = GrammarTraceSQL.__table__
    # started_at/ended_at prefer event.observed_at (real occurrence time,
    # via _created_at()'s existing fallback chain) over event.emitted_at
    # (bus-publish time -- always uniform across one flush batch, by
    # design, never a meaningful trace-duration signal). Found by review
    # 2026-07-14: producers whose observed_at is itself accurate per-atom
    # (see services/orion-cortex-exec/app/grammar_emit.py) now get a real
    # started_at/ended_at spread here too; producers that don't yet have
    # accurate observed_at are no worse off than before.
    trace_ts = _created_at(event)
    values = {
        "trace_id": event.trace_id,
        "trace_type": trace_type,
        "session_id": event.session_id,
        "turn_id": event.turn_id,
        "root_event_id": root_event_id,
        "started_at": trace_ts,
        "ended_at": None,
        "status": "open",
        "summary": None,
        "created_at": now,
    }
    if event.event_kind == "trace_started":
        stmt = pg_insert(table).values(**values).on_conflict_do_update(
            index_elements=["trace_id"],
            set_={
                "trace_type": trace_type,
                "session_id": event.session_id,
                "turn_id": event.turn_id,
                "root_event_id": root_event_id,
                "started_at": trace_ts,
                "status": "open",
            },
        )
    elif event.event_kind == "trace_ended":
        stmt = pg_insert(table).values(**values).on_conflict_do_update(
            index_elements=["trace_id"],
            set_={"status": "closed", "ended_at": trace_ts},
        )
    else:
        stmt = pg_insert(table).values(**values).on_conflict_do_nothing(
            index_elements=["trace_id"],
        )
    session.execute(stmt)


def _event_row(event: GrammarEventV1) -> dict[str, Any]:
    prov = event.provenance
    return {
        "event_id": event.event_id,
        "trace_id": event.trace_id,
        "parent_event_id": event.parent_event_id,
        "root_event_id": event.root_event_id,
        "event_kind": event.event_kind,
        "session_id": event.session_id,
        "turn_id": event.turn_id,
        "correlation_id": event.correlation_id,
        "layer": event.layer,
        "dimensions": list(event.dimensions),
        "emitted_at": event.emitted_at,
        "observed_at": event.observed_at,
        "source_service": prov.source_service,
        "source_component": prov.source_component,
        "source_event_id": prov.source_event_id,
        "payload_ref": prov.source_payload_ref,
        "event_json": event.model_dump(mode="json"),
        "created_at": _created_at(event),
    }


def _atom_row(atom: GrammarAtomV1, event: GrammarEventV1) -> dict[str, Any]:
    time_start = None
    time_end = None
    if atom.time_range is not None:
        time_start = atom.time_range.start
        time_end = atom.time_range.end
    return {
        "atom_id": atom.atom_id,
        "trace_id": atom.trace_id,
        "atom_type": atom.atom_type,
        "semantic_role": atom.semantic_role,
        "layer": atom.layer,
        "dimensions": list(atom.dimensions),
        "summary": atom.summary,
        "text_value": atom.text_value,
        "confidence": atom.confidence,
        "salience": atom.salience,
        "uncertainty": atom.uncertainty,
        "time_start": time_start,
        "time_end": time_end,
        "source_event_id": atom.source_event_id or event.event_id,
        "payload_ref": atom.payload_ref,
        "renderer_hint": atom.renderer_hint,
        "atom_json": atom.model_dump(mode="json"),
        "created_at": _created_at(event),
    }


def _edge_row(edge: GrammarEdgeV1, event: GrammarEventV1) -> dict[str, Any]:
    return {
        "edge_id": edge.edge_id,
        "trace_id": edge.trace_id,
        "from_atom_id": edge.from_atom_id,
        "to_atom_id": edge.to_atom_id,
        "relation_type": edge.relation_type,
        "confidence": edge.confidence,
        "salience": edge.salience,
        "layer_from": edge.layer_from,
        "layer_to": edge.layer_to,
        "temporal_relation": edge.temporal_relation,
        "evidence_event_ids": list(edge.evidence_event_ids),
        "edge_json": edge.model_dump(mode="json"),
        "created_at": _created_at(event),
    }


def _temporal_hop_row(hop: TemporalHopV1, event: GrammarEventV1) -> dict[str, Any]:
    target_time_start = None
    target_time_end = None
    if hop.target_time_range is not None:
        target_time_start = hop.target_time_range.start
        target_time_end = hop.target_time_range.end
    return {
        "hop_id": hop.hop_id,
        "trace_id": hop.trace_id,
        "from_atom_id": hop.from_atom_id,
        "to_atom_id": hop.to_atom_id,
        "hop_type": hop.hop_type,
        "direction": hop.direction,
        "reason": hop.reason,
        "confidence": hop.confidence,
        "turn_distance": hop.turn_distance,
        "session_distance": hop.session_distance,
        "target_time_start": target_time_start,
        "target_time_end": target_time_end,
        "hop_json": hop.model_dump(mode="json"),
        "created_at": _created_at(event),
    }


def _compaction_row(compaction: GrammarCompactionV1, event: GrammarEventV1) -> dict[str, Any]:
    return {
        "compaction_id": compaction.compaction_id,
        "trace_id": compaction.trace_id,
        "source_atom_ids": list(compaction.source_atom_ids),
        "output_atom_id": compaction.output_atom_id,
        "compaction_type": compaction.compaction_type,
        "method": compaction.method,
        "summary": compaction.summary,
        "preserves": list(compaction.preserves),
        "drops": list(compaction.drops),
        "confidence": compaction.confidence,
        "compaction_json": compaction.model_dump(mode="json"),
        "created_at": _created_at(event),
    }


def _projection_row(projection: GrammarProjectionV1, event: GrammarEventV1) -> dict[str, Any]:
    return {
        "projection_id": projection.projection_id,
        "trace_id": projection.trace_id,
        "source_atom_ids": list(projection.source_atom_ids),
        "projection_type": projection.projection_type,
        "summary": projection.summary,
        "confidence": projection.confidence,
        "expires_at": projection.expires_at,
        "projected_atom_id": projection.projected_atom_id,
        "projection_json": projection.model_dump(mode="json"),
        "created_at": _created_at(event),
    }


def _bulk_insert_events(session: Any, events: list[GrammarEventV1]) -> list[str]:
    if not events:
        return []
    rows = [_event_row(event) for event in events]
    stmt = (
        pg_insert(GrammarEventSQL.__table__)
        .values(rows)
        .on_conflict_do_nothing(index_elements=["event_id"])
        .returning(GrammarEventSQL.event_id)
    )
    return [row[0] for row in session.execute(stmt).fetchall()]


def _bulk_insert_derived(session: Any, events: list[GrammarEventV1]) -> None:
    atoms: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    hops: list[dict[str, Any]] = []
    compactions: list[dict[str, Any]] = []
    projections: list[dict[str, Any]] = []

    for event in events:
        if event.event_kind == "atom_emitted" and event.atom is not None:
            atoms.append(_atom_row(event.atom, event))
        elif event.event_kind == "edge_emitted" and event.edge is not None:
            edges.append(_edge_row(event.edge, event))
        elif event.event_kind == "temporal_hop_emitted" and event.temporal_hop is not None:
            hops.append(_temporal_hop_row(event.temporal_hop, event))
        elif event.event_kind == "compaction_emitted" and event.compaction is not None:
            compactions.append(_compaction_row(event.compaction, event))
        elif event.event_kind == "projection_emitted" and event.projection is not None:
            projections.append(_projection_row(event.projection, event))

    if atoms:
        session.execute(
            pg_insert(GrammarAtomSQL.__table__)
            .values(atoms)
            .on_conflict_do_nothing(index_elements=["atom_id"])
        )
    if edges:
        session.execute(
            pg_insert(GrammarEdgeSQL.__table__)
            .values(edges)
            .on_conflict_do_nothing(index_elements=["edge_id"])
        )
    if hops:
        session.execute(
            pg_insert(GrammarTemporalHopSQL.__table__)
            .values(hops)
            .on_conflict_do_nothing(index_elements=["hop_id"])
        )
    if compactions:
        session.execute(
            pg_insert(GrammarCompactionSQL.__table__)
            .values(compactions)
            .on_conflict_do_nothing(index_elements=["compaction_id"])
        )
    if projections:
        session.execute(
            pg_insert(GrammarProjectionSQL.__table__)
            .values(projections)
            .on_conflict_do_nothing(index_elements=["projection_id"])
        )


def apply_grammar_event(session: Any, event: GrammarEventV1) -> bool:
    """Append one grammar event. Returns False if event_id already exists (dedupe)."""
    return apply_grammar_trace_batch(session, [event]) == 1


def apply_grammar_trace_batch(session: Any, events: list[GrammarEventV1]) -> int:
    """Append a trace's events in one transaction. Returns count of new events inserted."""
    if not events:
        return 0

    trace_kinds = {"trace_started", "trace_ended"}
    ensured_trace = False
    for event in events:
        if event.event_kind in trace_kinds:
            _upsert_trace(session, event)
            ensured_trace = True
    if not ensured_trace:
        _upsert_trace(session, events[0])
    session.flush()

    inserted_ids = _bulk_insert_events(session, events)
    if inserted_ids:
        inserted = {event_id for event_id in inserted_ids}
        _bulk_insert_derived(session, [event for event in events if event.event_id in inserted])

    return len(inserted_ids)
