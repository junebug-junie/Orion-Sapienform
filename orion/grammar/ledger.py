"""Append-only grammar ledger: trace upsert, event insert, derived child rows."""

from __future__ import annotations

from datetime import datetime, timezone
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


def _event_exists(session: Any, event_id: str) -> bool:
    return (
        session.query(GrammarEventSQL)
        .filter(GrammarEventSQL.event_id == event_id)
        .first()
        is not None
    )


def _trace_type_from_event(event: GrammarEventV1) -> str:
    prov = event.provenance
    if prov.source_component:
        return prov.source_component
    if event.layer:
        return event.layer
    return "unknown"


def _ensure_trace(session: Any, event: GrammarEventV1) -> GrammarTraceSQL:
    row = (
        session.query(GrammarTraceSQL)
        .filter(GrammarTraceSQL.trace_id == event.trace_id)
        .first()
    )
    now = _utc_now()

    if row is None:
        row = GrammarTraceSQL(
            trace_id=event.trace_id,
            trace_type=_trace_type_from_event(event),
            session_id=event.session_id,
            turn_id=event.turn_id,
            root_event_id=event.root_event_id or event.event_id,
            started_at=event.emitted_at,
            status="open",
            created_at=now,
        )
        session.add(row)
        return row

    if event.event_kind == "trace_started":
        row.trace_type = _trace_type_from_event(event)
        if event.session_id is not None:
            row.session_id = event.session_id
        if event.turn_id is not None:
            row.turn_id = event.turn_id
        if event.root_event_id is not None:
            row.root_event_id = event.root_event_id
        row.started_at = event.emitted_at
        row.status = "open"
    elif event.event_kind == "trace_ended":
        row.status = "closed"
        row.ended_at = event.emitted_at

    return row


def _insert_event_row(session: Any, event: GrammarEventV1) -> None:
    prov = event.provenance
    session.add(
        GrammarEventSQL(
            event_id=event.event_id,
            trace_id=event.trace_id,
            parent_event_id=event.parent_event_id,
            root_event_id=event.root_event_id,
            event_kind=event.event_kind,
            session_id=event.session_id,
            turn_id=event.turn_id,
            correlation_id=event.correlation_id,
            layer=event.layer,
            dimensions=list(event.dimensions),
            emitted_at=event.emitted_at,
            observed_at=event.observed_at,
            source_service=prov.source_service,
            source_component=prov.source_component,
            source_event_id=prov.source_event_id,
            payload_ref=prov.source_payload_ref,
            event_json=event.model_dump(mode="json"),
            created_at=_created_at(event),
        )
    )


def _insert_atom(session: Any, atom: GrammarAtomV1, event: GrammarEventV1) -> None:
    time_start = None
    time_end = None
    if atom.time_range is not None:
        time_start = atom.time_range.start
        time_end = atom.time_range.end

    session.add(
        GrammarAtomSQL(
            atom_id=atom.atom_id,
            trace_id=atom.trace_id,
            atom_type=atom.atom_type,
            semantic_role=atom.semantic_role,
            layer=atom.layer,
            dimensions=list(atom.dimensions),
            summary=atom.summary,
            text_value=atom.text_value,
            confidence=atom.confidence,
            salience=atom.salience,
            uncertainty=atom.uncertainty,
            time_start=time_start,
            time_end=time_end,
            source_event_id=atom.source_event_id or event.event_id,
            payload_ref=atom.payload_ref,
            renderer_hint=atom.renderer_hint,
            atom_json=atom.model_dump(mode="json"),
            created_at=_created_at(event),
        )
    )


def _insert_edge(session: Any, edge: GrammarEdgeV1, event: GrammarEventV1) -> None:
    session.add(
        GrammarEdgeSQL(
            edge_id=edge.edge_id,
            trace_id=edge.trace_id,
            from_atom_id=edge.from_atom_id,
            to_atom_id=edge.to_atom_id,
            relation_type=edge.relation_type,
            confidence=edge.confidence,
            salience=edge.salience,
            layer_from=edge.layer_from,
            layer_to=edge.layer_to,
            temporal_relation=edge.temporal_relation,
            evidence_event_ids=list(edge.evidence_event_ids),
            edge_json=edge.model_dump(mode="json"),
            created_at=_created_at(event),
        )
    )


def _insert_temporal_hop(session: Any, hop: TemporalHopV1, event: GrammarEventV1) -> None:
    target_time_start = None
    target_time_end = None
    if hop.target_time_range is not None:
        target_time_start = hop.target_time_range.start
        target_time_end = hop.target_time_range.end

    session.add(
        GrammarTemporalHopSQL(
            hop_id=hop.hop_id,
            trace_id=hop.trace_id,
            from_atom_id=hop.from_atom_id,
            to_atom_id=hop.to_atom_id,
            hop_type=hop.hop_type,
            direction=hop.direction,
            reason=hop.reason,
            confidence=hop.confidence,
            turn_distance=hop.turn_distance,
            session_distance=hop.session_distance,
            target_time_start=target_time_start,
            target_time_end=target_time_end,
            hop_json=hop.model_dump(mode="json"),
            created_at=_created_at(event),
        )
    )


def _insert_compaction(session: Any, compaction: GrammarCompactionV1, event: GrammarEventV1) -> None:
    session.add(
        GrammarCompactionSQL(
            compaction_id=compaction.compaction_id,
            trace_id=compaction.trace_id,
            source_atom_ids=list(compaction.source_atom_ids),
            output_atom_id=compaction.output_atom_id,
            compaction_type=compaction.compaction_type,
            method=compaction.method,
            summary=compaction.summary,
            preserves=list(compaction.preserves),
            drops=list(compaction.drops),
            confidence=compaction.confidence,
            compaction_json=compaction.model_dump(mode="json"),
            created_at=_created_at(event),
        )
    )


def _insert_projection(session: Any, projection: GrammarProjectionV1, event: GrammarEventV1) -> None:
    session.add(
        GrammarProjectionSQL(
            projection_id=projection.projection_id,
            trace_id=projection.trace_id,
            source_atom_ids=list(projection.source_atom_ids),
            projection_type=projection.projection_type,
            summary=projection.summary,
            confidence=projection.confidence,
            expires_at=projection.expires_at,
            projected_atom_id=projection.projected_atom_id,
            projection_json=projection.model_dump(mode="json"),
            created_at=_created_at(event),
        )
    )


def apply_grammar_event(session: Any, event: GrammarEventV1) -> bool:
    """Append one grammar event. Returns False if event_id already exists (dedupe)."""
    if _event_exists(session, event.event_id):
        return False

    _ensure_trace(session, event)
    _insert_event_row(session, event)

    if event.event_kind == "atom_emitted" and event.atom is not None:
        _insert_atom(session, event.atom, event)
    elif event.event_kind == "edge_emitted" and event.edge is not None:
        _insert_edge(session, event.edge, event)
    elif event.event_kind == "temporal_hop_emitted" and event.temporal_hop is not None:
        _insert_temporal_hop(session, event.temporal_hop, event)
    elif event.event_kind == "compaction_emitted" and event.compaction is not None:
        _insert_compaction(session, event.compaction, event)
    elif event.event_kind == "projection_emitted" and event.projection is not None:
        _insert_projection(session, event.projection, event)

    return True
