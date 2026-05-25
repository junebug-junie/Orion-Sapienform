from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from uuid import uuid4

from orion.schemas.grammar import (
    GrammarAtomV1,
    GrammarEdgeV1,
    GrammarEventV1,
    GrammarProvenanceV1,
)

PRESSURE_SOURCE_SERVICE = "orion-substrate-organs"
PRESSURE_SOURCE_COMPONENT = "biometrics_pressure"


def _hash_id(*parts: object, prefix: str) -> str:
    raw = "|".join(str(p) for p in parts)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def _safe_ts(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def build_pressure_candidate_events(
    *,
    node_id: str,
    semantic_role: str,
    evidence_event_ids: list[str],
    confidence: float,
    observed_at: datetime | None = None,
) -> list[GrammarEventV1]:
    """Build a pressure candidate trace: trace_started + atom + edge + trace_ended."""
    clock = observed_at or datetime.now(timezone.utc)
    if clock.tzinfo is None:
        clock = clock.replace(tzinfo=timezone.utc)
    ts = _safe_ts(clock)
    trace_id = f"substrate.pressure:{node_id}:{ts}"

    provenance = GrammarProvenanceV1(
        source_service=PRESSURE_SOURCE_SERVICE,
        source_component=PRESSURE_SOURCE_COMPONENT,
        source_trace_id=trace_id,
    )

    root = GrammarEventV1(
        event_id=_hash_id(trace_id, "trace_started", prefix="gev"),
        event_kind="trace_started",
        trace_id=trace_id,
        emitted_at=clock,
        observed_at=clock,
        layer="substrate",
        dimensions=["pressure", "node"],
        provenance=provenance,
    )
    root_id = root.event_id

    atom = GrammarAtomV1(
        atom_id=f"{trace_id}:{semantic_role}",
        trace_id=trace_id,
        atom_type="salience_marker",
        semantic_role=semantic_role,
        layer="substrate",
        dimensions=["pressure", "node"],
        summary=f"{semantic_role} for {node_id}",
        text_value=node_id,
        confidence=confidence,
        salience=confidence,
    )
    atom_event = GrammarEventV1(
        event_id=_hash_id(trace_id, semantic_role, atom.atom_id, prefix="gev"),
        event_kind="atom_emitted",
        trace_id=trace_id,
        parent_event_id=root_id,
        root_event_id=root_id,
        emitted_at=clock,
        observed_at=clock,
        layer="substrate",
        dimensions=["pressure", "node"],
        atom=atom,
        provenance=provenance,
    )

    edge = GrammarEdgeV1(
        edge_id=f"{trace_id}:edge:evidence:{uuid4().hex[:8]}",
        trace_id=trace_id,
        from_atom_id=atom.atom_id,
        to_atom_id=atom.atom_id,
        relation_type="derived_from",
        confidence=confidence,
        evidence_event_ids=list(evidence_event_ids),
    )
    edge_event = GrammarEventV1(
        event_id=_hash_id(trace_id, "edge", edge.edge_id, prefix="gev"),
        event_kind="edge_emitted",
        trace_id=trace_id,
        parent_event_id=root_id,
        root_event_id=root_id,
        emitted_at=clock,
        observed_at=clock,
        layer="substrate",
        dimensions=["pressure", "node"],
        edge=edge,
        provenance=provenance,
    )

    end = GrammarEventV1(
        event_id=_hash_id(trace_id, "trace_ended", prefix="gev"),
        event_kind="trace_ended",
        trace_id=trace_id,
        parent_event_id=root_id,
        root_event_id=root_id,
        emitted_at=clock,
        observed_at=clock,
        layer="substrate",
        dimensions=["pressure", "node"],
        provenance=provenance,
    )

    return [root, atom_event, edge_event, end]
