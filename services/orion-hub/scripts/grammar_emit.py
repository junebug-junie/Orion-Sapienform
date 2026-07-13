"""Hub chat turn grammar event emitter.

Pure builder — no I/O, no Redis, no SQL.
Mirrors the pattern from services/orion-biometrics/app/grammar_emit.py.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from orion.schemas.grammar import (
    GrammarAtomV1,
    GrammarEdgeV1,
    GrammarEventV1,
    GrammarProvenanceV1,
)


def _dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def _hash_id(*parts: object, prefix: str) -> str:
    raw = "|".join(str(p) for p in parts)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def _event(
    *,
    event_kind: str,
    trace_id: str,
    emitted_at: datetime,
    observed_at: datetime,
    provenance: GrammarProvenanceV1,
    atom: GrammarAtomV1 | None = None,
    edge: GrammarEdgeV1 | None = None,
    parent_event_id: str | None = None,
    root_event_id: str | None = None,
    layer: str | None = None,
    dimensions: list[str] | None = None,
    _stable_key: str | None = None,
) -> GrammarEventV1:
    body_key = atom.atom_id if atom else edge.edge_id if edge else (_stable_key or uuid4().hex)
    return GrammarEventV1(
        event_id=_hash_id(trace_id, event_kind, body_key, prefix="gev"),
        event_kind=event_kind,  # type: ignore[arg-type]
        trace_id=trace_id,
        parent_event_id=parent_event_id,
        root_event_id=root_event_id,
        emitted_at=emitted_at,
        observed_at=observed_at,
        layer=layer,
        dimensions=dimensions or [],
        atom=atom,
        edge=edge,
        provenance=provenance,
    )


def build_chat_turn_grammar_events(
    *,
    turn_id: str,
    session_id: str,
    node_id: str = "athena",
    word_count: int = 0,
    repair_pressure_level: float = 0.0,
    repair_pressure_confidence: float = 0.0,
    repair_pressure_dims: dict[str, float] | None = None,
    has_repair_signal: bool = False,
    stance_disposition: str | None = None,
    stance_disposition_reasons: list[str] | None = None,
    stance_boundary_register: bool = False,
    observed_at: datetime | None = None,
    code_version: str | None = None,
) -> list[GrammarEventV1]:
    observed_at_ = _dt(observed_at)
    emitted_at = datetime.now(timezone.utc)

    trace_id = f"hub.chat:{node_id}:{turn_id}"

    provenance = GrammarProvenanceV1(
        source_service="orion-hub",
        source_component="hub_chat_grammar_emit",
        source_event_id=turn_id,
        source_trace_id=trace_id,
        source_payload_ref=f"hub.chat:{session_id}:{turn_id}",
        code_version=code_version,
    )

    root_event = _event(
        event_kind="trace_started",
        trace_id=trace_id,
        emitted_at=emitted_at,
        observed_at=observed_at_,
        provenance=provenance,
        layer="chat",
        dimensions=["conversation", "turn", "session"],
        _stable_key=f"{trace_id}:trace_started",
    )
    root_id = root_event.event_id

    def atom_id(role: str) -> str:
        return f"{trace_id}:{role}"

    # Build atoms in order
    session_context_atom = GrammarAtomV1(
        atom_id=atom_id("session_context"),
        trace_id=trace_id,
        atom_type="entity",
        semantic_role="session_context",
        layer="context",
        dimensions=["conversation", "session"],
        summary=f"Chat session {session_id} on {node_id}",
        text_value=session_id,
        confidence=1.0,
        salience=0.6,
        source_event_id=turn_id,
        payload_ref=f"hub.session:{session_id}",
    )

    user_utterance_atom = GrammarAtomV1(
        atom_id=atom_id("user_utterance"),
        trace_id=trace_id,
        atom_type="observation",
        semantic_role="user_utterance",
        layer="raw_input",
        dimensions=["conversation", "turn"],
        summary=f"User message in session {session_id} ({word_count} words)",
        text_value=None,  # NEVER store raw text
        confidence=1.0,
        salience=min(1.0, 0.4 + word_count / 200.0),
        source_event_id=turn_id,
        payload_ref=f"hub.chat:{session_id}:{turn_id}",
    )

    events: list[GrammarEventV1] = [root_event]

    # Emit atom events in order: session_context, user_utterance
    for atom in (session_context_atom, user_utterance_atom):
        events.append(
            _event(
                event_kind="atom_emitted",
                trace_id=trace_id,
                emitted_at=emitted_at,
                observed_at=observed_at_,
                provenance=provenance,
                atom=atom,
                parent_event_id=root_id,
                root_event_id=root_id,
                layer=atom.layer,
                dimensions=atom.dimensions,
            )
        )

    # Optionally emit repair_signal atom
    repair_signal_atom: GrammarAtomV1 | None = None
    if has_repair_signal:
        repair_signal_atom = GrammarAtomV1(
            atom_id=atom_id("repair_signal"),
            trace_id=trace_id,
            atom_type="signal",
            semantic_role="repair_signal",
            layer="organ_signal",
            dimensions=["repair", "pressure"],
            summary=(
                f"Repair pressure level={repair_pressure_level:.2f} "
                f"confidence={repair_pressure_confidence:.2f}"
            ),
            text_value=None,
            confidence=repair_pressure_confidence,
            salience=repair_pressure_level,
            source_event_id=turn_id,
            payload_ref=f"hub.repair_pressure:{turn_id}",
        )
        events.append(
            _event(
                event_kind="atom_emitted",
                trace_id=trace_id,
                emitted_at=emitted_at,
                observed_at=observed_at_,
                provenance=provenance,
                atom=repair_signal_atom,
                parent_event_id=root_id,
                root_event_id=root_id,
                layer=repair_signal_atom.layer,
                dimensions=repair_signal_atom.dimensions,
            )
        )

    # Optionally emit stance_disposition atom (Thought's proceed/defer/refuse decision --
    # only meaningful for the unified-turn pipeline, which is the only producer that has
    # a stance decision to report; the classic path leaves stance_disposition=None).
    stance_disposition_atom: GrammarAtomV1 | None = None
    if stance_disposition:
        reasons = stance_disposition_reasons or []
        summary = f"Stance disposition: {stance_disposition}"
        if reasons:
            summary += f" ({'; '.join(reasons)})"
        if stance_boundary_register:
            summary += " [boundary_register]"
        stance_disposition_atom = GrammarAtomV1(
            atom_id=atom_id("stance_disposition"),
            trace_id=trace_id,
            atom_type="signal",
            semantic_role="stance_disposition",
            layer="organ_signal",
            dimensions=["conversation", "turn", "stance"],
            summary=summary,
            text_value=stance_disposition,
            confidence=1.0,
            salience=1.0 if stance_disposition != "proceed" else 0.3,
            source_event_id=turn_id,
            payload_ref=f"hub.stance:{turn_id}",
        )
        events.append(
            _event(
                event_kind="atom_emitted",
                trace_id=trace_id,
                emitted_at=emitted_at,
                observed_at=observed_at_,
                provenance=provenance,
                atom=stance_disposition_atom,
                parent_event_id=root_id,
                root_event_id=root_id,
                layer=stance_disposition_atom.layer,
                dimensions=stance_disposition_atom.dimensions,
            )
        )

    # Edge: user_utterance → derived_from → session_context
    edge_utterance_context = GrammarEdgeV1(
        edge_id=f"{trace_id}:edge:user_utterance:derived_from:session_context",
        trace_id=trace_id,
        from_atom_id=user_utterance_atom.atom_id,
        to_atom_id=session_context_atom.atom_id,
        relation_type="derived_from",
        confidence=0.9,
        salience=0.7,
        layer_from=user_utterance_atom.layer,
        layer_to=session_context_atom.layer,
        evidence_event_ids=[turn_id],
    )
    events.append(
        _event(
            event_kind="edge_emitted",
            trace_id=trace_id,
            emitted_at=emitted_at,
            observed_at=observed_at_,
            provenance=provenance,
            edge=edge_utterance_context,
            parent_event_id=root_id,
            root_event_id=root_id,
            layer=edge_utterance_context.layer_to,
            dimensions=["conversation", "turn", "session"],
        )
    )

    # Edge: repair_signal → derived_from → user_utterance (only if repair_signal present)
    if repair_signal_atom is not None:
        edge_repair_utterance = GrammarEdgeV1(
            edge_id=f"{trace_id}:edge:repair_signal:derived_from:user_utterance",
            trace_id=trace_id,
            from_atom_id=repair_signal_atom.atom_id,
            to_atom_id=user_utterance_atom.atom_id,
            relation_type="derived_from",
            confidence=repair_pressure_confidence,
            salience=repair_pressure_level,
            layer_from=repair_signal_atom.layer,
            layer_to=user_utterance_atom.layer,
            evidence_event_ids=[turn_id],
        )
        events.append(
            _event(
                event_kind="edge_emitted",
                trace_id=trace_id,
                emitted_at=emitted_at,
                observed_at=observed_at_,
                provenance=provenance,
                edge=edge_repair_utterance,
                parent_event_id=root_id,
                root_event_id=root_id,
                layer=edge_repair_utterance.layer_to,
                dimensions=["conversation", "turn", "repair"],
            )
        )

    # Edge: stance_disposition → derived_from → user_utterance
    if stance_disposition_atom is not None:
        edge_stance_utterance = GrammarEdgeV1(
            edge_id=f"{trace_id}:edge:stance_disposition:derived_from:user_utterance",
            trace_id=trace_id,
            from_atom_id=stance_disposition_atom.atom_id,
            to_atom_id=user_utterance_atom.atom_id,
            relation_type="derived_from",
            confidence=1.0,
            salience=stance_disposition_atom.salience or 0.0,
            layer_from=stance_disposition_atom.layer,
            layer_to=user_utterance_atom.layer,
            evidence_event_ids=[turn_id],
        )
        events.append(
            _event(
                event_kind="edge_emitted",
                trace_id=trace_id,
                emitted_at=emitted_at,
                observed_at=observed_at_,
                provenance=provenance,
                edge=edge_stance_utterance,
                parent_event_id=root_id,
                root_event_id=root_id,
                layer=edge_stance_utterance.layer_to,
                dimensions=["conversation", "turn", "stance"],
            )
        )

    events.append(
        _event(
            event_kind="trace_ended",
            trace_id=trace_id,
            emitted_at=datetime.now(timezone.utc),
            observed_at=observed_at_,
            provenance=provenance,
            parent_event_id=root_id,
            root_event_id=root_id,
            layer="chat",
            dimensions=["conversation", "turn", "session"],
            _stable_key=f"{trace_id}:trace_ended",
        )
    )

    return events
