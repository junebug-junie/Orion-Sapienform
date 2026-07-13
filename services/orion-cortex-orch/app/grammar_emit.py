"""Orch route-arbitration grammar event emitter.

Pure builder -- no I/O, no Redis, no SQL.

Mirrors services/orion-hub/scripts/grammar_emit.py's pattern
(build_chat_turn_grammar_events) and the _event/_hash_id helper conventions
used by services/orion-cortex-exec/app/grammar_emit.py.

This module turns the route_metadata dict already computed in
orchestrator.call_verb_runtime() (execution_lane, execution_lane_reason,
mind_requested, mind_skip_reason, output_mode -- see the "Route arbitration
facts" block in orchestrator.py) into a two-event shadow GrammarEventV1
trace: does not recompute any arbitration fact, only serializes it.

The trace_id prefix ("orch.route:"), atom semantic_role
("route_arbitration_decided"), and the comma-separated key=value summary
format are a pinned contract consumed by the (separately owned) route_loop
reducer at orion/substrate/route_loop/grammar_extract.py -- do not change
field names or the summary format without updating both sides. See
docs/superpowers/specs/2026-07-12-orch-route-grammar-lane-design.md.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from orion.schemas.grammar import (
    GrammarAtomV1,
    GrammarEventV1,
    GrammarProvenanceV1,
)

# Matches orion/substrate/route_loop/grammar_extract.py's _KV_RE anchor:
# consumer parses summary with re.compile(r"(\w+)=([^,;\s]+)"), so no value
# may contain a comma, semicolon, or whitespace.
_UNSAFE_CHARS_RE = re.compile(r"[,;\s]+")


def _safe(value: Any) -> str:
    """Defensively collapse chars that would break the consumer's kv-parse.

    Today's values all come from internal enums/reason strings, but this
    keeps the summary well-formed even if that ever changes.
    """
    return _UNSAFE_CHARS_RE.sub("_", str(value))


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
    parent_event_id: str | None = None,
    root_event_id: str | None = None,
    layer: str | None = None,
    dimensions: list[str] | None = None,
    session_id: str | None = None,
    turn_id: str | None = None,
    correlation_id: str | None = None,
    _stable_key: str | None = None,
) -> GrammarEventV1:
    body_key = atom.atom_id if atom else (_stable_key or uuid4().hex)
    return GrammarEventV1(
        event_id=_hash_id(trace_id, event_kind, body_key, prefix="gev"),
        event_kind=event_kind,  # type: ignore[arg-type]
        trace_id=trace_id,
        parent_event_id=parent_event_id,
        root_event_id=root_event_id,
        session_id=session_id,
        turn_id=turn_id,
        correlation_id=correlation_id,
        emitted_at=emitted_at,
        observed_at=observed_at,
        layer=layer,
        dimensions=dimensions or [],
        atom=atom,
        provenance=provenance,
    )


def build_route_arbitration_grammar_events(
    *,
    correlation_id: str,
    node_id: str,
    route_metadata: dict[str, Any],
    session_id: str | None = None,
    turn_id: str | None = None,
    observed_at: datetime | None = None,
    code_version: str | None = None,
) -> list[GrammarEventV1]:
    """Build the two-event shadow trace for one turn's route arbitration.

    Pure function: no bus I/O, reuses route_metadata as-is (does not
    recompute execution_lane/mind_requested/output_mode). Returns exactly
    [trace_started, atom_emitted].
    """
    observed_at_ = _dt(observed_at)
    emitted_at = datetime.now(timezone.utc)

    trace_id = f"orch.route:{node_id}:{correlation_id}"

    provenance = GrammarProvenanceV1(
        source_service="orion-cortex-orch",
        source_component="orch_route_grammar_emit",
        source_event_id=correlation_id,
        source_trace_id=trace_id,
        source_payload_ref=f"orch.route:{correlation_id}",
        code_version=code_version,
    )

    root_event = _event(
        event_kind="trace_started",
        trace_id=trace_id,
        emitted_at=emitted_at,
        observed_at=observed_at_,
        provenance=provenance,
        layer="route",
        dimensions=["route", "arbitration"],
        session_id=session_id,
        turn_id=turn_id,
        correlation_id=correlation_id,
        _stable_key=f"{trace_id}:trace_started",
    )
    root_id = root_event.event_id

    summary = (
        f"lane={_safe(route_metadata.get('execution_lane') or 'unknown')},"
        f"lane_reason={_safe(route_metadata.get('execution_lane_reason') or 'unknown')},"
        f"mind_requested={'true' if route_metadata.get('mind_requested') else 'false'},"
        f"mind_skip_reason={_safe(route_metadata.get('mind_skip_reason') or 'none')},"
        f"output_mode={_safe(route_metadata.get('output_mode') or 'unknown')}"
    )

    atom = GrammarAtomV1(
        atom_id=f"{trace_id}:route_arbitration_decided",
        trace_id=trace_id,
        atom_type="reasoning_step",
        semantic_role="route_arbitration_decided",
        layer="route",
        dimensions=["route", "arbitration"],
        summary=summary,
        text_value=None,  # never store raw text -- these are categorical facts, not user content
        confidence=1.0,
        salience=1.0,
        source_event_id=correlation_id,
        payload_ref=f"orch.route:{correlation_id}",
    )

    atom_event = _event(
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
        session_id=session_id,
        turn_id=turn_id,
        correlation_id=correlation_id,
    )

    return [root_event, atom_event]
