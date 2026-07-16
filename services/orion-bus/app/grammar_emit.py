from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from orion.schemas.grammar import (
    GrammarAtomV1,
    GrammarEdgeV1,
    GrammarEventV1,
    GrammarProvenanceV1,
    TimeRangeV1,
)


def bus_transport_trace_id(node_id: str, sample_window_id: str) -> str:
    return f"bus.transport:{node_id}:{sample_window_id}"


def _hash_id(*parts: object, prefix: str) -> str:
    raw = "|".join(str(p) for p in parts)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


@dataclass
class BusTransportGrammarCollector:
    node_id: str
    sample_window_id: str
    observed_at: datetime
    code_version: str | None = None
    _atoms: dict[str, GrammarAtomV1] = field(default_factory=dict)
    # Real wall-clock moment each atom was actually recorded, keyed by
    # atom_id. Populated by _put_atom() -- same fix shape as
    # CortexExecGrammarCollector/HarnessGrammarCollector's _atom_observed_at
    # (services/orion-cortex-exec/app/grammar_emit.py,
    # orion/harness/grammar_emit.py). Before this, every GrammarEventV1's
    # observed_at in a bus-transport trace was the same value captured once
    # at collector construction (self.observed_at, trace-START). Unlike the
    # other two siblings, no idempotency-per-atom_id guard is needed here:
    # a fresh BusTransportGrammarCollector is constructed per observer tick
    # (ObserverRollup.to_collector() in bus_observer.py, and the except-path
    # fail_collector in run_observer_tick()) and every record_*() method is
    # called at most once per instance -- confirmed by tracing the real call
    # sites, not assumed.
    _atom_observed_at: dict[str, datetime] = field(default_factory=dict)
    _edge_specs: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def trace_id(self) -> str:
        return bus_transport_trace_id(self.node_id, self.sample_window_id)

    def _provenance(self, payload_ref: str) -> GrammarProvenanceV1:
        return GrammarProvenanceV1(
            source_service="orion-bus",
            source_component="bus_transport_grammar_emit",
            source_event_id=f"{self.node_id}:{self.sample_window_id}",
            source_trace_id=self.trace_id,
            source_payload_ref=payload_ref,
            code_version=self.code_version,
        )

    def _atom_id(self, role: str) -> str:
        return f"{self.trace_id}:{role}"

    def _put_atom(self, role: str, atom: GrammarAtomV1) -> None:
        now = datetime.now(timezone.utc)
        # Same source timestamp for both, set together so they can never
        # drift -- see CortexExecGrammarCollector/HarnessGrammarCollector's
        # identical pattern for why time_range is stamped here too
        # (ledger.py's atom-row persistence and the Grammar Atlas API
        # read it).
        atom.time_range = TimeRangeV1(start=now, end=now)
        self._atoms[role] = atom
        self._atom_observed_at[atom.atom_id] = now

    def observed_at_for(self, atom_id: str) -> datetime:
        """Real recorded timestamp for a given atom_id, falling back to
        trace-start observed_at only if that atom somehow bypassed
        _put_atom(). Single source for the lookup build_bus_transport_grammar_events()
        needs per atom and per edge, so a future change to the fallback
        policy can't be applied to some call sites and missed at others."""
        return self._atom_observed_at.get(atom_id, self.observed_at)

    def record_tick_started(self) -> None:
        self._put_atom(
            "bus_observer_tick_started",
            GrammarAtomV1(
                atom_id=self._atom_id("bus_observer_tick_started"),
                trace_id=self.trace_id,
                atom_type="signal",
                semantic_role="bus_observer_tick_started",
                layer="transport",
                dimensions=["bus", "transport", "observer"],
                summary=(
                    f"Bus observer tick started node_id={self.node_id} "
                    f"sample_window_id={self.sample_window_id}"
                ),
                confidence=1.0,
                salience=0.5,
                source_event_id=self.sample_window_id,
                payload_ref=f"bus.transport.tick:{self.sample_window_id}",
            ),
        )

    def record_tick_completed(self, *, streams_observed: int) -> None:
        self._put_atom(
            "bus_observer_tick_completed",
            GrammarAtomV1(
                atom_id=self._atom_id("bus_observer_tick_completed"),
                trace_id=self.trace_id,
                atom_type="signal",
                semantic_role="bus_observer_tick_completed",
                layer="transport",
                dimensions=["bus", "transport", "observer"],
                summary=(
                    f"Bus observer tick completed streams_observed={streams_observed} "
                    f"sample_window_id={self.sample_window_id}"
                ),
                confidence=1.0,
                salience=0.5,
                source_event_id=self.sample_window_id,
                payload_ref=f"bus.transport.tick_done:{self.sample_window_id}",
            ),
        )
        if "bus_observer_tick_started" in self._atoms:
            self._edge_specs.append(
                (
                    self._atoms["bus_observer_tick_started"].atom_id,
                    self._atoms["bus_observer_tick_completed"].atom_id,
                    "temporal_successor",
                )
            )

    def record_tick_failed(self, *, error_kind: str) -> None:
        self._put_atom(
            "bus_observer_tick_failed",
            GrammarAtomV1(
                atom_id=self._atom_id("bus_observer_tick_failed"),
                trace_id=self.trace_id,
                atom_type="uncertainty_marker",
                semantic_role="bus_observer_tick_failed",
                layer="transport",
                dimensions=["bus", "transport", "failure"],
                summary=f"Bus observer tick failed error_kind={error_kind}",
                confidence=0.9,
                salience=0.9,
                source_event_id=self.sample_window_id,
                payload_ref=f"bus.transport.tick_failed:{self.sample_window_id}",
            ),
        )

    def record_health_observed(self, *, redis_ping_ok: bool) -> None:
        self._put_atom(
            "bus_health_observed",
            GrammarAtomV1(
                atom_id=self._atom_id("bus_health_observed"),
                trace_id=self.trace_id,
                atom_type="observation",
                semantic_role="bus_health_observed",
                layer="transport",
                dimensions=["bus", "health", "redis"],
                summary=(
                    f"Redis bus core health probe node_id={self.node_id} "
                    f"redis_ping_ok={str(redis_ping_ok).lower()} "
                    f"sample_window_id={self.sample_window_id}"
                ),
                confidence=1.0,
                salience=0.8,
                source_event_id=self.sample_window_id,
                payload_ref=f"bus.transport.health:{self.sample_window_id}",
            ),
        )
        if "bus_observer_tick_started" in self._atoms:
            self._edge_specs.append(
                (
                    self._atoms["bus_observer_tick_started"].atom_id,
                    self._atoms["bus_health_observed"].atom_id,
                    "contains",
                )
            )

    def record_stream_depth(self, *, stream_key: str, stream_length: int) -> None:
        role = f"bus_stream_depth_observed:{stream_key}"
        self._put_atom(
            role,
            GrammarAtomV1(
                atom_id=self._atom_id(role),
                trace_id=self.trace_id,
                atom_type="observation",
                semantic_role="bus_stream_depth_observed",
                layer="transport",
                dimensions=["bus", "stream", "depth"],
                summary=(
                    f"Observed Redis stream depth stream_key={stream_key} "
                    f"stream_length={stream_length} sample_window_id={self.sample_window_id}"
                ),
                confidence=1.0,
                salience=0.7,
                source_event_id=self.sample_window_id,
                payload_ref=f"bus.transport.depth:{stream_key}:{self.sample_window_id}",
            ),
        )
        if "bus_health_observed" in self._atoms:
            self._edge_specs.append(
                (
                    self._atoms["bus_health_observed"].atom_id,
                    self._atoms[role].atom_id,
                    "contains",
                )
            )

    def record_backpressure(
        self,
        *,
        stream_key: str,
        stream_length: int,
        threshold: int,
        severity: str,
    ) -> None:
        role = f"bus_backpressure_observed:{stream_key}"
        self._put_atom(
            role,
            GrammarAtomV1(
                atom_id=self._atom_id(role),
                trace_id=self.trace_id,
                atom_type="uncertainty_marker",
                semantic_role="bus_backpressure_observed",
                layer="transport",
                dimensions=["bus", "backpressure", "stream"],
                summary=(
                    f"Bus stream depth exceeded threshold stream_key={stream_key} "
                    f"stream_length={stream_length} threshold={threshold} severity={severity}"
                ),
                confidence=0.95,
                salience=0.85,
                source_event_id=self.sample_window_id,
                payload_ref=f"bus.transport.backpressure:{stream_key}:{self.sample_window_id}",
            ),
        )
        depth_role = f"bus_stream_depth_observed:{stream_key}"
        if depth_role in self._atoms:
            self._edge_specs.append(
                (
                    self._atoms[depth_role].atom_id,
                    self._atoms[role].atom_id,
                    "derived_from",
                )
            )

    def record_uncataloged_stream(self, *, stream_key: str) -> None:
        role = f"bus_configured_stream_uncataloged:{stream_key}"
        self._put_atom(
            role,
            GrammarAtomV1(
                atom_id=self._atom_id(role),
                trace_id=self.trace_id,
                atom_type="uncertainty_marker",
                semantic_role="bus_configured_stream_uncataloged",
                layer="transport",
                dimensions=["bus", "catalog", "contract"],
                summary=(
                    f"Configured observer stream is not declared in channel catalog "
                    f"stream_key={stream_key} source=bus_observer "
                    f"sample_window_id={self.sample_window_id}"
                ),
                confidence=0.9,
                salience=0.8,
                source_event_id=self.sample_window_id,
                payload_ref=f"bus.transport.uncataloged_stream:{stream_key}",
            ),
        )


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
) -> GrammarEventV1:
    body_key = atom.atom_id if atom else edge.edge_id if edge else uuid4().hex
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


def build_bus_transport_grammar_events(
    collector: BusTransportGrammarCollector,
) -> list[GrammarEventV1]:
    observed_at = collector.observed_at
    emitted_at = datetime.now(timezone.utc)
    trace_id = collector.trace_id
    provenance = collector._provenance(f"bus.transport.trace:{collector.sample_window_id}")

    root = _event(
        event_kind="trace_started",
        trace_id=trace_id,
        emitted_at=emitted_at,
        observed_at=observed_at,
        provenance=provenance,
        layer="transport",
        dimensions=["bus", "transport"],
    )
    root_id = root.event_id
    events: list[GrammarEventV1] = [root]

    for atom in collector._atoms.values():
        # Real wall-clock moment this atom was actually recorded (see
        # _put_atom / _atom_observed_at docstring above), not the single
        # trace-START value every atom in a trace previously shared. Falls
        # back to trace-start observed_at only if an atom somehow bypassed
        # _put_atom(). emitted_at stays the shared flush-time value.
        atom_ts = collector.observed_at_for(atom.atom_id)
        events.append(
            _event(
                event_kind="atom_emitted",
                trace_id=trace_id,
                emitted_at=emitted_at,
                observed_at=atom_ts,
                provenance=provenance,
                atom=atom,
                parent_event_id=root_id,
                root_event_id=root_id,
                layer=atom.layer,
                dimensions=atom.dimensions,
            )
        )

    for from_atom_id, to_atom_id, relation_type in collector._edge_specs:
        edge = GrammarEdgeV1(
            edge_id=f"{trace_id}:edge:{from_atom_id}:{relation_type}:{to_atom_id}",
            trace_id=trace_id,
            from_atom_id=from_atom_id,
            to_atom_id=to_atom_id,
            relation_type=relation_type,  # type: ignore[arg-type]
            confidence=0.9,
            salience=0.6,
            evidence_event_ids=[collector.sample_window_id],
        )
        # observed_at = real moment the target atom happened; emitted_at
        # stays the shared flush-time value, same reasoning as above.
        edge_ts = collector.observed_at_for(to_atom_id)
        events.append(
            _event(
                event_kind="edge_emitted",
                trace_id=trace_id,
                emitted_at=emitted_at,
                observed_at=edge_ts,
                provenance=provenance,
                edge=edge,
                parent_event_id=root_id,
                root_event_id=root_id,
                layer="transport",
                dimensions=["bus", "transport"],
            )
        )

    # Real trace-end moment = the last atom actually recorded, not
    # collector.observed_at (trace-START, which trace_ended previously
    # reused, collapsing trace_started/trace_ended's observed_at to the
    # same instant). Falls back to trace-start observed_at only for a
    # trace with zero recorded atoms.
    trace_ended_at = max(collector._atom_observed_at.values(), default=observed_at)
    events.append(
        _event(
            event_kind="trace_ended",
            trace_id=trace_id,
            emitted_at=datetime.now(timezone.utc),
            observed_at=trace_ended_at,
            provenance=provenance,
            parent_event_id=root_id,
            root_event_id=root_id,
            layer="transport",
            dimensions=["bus", "transport"],
        )
    )
    return events
