from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable
from uuid import uuid4

from orion.schemas.grammar import (
    GrammarAtomV1,
    GrammarEdgeV1,
    GrammarEventV1,
    GrammarProvenanceV1,
    TimeRangeV1,
)
from orion.substrate.execution_loop.ids import cortex_exec_trace_id

PublishFn = Callable[..., Awaitable[None]]

logger = logging.getLogger("orion.harness.grammar_emit")


def short_error_kind(error: str | None) -> str:
    if not error:
        return "unknown"
    token = re.split(r"[:|\s]+", str(error).strip(), maxsplit=1)[0]
    return (token or "unknown")[:64]


def compute_harness_reasoning_present(
    *,
    step_count: int,
    reflection_ran: bool,
    quick_lane_skipped_5b: bool,
    grammar_receipt_count: int,
) -> bool:
    if step_count > 0:
        return True
    if reflection_ran and not quick_lane_skipped_5b:
        return True
    return grammar_receipt_count > 0


def compute_harness_thinking_source(
    *,
    step_count: int,
    reflection_ran: bool,
    quick_lane_skipped_5b: bool,
) -> str:
    if step_count > 0:
        return "harness_fcc"
    if reflection_ran and not quick_lane_skipped_5b:
        return "finalize_reflect"
    return "none"


def _hash_id(*parts: object, prefix: str) -> str:
    raw = "|".join(str(p) for p in parts)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


@dataclass
class HarnessGrammarCollector:
    node_name: str
    correlation_id: str
    observed_at: datetime
    code_version: str | None = None
    session_id: str | None = None
    turn_id: str | None = None
    _atoms: dict[str, GrammarAtomV1] = field(default_factory=dict)
    # Real wall-clock moment each atom was actually recorded, keyed by
    # atom_id. Populated by _put_atom() -- same fix shape as
    # CortexExecGrammarCollector's _atom_observed_at
    # (services/orion-cortex-exec/app/grammar_emit.py, PR #1039/spec
    # 2026-07-14-cortex-exec-grammar-atom-wall-clock-spec.md). Before this,
    # every GrammarEventV1's observed_at in a harness trace was the same
    # value captured once at collector construction (self.observed_at,
    # trace-START -- not "flush time", that's the separate emitted_at set
    # in build_harness_grammar_events()). Confirmed live: harness traces
    # with 7-55 atoms and up to 2m15s of real wall-clock span between their
    # sibling (bare-correlation-id) trace's created_at rows all showed
    # exactly 1 distinct observed_at value across every atom.
    _atom_observed_at: dict[str, datetime] = field(default_factory=dict)
    _edge_specs: list[tuple[str, str, str]] = field(default_factory=list)
    _last_completed_atom_id: str | None = None

    @property
    def trace_id(self) -> str:
        return cortex_exec_trace_id(self.node_name, self.correlation_id)

    def _provenance(self, payload_ref: str) -> GrammarProvenanceV1:
        return GrammarProvenanceV1(
            source_service="orion-harness-governor",
            source_component="harness_grammar_emit",
            source_event_id=f"{self.correlation_id}:{payload_ref}",
            source_trace_id=self.trace_id,
            source_payload_ref=payload_ref,
            code_version=self.code_version,
        )

    def _atom_id(self, key: str) -> str:
        return f"{self.trace_id}:{key}"

    def _put_atom(self, key: str, atom: GrammarAtomV1) -> None:
        now = datetime.now(timezone.utc)
        # Same source timestamp for both, set together so they can never
        # drift -- see CortexExecGrammarCollector._put_atom's identical
        # pattern for why time_range is stamped here too (ledger.py's
        # atom-row persistence and the Grammar Atlas API read it).
        atom.time_range = TimeRangeV1(start=now, end=now)
        self._atoms[key] = atom
        self._atom_observed_at[atom.atom_id] = now

    def record_request_received(self) -> None:
        ref = f"harness.exec.request:{self.correlation_id}"
        self._put_atom(
            "exec_request_received",
            GrammarAtomV1(
                atom_id=self._atom_id("exec_request_received"),
                trace_id=self.trace_id,
                atom_type="observation",
                semantic_role="exec_request_received",
                layer="intake",
                dimensions=["execution", "request", "harness"],
                summary="Harness exec received plan request for verb=orion_unified, mode=orion, steps=0",
                confidence=1.0,
                salience=1.0,
                source_event_id=self.correlation_id,
                payload_ref=ref,
            ),
        )

    def record_plan_started(self, *, step_count: int) -> None:
        ref = f"harness.exec.plan:{self.correlation_id}"
        self._put_atom(
            "exec_plan_started",
            GrammarAtomV1(
                atom_id=self._atom_id("exec_plan_started"),
                trace_id=self.trace_id,
                atom_type="action_candidate",
                semantic_role="exec_plan_started",
                layer="plan",
                dimensions=["execution", "plan", "agency"],
                summary=f"Execution plan started for verb=orion_unified; step_count={step_count}",
                confidence=1.0,
                salience=0.9,
                source_event_id=self.correlation_id,
                payload_ref=ref,
            ),
        )
        if "exec_request_received" in self._atoms:
            self._edge_specs.append(
                (
                    self._atoms["exec_request_received"].atom_id,
                    self._atoms["exec_plan_started"].atom_id,
                    "contains",
                )
            )

    def record_recall_gate_observed(
        self, *, run_recall: bool, profile: str | None, reason: str
    ) -> None:
        ref = f"harness.exec.recall_gate:{self.correlation_id}"
        self._put_atom(
            "exec_recall_gate_observed",
            GrammarAtomV1(
                atom_id=self._atom_id("exec_recall_gate_observed"),
                trace_id=self.trace_id,
                atom_type="signal",
                semantic_role="exec_recall_gate_observed",
                layer="memory_gate",
                dimensions=["execution", "recall", "memory"],
                summary=(
                    f"Recall policy resolved: run={run_recall}, "
                    f"profile={profile or 'none'}, reason={reason}"
                ),
                confidence=0.95,
                salience=0.7,
                source_event_id=self.correlation_id,
                payload_ref=ref,
            ),
        )
        if "exec_plan_started" in self._atoms:
            self._edge_specs.append(
                (
                    self._atoms["exec_plan_started"].atom_id,
                    self._atoms["exec_recall_gate_observed"].atom_id,
                    "contains",
                )
            )

    def record_step_started(self, *, order: int, summary: str) -> None:
        key = f"exec_step_started:{order}"
        ref = f"harness.exec.step:{self.correlation_id}:{order}"
        clipped = summary[:500]
        atom = GrammarAtomV1(
            atom_id=self._atom_id(key),
            trace_id=self.trace_id,
            atom_type="action_candidate",
            semantic_role="exec_step_started",
            layer="step",
            dimensions=["execution", "step", "harness"],
            summary=f"Step started: order={order}, summary={clipped}",
            confidence=1.0,
            salience=0.85,
            source_event_id=f"{self.correlation_id}:{order}",
            payload_ref=ref,
        )
        self._put_atom(key, atom)
        if self._last_completed_atom_id:
            self._edge_specs.append(
                (self._last_completed_atom_id, atom.atom_id, "temporal_successor")
            )
        elif "exec_recall_gate_observed" in self._atoms:
            self._edge_specs.append(
                (
                    self._atoms["exec_recall_gate_observed"].atom_id,
                    atom.atom_id,
                    "contains",
                )
            )
        elif "exec_plan_started" in self._atoms:
            self._edge_specs.append(
                (
                    self._atoms["exec_plan_started"].atom_id,
                    atom.atom_id,
                    "contains",
                )
            )

    def record_step_completed(self, *, order: int) -> None:
        started_key = f"exec_step_started:{order}"
        key = f"exec_step_completed:{order}"
        ref = f"harness.exec.step_result:{self.correlation_id}:{order}"
        atom = GrammarAtomV1(
            atom_id=self._atom_id(key),
            trace_id=self.trace_id,
            atom_type="reasoning_step",
            semantic_role="exec_step_completed",
            layer="step",
            dimensions=["execution", "step", "result"],
            summary=f"Step completed: order={order}, status=success",
            confidence=0.95,
            salience=0.8,
            source_event_id=f"{self.correlation_id}:{order}",
            payload_ref=ref,
        )
        self._put_atom(key, atom)
        started = self._atoms.get(started_key)
        if started:
            self._edge_specs.append((started.atom_id, atom.atom_id, "derived_from"))
        self._last_completed_atom_id = atom.atom_id

    def record_step_failed(self, *, order: int, error_kind: str) -> None:
        started_key = f"exec_step_started:{order}"
        key = f"exec_step_failed:{order}"
        ref = f"harness.exec.step_result:{self.correlation_id}:{order}"
        atom = GrammarAtomV1(
            atom_id=self._atom_id(key),
            trace_id=self.trace_id,
            atom_type="uncertainty_marker",
            semantic_role="exec_step_failed",
            layer="step",
            dimensions=["execution", "failure", "step"],
            summary=f"Step failed: order={order}, error_kind={error_kind}",
            confidence=0.9,
            salience=0.9,
            source_event_id=f"{self.correlation_id}:{order}",
            payload_ref=ref,
        )
        self._put_atom(key, atom)
        started = self._atoms.get(started_key)
        if started:
            self._edge_specs.append((started.atom_id, atom.atom_id, "derived_from"))
        self._last_completed_atom_id = atom.atom_id

    def record_tool_provenance_mismatch(self, *, mismatch: str) -> None:
        """Draft used live-immediacy language while this turn's own tool
        trace shows a fetch-shaped call -- the confabulation pattern from
        project_orion_substrate_bridge_confabulation. uncertainty_marker is
        the right atom_type here (same as record_step_failed): not a hard
        motor failure, but a flag on the claim's own grounding."""
        ref = f"harness.exec.tool_provenance:{self.correlation_id}"
        self._put_atom(
            "exec_tool_provenance_mismatch",
            GrammarAtomV1(
                atom_id=self._atom_id("exec_tool_provenance_mismatch"),
                trace_id=self.trace_id,
                atom_type="uncertainty_marker",
                semantic_role="exec_tool_provenance_mismatch",
                layer="result",
                dimensions=["execution", "grounding", "tool_use"],
                summary=mismatch,
                confidence=0.7,
                salience=0.8,
                source_event_id=self.correlation_id,
                payload_ref=ref,
            ),
        )
        if self._last_completed_atom_id:
            self._edge_specs.append(
                (
                    self._last_completed_atom_id,
                    self._atoms["exec_tool_provenance_mismatch"].atom_id,
                    "derived_from",
                )
            )
        # Unlike record_step_failed, this atom was missing this update --
        # without it, record_result_assembled's own derived_from edge still
        # points at the last *step* atom instead of this one, leaving the
        # mismatch atom a graph leaf with no outgoing edge into the
        # result-assembled/emitted chain despite being about the result.
        self._last_completed_atom_id = self._atoms["exec_tool_provenance_mismatch"].atom_id

    def record_result_assembled(
        self,
        *,
        status: str,
        final_text_present: bool,
        step_count: int,
        grammar_receipt_count: int,
        reflection_ran: bool,
        quick_lane_skipped_5b: bool,
    ) -> None:
        reasoning_present = compute_harness_reasoning_present(
            step_count=step_count,
            reflection_ran=reflection_ran,
            quick_lane_skipped_5b=quick_lane_skipped_5b,
            grammar_receipt_count=grammar_receipt_count,
        )
        thinking_source = compute_harness_thinking_source(
            step_count=step_count,
            reflection_ran=reflection_ran,
            quick_lane_skipped_5b=quick_lane_skipped_5b,
        )
        ref = f"harness.exec.result:{self.correlation_id}"
        self._put_atom(
            "exec_result_assembled",
            GrammarAtomV1(
                atom_id=self._atom_id("exec_result_assembled"),
                trace_id=self.trace_id,
                atom_type="spoken_output",
                semantic_role="exec_result_assembled",
                layer="result",
                dimensions=["execution", "speech", "reasoning"],
                summary=(
                    f"Final result assembled: status={status}, "
                    f"final_text_present={final_text_present}, "
                    f"reasoning_present={reasoning_present}, "
                    f"thinking_source={thinking_source}"
                ),
                confidence=0.95,
                salience=0.95,
                source_event_id=self.correlation_id,
                payload_ref=ref,
            ),
        )
        if self._last_completed_atom_id:
            self._edge_specs.append(
                (
                    self._last_completed_atom_id,
                    self._atoms["exec_result_assembled"].atom_id,
                    "derived_from",
                )
            )

    def record_result_emitted(self, *, reply_present: bool, status: str) -> None:
        ref = f"harness.exec.egress:{self.correlation_id}"
        self._put_atom(
            "exec_result_emitted",
            GrammarAtomV1(
                atom_id=self._atom_id("exec_result_emitted"),
                trace_id=self.trace_id,
                atom_type="signal",
                semantic_role="exec_result_emitted",
                layer="egress",
                dimensions=["execution", "result", "bus"],
                summary=(
                    f"Harness exec result emitted to reply_to={reply_present}, status={status}"
                ),
                confidence=1.0,
                salience=0.8,
                source_event_id=self.correlation_id,
                payload_ref=ref,
            ),
        )
        if "exec_result_assembled" in self._atoms:
            self._edge_specs.append(
                (
                    self._atoms["exec_result_assembled"].atom_id,
                    self._atoms["exec_result_emitted"].atom_id,
                    "rendered_as",
                )
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
    session_id: str | None = None,
    turn_id: str | None = None,
    correlation_id: str | None = None,
) -> GrammarEventV1:
    body_key = atom.atom_id if atom else edge.edge_id if edge else uuid4().hex
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
        edge=edge,
        provenance=provenance,
    )


def build_harness_grammar_events(
    collector: HarnessGrammarCollector,
) -> list[GrammarEventV1]:
    observed_at = collector.observed_at
    emitted_at = datetime.now(timezone.utc)
    trace_id = collector.trace_id
    provenance = collector._provenance(f"harness.exec.trace:{collector.correlation_id}")

    root = _event(
        event_kind="trace_started",
        trace_id=trace_id,
        emitted_at=emitted_at,
        observed_at=observed_at,
        provenance=provenance,
        layer="execution",
        dimensions=["execution", "harness", "plan"],
        session_id=collector.session_id,
        turn_id=collector.turn_id,
        correlation_id=collector.correlation_id,
    )
    root_id = root.event_id
    events: list[GrammarEventV1] = [root]

    for atom in collector._atoms.values():
        # Real wall-clock moment this atom was actually recorded (see
        # _put_atom / _atom_observed_at docstring above), not the single
        # trace-START value every atom in a trace previously shared. Falls
        # back to trace-start observed_at only if an atom somehow bypassed
        # _put_atom(). emitted_at stays the shared flush-time value.
        atom_ts = collector._atom_observed_at.get(atom.atom_id, observed_at)
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
                session_id=collector.session_id,
                turn_id=collector.turn_id,
                correlation_id=collector.correlation_id,
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
            salience=0.7,
            evidence_event_ids=[collector.correlation_id],
        )
        # observed_at = real moment the target atom happened; emitted_at
        # stays the shared flush-time value, same reasoning as above.
        edge_ts = collector._atom_observed_at.get(to_atom_id, observed_at)
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
                layer="step",
                dimensions=["execution", "plan", "step"],
                session_id=collector.session_id,
                turn_id=collector.turn_id,
                correlation_id=collector.correlation_id,
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
            layer="execution",
            dimensions=["execution", "harness", "plan"],
            session_id=collector.session_id,
            turn_id=collector.turn_id,
            correlation_id=collector.correlation_id,
        )
    )
    return events


def build_harness_grammar_finalize_events(
    collector: HarnessGrammarCollector,
) -> list[GrammarEventV1]:
    """Emit only post-motor finalize atoms (refreshed assembled + egress).

    Motor publish already sent request/plan/step lifecycle; finalize must not
    replay the full trace (duplicate trace_started / step atoms on the bus).
    """
    roles = ("exec_result_assembled", "exec_result_emitted")
    atoms = [collector._atoms[r] for r in roles if r in collector._atoms]
    if not atoms:
        return []

    observed_at = collector.observed_at
    emitted_at = datetime.now(timezone.utc)
    trace_id = collector.trace_id
    provenance = collector._provenance(f"harness.exec.finalize:{collector.correlation_id}")
    events: list[GrammarEventV1] = []
    for atom in atoms:
        atom_ts = collector._atom_observed_at.get(atom.atom_id, observed_at)
        events.append(
            _event(
                event_kind="atom_emitted",
                trace_id=trace_id,
                emitted_at=emitted_at,
                observed_at=atom_ts,
                provenance=provenance,
                atom=atom,
                layer=atom.layer,
                dimensions=atom.dimensions,
                session_id=collector.session_id,
                turn_id=collector.turn_id,
                correlation_id=collector.correlation_id,
            )
        )

    assembled = collector._atoms.get("exec_result_assembled")
    emitted = collector._atoms.get("exec_result_emitted")
    if assembled and emitted:
        edge = GrammarEdgeV1(
            edge_id=f"{trace_id}:edge:{assembled.atom_id}:rendered_as:{emitted.atom_id}",
            trace_id=trace_id,
            from_atom_id=assembled.atom_id,
            to_atom_id=emitted.atom_id,
            relation_type="rendered_as",  # type: ignore[arg-type]
            confidence=0.9,
            salience=0.7,
            evidence_event_ids=[collector.correlation_id],
        )
        edge_ts = collector._atom_observed_at.get(emitted.atom_id, observed_at)
        events.append(
            _event(
                event_kind="edge_emitted",
                trace_id=trace_id,
                emitted_at=emitted_at,
                observed_at=edge_ts,
                provenance=provenance,
                edge=edge,
                layer="step",
                dimensions=["execution", "plan", "step"],
                session_id=collector.session_id,
                turn_id=collector.turn_id,
                correlation_id=collector.correlation_id,
            )
        )
    return events


async def publish_harness_lifecycle_grammar(
    bus: Any,
    *,
    channel: str,
    events: list[GrammarEventV1],
    publish_fn: PublishFn | None = None,
) -> None:
    corr = events[0].correlation_id if events else "unknown"
    if not events:
        logger.info(
            "harness_lifecycle_grammar_publish_skipped corr=%s channel=%s reason=empty_events",
            corr,
            channel,
        )
        return
    roles = sorted({e.atom.semantic_role for e in events if e.atom})
    published = 0
    for event in events:
        try:
            if publish_fn is not None:
                await publish_fn(bus, event=event, channel=channel)
            else:
                from orion.grammar.publish import publish_grammar_event

                await publish_grammar_event(
                    bus,
                    event,
                    source_name="orion-harness-governor",
                    channel=channel,
                )
            published += 1
        except Exception:
            logger.warning(
                "harness_lifecycle_grammar_publish_failed corr=%s event_kind=%s",
                event.correlation_id or corr,
                event.event_kind,
                exc_info=True,
            )
    logger.info(
        "harness_lifecycle_grammar_published corr=%s channel=%s trace_id=%s events=%s roles=%s",
        corr,
        channel,
        events[0].trace_id,
        published,
        roles,
    )
