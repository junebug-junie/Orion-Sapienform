from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from orion.schemas.cortex.schemas import PlanExecutionRequest
from orion.schemas.grammar import (
    GrammarAtomV1,
    GrammarEdgeV1,
    GrammarEventV1,
    GrammarProvenanceV1,
)
from orion.substrate.execution_loop.ids import cortex_exec_trace_id


def short_error_kind(error: str | None) -> str:
    if not error:
        return "unknown"
    token = re.split(r"[:|\s]+", str(error).strip(), maxsplit=1)[0]
    return (token or "unknown")[:64]


def _hash_id(*parts: object, prefix: str) -> str:
    raw = "|".join(str(p) for p in parts)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


CORTEX_EXEC_ISOLATED_TRACE_LANES = frozenset({
    "harness_finalize_reflect",
    "orion_voice_finalize",
})


def trace_lane_for_verb(verb_name: str | None) -> str | None:
    verb = (verb_name or "").strip()
    if verb in CORTEX_EXEC_ISOLATED_TRACE_LANES:
        return verb
    return None


@dataclass
class CortexExecGrammarCollector:
    node_name: str
    correlation_id: str
    code_version: str | None
    observed_at: datetime
    session_id: str | None = None
    turn_id: str | None = None
    trace_lane: str | None = None
    _atoms: dict[str, GrammarAtomV1] = field(default_factory=dict)
    _edge_specs: list[tuple[str, str, str]] = field(default_factory=list)
    _last_completed_atom_id: str | None = None

    @property
    def trace_id(self) -> str:
        return cortex_exec_trace_id(self.node_name, self.correlation_id, lane=self.trace_lane)

    def _provenance(self, payload_ref: str) -> GrammarProvenanceV1:
        return GrammarProvenanceV1(
            source_service="orion-cortex-exec",
            source_component="cortex_exec_grammar_emit",
            source_event_id=f"{self.correlation_id}:{payload_ref}",
            source_trace_id=self.trace_id,
            source_payload_ref=payload_ref,
            code_version=self.code_version,
        )

    def _atom_id(self, key: str) -> str:
        return f"{self.trace_id}:{key}"

    def _put_atom(self, key: str, atom: GrammarAtomV1) -> None:
        self._atoms[key] = atom

    def record_request_received(self, *, req: PlanExecutionRequest, mode: str) -> None:
        verb = req.plan.verb_name or "unknown"
        n = len(req.plan.steps or [])
        ref = f"cortex.exec.request:{self.correlation_id}"
        self._put_atom(
            "exec_request_received",
            GrammarAtomV1(
                atom_id=self._atom_id("exec_request_received"),
                trace_id=self.trace_id,
                atom_type="observation",
                semantic_role="exec_request_received",
                layer="intake",
                dimensions=["execution", "request", "cortex"],
                summary=f"Cortex exec received plan request for verb={verb}, mode={mode}, steps={n}",
                confidence=1.0,
                salience=1.0,
                source_event_id=self.correlation_id,
                payload_ref=ref,
            ),
        )

    def record_validation_failed(self, *, error_kind: str = "validation_failed") -> None:
        ref = f"cortex.exec.request_invalid:{self.correlation_id}"
        self._put_atom(
            "exec_request_invalid",
            GrammarAtomV1(
                atom_id=self._atom_id("exec_request_invalid"),
                trace_id=self.trace_id,
                atom_type="uncertainty_marker",
                semantic_role="exec_request_invalid",
                layer="intake",
                dimensions=["execution", "request", "validation"],
                summary=f"Cortex exec request validation failed: error_kind={error_kind}",
                confidence=1.0,
                salience=0.9,
                source_event_id=self.correlation_id,
                payload_ref=ref,
            ),
        )

    def record_plan_started(
        self, *, req: PlanExecutionRequest, depth: int | None, step_count: int
    ) -> None:
        verb = req.plan.verb_name or "unknown"
        ref = f"cortex.exec.plan:{self.correlation_id}"
        self._put_atom(
            "exec_plan_started",
            GrammarAtomV1(
                atom_id=self._atom_id("exec_plan_started"),
                trace_id=self.trace_id,
                atom_type="action_candidate",
                semantic_role="exec_plan_started",
                layer="plan",
                dimensions=["execution", "plan", "agency"],
                summary=(
                    f"Execution plan started for verb={verb}; "
                    f"step_count={step_count}; depth={depth if depth is not None else 'none'}"
                ),
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
        ref = f"cortex.exec.recall_gate:{self.correlation_id}"
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

    def record_step_started(
        self,
        *,
        order: int,
        step_name: str,
        verb_name: str,
        services: list[str],
    ) -> None:
        key = f"exec_step_started:{order}:{step_name}"
        ref = f"cortex.exec.step:{self.correlation_id}:{order}:{step_name}"
        svc_list = ",".join(services) if services else "none"
        atom = GrammarAtomV1(
            atom_id=self._atom_id(key),
            trace_id=self.trace_id,
            atom_type="action_candidate",
            semantic_role="exec_step_started",
            layer="step",
            dimensions=["execution", "step", "service"],
            summary=(
                f"Step started: order={order}, step={step_name}, "
                f"verb={verb_name}, services={svc_list}"
            ),
            confidence=1.0,
            salience=0.85,
            source_event_id=f"{self.correlation_id}:{order}",
            payload_ref=ref,
        )
        self._atoms[key] = atom
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

    def record_step_completed(
        self,
        *,
        order: int,
        step_name: str,
        latency_ms: int | None,
        result_service_keys: list[str],
    ) -> None:
        started_key = f"exec_step_started:{order}:{step_name}"
        key = f"exec_step_completed:{order}:{step_name}"
        ref = f"cortex.exec.step_result:{self.correlation_id}:{order}:{step_name}"
        keys = ",".join(sorted(result_service_keys)) if result_service_keys else "none"
        atom = GrammarAtomV1(
            atom_id=self._atom_id(key),
            trace_id=self.trace_id,
            atom_type="reasoning_step",
            semantic_role="exec_step_completed",
            layer="step",
            dimensions=["execution", "step", "result"],
            summary=(
                f"Step completed: step={step_name}, status=success, "
                f"latency_ms={latency_ms or 0}, result_services={keys}"
            ),
            confidence=0.95,
            salience=0.8,
            source_event_id=f"{self.correlation_id}:{order}",
            payload_ref=ref,
        )
        self._atoms[key] = atom
        started = self._atoms.get(started_key)
        if started:
            self._edge_specs.append((started.atom_id, atom.atom_id, "derived_from"))
        self._last_completed_atom_id = atom.atom_id

    def record_step_failed(self, *, order: int, step_name: str, error_kind: str) -> None:
        started_key = f"exec_step_started:{order}:{step_name}"
        key = f"exec_step_failed:{order}:{step_name}"
        ref = f"cortex.exec.step_result:{self.correlation_id}:{order}:{step_name}"
        atom = GrammarAtomV1(
            atom_id=self._atom_id(key),
            trace_id=self.trace_id,
            atom_type="uncertainty_marker",
            semantic_role="exec_step_failed",
            layer="step",
            dimensions=["execution", "failure", "step"],
            summary=f"Step failed: step={step_name}, error_kind={error_kind}",
            confidence=0.9,
            salience=0.9,
            source_event_id=f"{self.correlation_id}:{order}",
            payload_ref=ref,
        )
        self._atoms[key] = atom
        started = self._atoms.get(started_key)
        if started:
            self._edge_specs.append((started.atom_id, atom.atom_id, "derived_from"))
        self._last_completed_atom_id = atom.atom_id

    def record_result_assembled(
        self,
        *,
        status: str,
        final_text_present: bool,
        reasoning_present: bool,
        thinking_source: str,
    ) -> None:
        ref = f"cortex.exec.result:{self.correlation_id}"
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
        ref = f"cortex.exec.egress:{self.correlation_id}"
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
                    f"Cortex exec result emitted to reply_to={reply_present}, status={status}"
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


def get_or_create_collector(
    ctx: dict[str, Any],
    *,
    correlation_id: str,
    node_name: str,
    code_version: str | None,
    trace_lane: str | None = None,
) -> CortexExecGrammarCollector:
    collector = ctx.get("_cortex_exec_grammar_collector")
    if collector is None:
        collector = new_cortex_exec_collector(
            correlation_id=correlation_id,
            ctx=ctx,
            code_version=code_version,
            node_name=node_name,
            trace_lane=trace_lane,
        )
        ctx["_cortex_exec_grammar_collector"] = collector
    return collector


def begin_plan_grammar(
    ctx: dict[str, Any],
    *,
    correlation_id: str,
    req: PlanExecutionRequest,
    mode: str,
    depth: int | None,
    run_recall: bool,
    recall_profile: str | None,
    recall_reason: str,
    node_name: str,
    code_version: str | None,
) -> CortexExecGrammarCollector:
    lane = trace_lane_for_verb(req.plan.verb_name)
    collector = get_or_create_collector(
        ctx,
        correlation_id=correlation_id,
        node_name=node_name,
        code_version=code_version,
        trace_lane=lane,
    )
    if not ctx.get("_cortex_exec_grammar_request_recorded"):
        collector.record_request_received(req=req, mode=mode)
        ctx["_cortex_exec_grammar_request_recorded"] = True
    collector.record_plan_started(req=req, depth=depth, step_count=len(req.plan.steps or []))
    collector.record_recall_gate_observed(
        run_recall=run_recall,
        profile=recall_profile,
        reason=recall_reason,
    )
    return collector


def record_assembled_grammar(
    ctx: dict[str, Any],
    *,
    status: str,
    final_text_present: bool,
    reasoning_present: bool,
    thinking_source: str,
) -> None:
    collector = ctx.get("_cortex_exec_grammar_collector")
    if collector is None:
        return
    collector.record_result_assembled(
        status=status,
        final_text_present=final_text_present,
        reasoning_present=reasoning_present,
        thinking_source=thinking_source,
    )


def new_cortex_exec_collector(
    *,
    correlation_id: str,
    ctx: dict[str, Any],
    code_version: str | None,
    node_name: str,
    trace_lane: str | None = None,
) -> CortexExecGrammarCollector:
    session_id = str(ctx.get("session_id") or ctx.get("sessionId") or "") or None
    turn_id = str(ctx.get("turn_id") or ctx.get("message_id") or ctx.get("messageId") or "") or None
    return CortexExecGrammarCollector(
        node_name=node_name,
        correlation_id=correlation_id,
        code_version=code_version,
        observed_at=datetime.now(timezone.utc),
        session_id=session_id,
        turn_id=turn_id,
        trace_lane=trace_lane,
    )


def build_cortex_exec_grammar_events(
    collector: CortexExecGrammarCollector,
) -> list[GrammarEventV1]:
    observed_at = collector.observed_at
    emitted_at = datetime.now(timezone.utc)
    trace_id = collector.trace_id
    provenance = collector._provenance(f"cortex.exec.trace:{collector.correlation_id}")

    root = _event(
        event_kind="trace_started",
        trace_id=trace_id,
        emitted_at=emitted_at,
        observed_at=observed_at,
        provenance=provenance,
        layer="execution",
        dimensions=["execution", "cortex", "plan"],
        session_id=collector.session_id,
        turn_id=collector.turn_id,
        correlation_id=collector.correlation_id,
    )
    root_id = root.event_id
    events: list[GrammarEventV1] = [root]

    for atom in collector._atoms.values():
        events.append(
            _event(
                event_kind="atom_emitted",
                trace_id=trace_id,
                emitted_at=emitted_at,
                observed_at=observed_at,
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
        events.append(
            _event(
                event_kind="edge_emitted",
                trace_id=trace_id,
                emitted_at=emitted_at,
                observed_at=observed_at,
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

    events.append(
        _event(
            event_kind="trace_ended",
            trace_id=trace_id,
            emitted_at=datetime.now(timezone.utc),
            observed_at=observed_at,
            provenance=provenance,
            parent_event_id=root_id,
            root_event_id=root_id,
            layer="execution",
            dimensions=["execution", "cortex", "plan"],
            session_id=collector.session_id,
            turn_id=collector.turn_id,
            correlation_id=collector.correlation_id,
        )
    )
    return events
