from __future__ import annotations

import re
from datetime import datetime, timezone

from orion.schemas.execution_projection import ExecutionRunStateV1
from orion.schemas.grammar import GrammarEventV1

from .constants import EXECUTION_SOURCE_SERVICES
from .ids import parse_execution_trace_id

_KV_RE = re.compile(r"(\w+)=([^,;\s]+)")


def _utc_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    return now if now.tzinfo else now.replace(tzinfo=timezone.utc)


def _parse_summary_kv(summary: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, val in _KV_RE.findall(summary or ""):
        out[key.lower()] = val.strip()
    return out


def _boolish(val: str | None) -> bool:
    return str(val or "").strip().lower() in {"true", "1", "yes", "on"}


def compute_pressure_hints(
    run: ExecutionRunStateV1,
    *,
    egress_emitted: bool,
) -> dict[str, float]:
    started = max(0, run.started_step_count)
    failed = max(0, run.failed_step_count)
    execution_load = min(1.0, started / 8.0)
    execution_friction = min(1.0, failed / max(1, started))
    reasoning_load = 0.35 if run.reasoning_present else 0.05
    status_fail = run.status.lower() in {"fail", "partial", "failed", "error"}
    failure_pressure = 1.0 if status_fail or failed > 0 else 0.0
    egress_confidence = 1.0 if egress_emitted else 0.25
    return {
        "execution_load": execution_load,
        "execution_friction": execution_friction,
        "reasoning_load": reasoning_load,
        "failure_pressure": failure_pressure,
        "egress_confidence": egress_confidence,
    }


def extract_execution_state_from_events(
    events: list[GrammarEventV1],
    *,
    now: datetime | None = None,
) -> ExecutionRunStateV1:
    clock = _utc_now(now)
    if not events:
        raise ValueError("events must not be empty")

    trace_id = events[0].trace_id
    parsed = parse_execution_trace_id(trace_id or "")
    node_id = parsed[0] if parsed else "unknown"
    correlation_id = parsed[1] if parsed else (events[0].correlation_id or "unknown")

    run = ExecutionRunStateV1(
        trace_id=trace_id or "",
        correlation_id=correlation_id,
        session_id=events[0].session_id,
        turn_id=events[0].turn_id,
        node_id=node_id,
        last_updated_at=clock,
    )

    egress_emitted = False
    for event in events:
        if event.provenance.source_service not in EXECUTION_SOURCE_SERVICES:
            continue
        atom = event.atom
        if not atom:
            continue
        role = atom.semantic_role or ""
        if role == "harness_fcc_step":
            continue
        kv = _parse_summary_kv(atom.summary or "")
        run.evidence_event_ids.append(event.event_id)

        if role == "exec_request_received":
            run.verb = kv.get("verb", run.verb)
            run.mode = kv.get("mode", run.mode)
        elif role == "exec_plan_started":
            try:
                run.step_count = int(kv.get("step_count", run.step_count) or run.step_count)
            except ValueError:
                pass
        elif role == "exec_recall_gate_observed":
            run.recall_observed = True
        elif role == "exec_step_started":
            run.started_step_count += 1
        elif role == "exec_step_completed":
            run.completed_step_count += 1
        elif role == "exec_step_failed":
            run.failed_step_count += 1
        elif role == "exec_result_assembled":
            run.status = kv.get("status", run.status)
            run.final_text_present = _boolish(kv.get("final_text_present"))
            run.reasoning_present = _boolish(kv.get("reasoning_present"))
            run.thinking_source = kv.get("thinking_source", run.thinking_source)
        elif role == "exec_result_emitted":
            egress_emitted = True

    run.pressure_hints = compute_pressure_hints(run, egress_emitted=egress_emitted)
    return run
