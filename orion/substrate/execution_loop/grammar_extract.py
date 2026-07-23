from __future__ import annotations

import math
import re
from datetime import datetime, timezone

from orion.schemas.execution_projection import ExecutionRunStateV1
from orion.schemas.grammar import GrammarEventV1

from .constants import EXECUTION_SOURCE_SERVICES
from .ids import parse_execution_trace_id

_KV_RE = re.compile(r"(\w+)=([^,;\s]+)")

# Mirrors merge.py's _COMPLIANCE_RANK (same ordinal meaning: never-downgrade rank for
# HarnessRunV1.compliance_verdict). Kept as a local duplicate rather than a shared import
# since merge.py already imports compute_pressure_hints from this module -- importing the
# rank table back from merge.py would be circular for 4 dict entries.
_COMPLIANCE_DEFICIT_RANK = {
    "unknown": 0,
    "completed": 0,
    "partial": 1,
    "failed": 2,
    "refused": 3,
}

# Reference denominator for harness_step_load's log-ratio, not a hard cap. Chosen well
# above execution_load's own hard cap (8) so an 8-step run reads meaningfully below
# saturation (~0.53) instead of identical to a 40-step run (~0.90) -- unlike
# execution_load's min(1.0, n/8.0). A starting anchor, not yet calibrated against live
# step-count distributions -- see spec doc's Missing Questions.
_HARNESS_STEP_LOAD_SATURATION_STEPS = 60


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
    # FCC-motor-only step load, split from the blended `execution_load` above (which
    # counts cortex-exec + harness-governor steps together and hard-caps at 8).
    # log1p-then-ratio, NOT bare log1p: every NODE_CHANNELS value gets hard-clamped to
    # [0,1] at write time by apply_perturbations() (mode="replace" ->
    # max(0.0, min(1.0, intensity)), services/orion-field-digester/app/digestion/
    # perturbation.py) -- bare log1p(n) exceeds 1.0 at n=2 (log1p(2)=1.099), saturating
    # this channel for virtually every real run and defeating its own purpose. Found in
    # code review, reproduced live against apply_perturbations() before this fix. The
    # ratio form stays genuinely continuous across the range that matters instead of
    # saturating almost immediately -- see
    # docs/superpowers/specs/2026-07-23-fcc-motor-field-digester-signals-design.md.
    harness_step_load = min(
        1.0,
        math.log1p(max(0, run.harness_started_step_count))
        / math.log1p(_HARNESS_STEP_LOAD_SATURATION_STEPS),
    )
    # 3 identical consecutive tool_result failures before saturating -- a starting
    # threshold, not yet calibrated against live short_error_kind() bucketing (see spec's
    # Missing Questions).
    tool_failure_streak_pressure = min(1.0, max(0, run.tool_failure_streak_max) / 3.0)
    avg_step_chars = run.step_char_sum / max(1, run.completed_step_count)
    # 4000 chars/step saturation is a starting anchor, not yet sampled against live data.
    avg_step_chars_pressure = min(1.0, avg_step_chars / 4000.0)
    compliance_deficit = (
        _COMPLIANCE_DEFICIT_RANK.get(run.compliance_verdict.strip().lower(), 0) / 3.0
    )
    # llm_serving_node is NOT injected here: pressure_hints must stay
    # float-only -- orion/substrate/relational/adapters/execution_ctx.py's
    # map_execution_ctx_to_substrate() does max(hints.values()) over this
    # exact dict, and a str value would raise TypeError there (caught, but
    # silently degrades that producer on every run). run.llm_serving_node is
    # already a first-class field on ExecutionRunStateV1 -- consumers that
    # need it (services/orion-field-digester/app/ingest/state_deltas.py)
    # should read it from there directly, not from pressure_hints.
    return {
        "execution_load": execution_load,
        "execution_friction": execution_friction,
        "reasoning_load": reasoning_load,
        "failure_pressure": failure_pressure,
        "egress_confidence": egress_confidence,
        "harness_step_load": harness_step_load,
        "tool_failure_streak_pressure": tool_failure_streak_pressure,
        "avg_step_chars_pressure": avg_step_chars_pressure,
        "compliance_deficit": compliance_deficit,
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
    # Prefer the event's own explicit correlation_id over the value parsed
    # out of trace_id. parse_execution_trace_id() only splits on the first
    # two colons (trace_id.split(":", 2)), so for any lane-suffixed trace_id
    # (e.g. HarnessGrammarCollector's "harness_motor" lane, or
    # CortexExecGrammarCollector's stance_react/harness_finalize_reflect/
    # orion_voice_finalize lanes) the parsed "correlation_id" slot actually
    # contains "{correlation_id}:{lane}" -- both real producers already
    # populate event.correlation_id cleanly (see
    # orion/harness/grammar_emit.py's _event()/build_harness_grammar_events()
    # and the cortex-exec sibling), so falling back to the parsed value only
    # when the clean field is genuinely absent avoids silently polluting
    # ExecutionRunStateV1.correlation_id -- a field whose name has an
    # established, unsuffixed meaning elsewhere in the system, and which is
    # exposed as-is on services/orion-substrate-runtime's
    # GET /projections/execution_trajectory debug endpoint. Found in review
    # of the trace_id lane fix (2026-07-15): before that fix,
    # HarnessGrammarCollector's trace_id was unlaned, so this precedence
    # inversion was latent; laning it made the pollution live for every
    # harness-governor execution run.
    correlation_id = events[0].correlation_id or (parsed[1] if parsed else "unknown")

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
            if event.provenance.source_service == "orion-harness-governor":
                run.harness_started_step_count += 1
        elif role == "exec_step_completed":
            run.completed_step_count += 1
        elif role == "exec_step_failed":
            run.failed_step_count += 1
        elif role == "exec_result_assembled":
            run.status = kv.get("status", run.status)
            run.final_text_present = _boolish(kv.get("final_text_present"))
            run.reasoning_present = _boolish(kv.get("reasoning_present"))
            run.thinking_source = kv.get("thinking_source", run.thinking_source)
            run.llm_serving_node = kv.get("llm_serving_node") or run.llm_serving_node
            run.compliance_verdict = kv.get("compliance_verdict", run.compliance_verdict)
            try:
                run.step_char_sum = int(
                    kv.get("step_char_sum", run.step_char_sum) or run.step_char_sum
                )
                run.step_char_max = int(
                    kv.get("step_char_max", run.step_char_max) or run.step_char_max
                )
                run.tool_failure_streak_max = int(
                    kv.get("tool_failure_streak_max", run.tool_failure_streak_max)
                    or run.tool_failure_streak_max
                )
            except ValueError:
                pass
        elif role == "exec_result_emitted":
            egress_emitted = True

    run.pressure_hints = compute_pressure_hints(run, egress_emitted=egress_emitted)
    return run
