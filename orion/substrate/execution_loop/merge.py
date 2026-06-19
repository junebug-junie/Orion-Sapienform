from __future__ import annotations

from copy import deepcopy

from orion.schemas.execution_projection import ExecutionRunStateV1

from .grammar_extract import compute_pressure_hints

_STATUS_RANK = {
    "unknown": 0,
    "success": 1,
    "partial": 2,
    "failed": 3,
    "fail": 3,
    "error": 4,
}


def _status_rank(status: str) -> int:
    return _STATUS_RANK.get((status or "unknown").strip().lower(), 0)


def _pick_status(existing: str, incoming: str) -> str:
    if _status_rank(incoming) > _status_rank(existing):
        return incoming
    return existing


def _pick_thinking_source(existing: str, incoming: str) -> str:
    ex = (existing or "none").strip().lower()
    inc = (incoming or "none").strip().lower()
    if ex != "none":
        return existing
    if inc != "none":
        return incoming
    return existing or "none"


def merge_execution_run_state(
    existing: ExecutionRunStateV1 | None,
    incoming: ExecutionRunStateV1,
) -> ExecutionRunStateV1:
    if existing is None:
        return incoming

    merged = deepcopy(existing)
    merged.started_step_count = max(existing.started_step_count, incoming.started_step_count)
    merged.completed_step_count = max(existing.completed_step_count, incoming.completed_step_count)
    merged.failed_step_count = max(existing.failed_step_count, incoming.failed_step_count)
    merged.step_count = max(existing.step_count, incoming.step_count)
    merged.recall_observed = existing.recall_observed or incoming.recall_observed
    merged.final_text_present = existing.final_text_present or incoming.final_text_present
    merged.reasoning_present = existing.reasoning_present or incoming.reasoning_present
    merged.thinking_source = _pick_thinking_source(existing.thinking_source, incoming.thinking_source)
    merged.status = _pick_status(existing.status, incoming.status)
    if incoming.verb != "unknown":
        merged.verb = incoming.verb
    if incoming.mode != "unknown":
        merged.mode = incoming.mode
    if incoming.session_id:
        merged.session_id = incoming.session_id
    if incoming.turn_id:
        merged.turn_id = incoming.turn_id

    merged.evidence_event_ids = sorted(
        set(existing.evidence_event_ids) | set(incoming.evidence_event_ids)
    )[-200:]
    merged.last_updated_at = incoming.last_updated_at

    egress_emitted = (
        existing.pressure_hints.get("egress_confidence", 0.0) >= 1.0
        or incoming.pressure_hints.get("egress_confidence", 0.0) >= 1.0
    )
    merged.pressure_hints = compute_pressure_hints(merged, egress_emitted=egress_emitted)
    return merged
