from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

from orion.schemas.grammar import GrammarEventV1
from orion.schemas.llm_inference_projection import LlmInferenceProjectionV1
from orion.schemas.reduction_receipt import ProjectionUpdateV1, ReductionReceiptV1
from orion.schemas.state_delta import StateDeltaV1
from orion.substrate.ids import stable_delta_id, stable_receipt_id

from .constants import (
    LLM_INFERENCE_PROJECTION_ID,
    LLM_INFERENCE_REDUCER_ID,
    LLM_INFERENCE_SOURCE_SERVICE,
    LLM_INFERENCE_TARGET_KIND,
)
from .extract import extract_llm_inference_states_from_events, parse_llm_inference_trace_id


def _utc_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    return now if now.tzinfo else now.replace(tzinfo=timezone.utc)


def _noop(reducer_id: str, events: list[GrammarEventV1], clock: datetime, warnings: list[str] | None = None) -> ReductionReceiptV1:
    noop_ids = [e.event_id for e in events]
    return ReductionReceiptV1(
        receipt_id=stable_receipt_id(
            reducer_id=reducer_id,
            accepted_event_ids=[],
            rejected_event_ids=[],
            merged_event_ids=[],
            noop_event_ids=noop_ids,
        ),
        noop_event_ids=noop_ids,
        warnings=list(warnings or []),
        created_at=clock,
    )


def reduce_llm_inference_trace_events(
    *,
    events: list[GrammarEventV1],
    projection: LlmInferenceProjectionV1,
    now: datetime | None = None,
    reducer_id: str = LLM_INFERENCE_REDUCER_ID,
) -> tuple[LlmInferenceProjectionV1, ReductionReceiptV1]:
    """One gateway window trace -> one receipt with one delta per serving node.

    Each window is a fresh reading (the gateway resets its counters every flush),
    so a node's state is replaced, not accumulated. A node with no upstream
    traffic this window gets no pressure hint at all -- "not measured" must not
    be written into the field as a calm 0.0.
    """
    clock = _utc_now(now)
    if not events:
        return projection, _noop(reducer_id, [], clock)
    trace_id = events[0].trace_id or ""
    if not parse_llm_inference_trace_id(trace_id):
        return projection, _noop(reducer_id, events, clock)
    if any(e.provenance.source_service != LLM_INFERENCE_SOURCE_SERVICE for e in events):
        return projection, _noop(reducer_id, events, clock)

    try:
        states, unattributed = extract_llm_inference_states_from_events(events, now=clock)
    except ValueError as exc:
        return projection, _noop(reducer_id, events, clock, warnings=[str(exc)])

    updated = deepcopy(projection)
    updated.projection_id = LLM_INFERENCE_PROJECTION_ID
    updated.generated_at = clock
    updated.last_unattributed_calls = unattributed
    updated.last_window_id = parse_llm_inference_trace_id(trace_id)[1]  # type: ignore[index]

    accepted = [e.event_id for e in events if e.atom and (e.atom.semantic_role or "").strip()]
    deltas: list[StateDeltaV1] = []
    projection_updates: list[ProjectionUpdateV1] = []
    for target_id, state in states.items():
        existing = updated.nodes.get(target_id)
        operation = "create" if existing is None else "update"
        updated.nodes[target_id] = state
        after = state.model_dump(mode="json")
        hints: dict[str, float] = {}
        if state.inference_failure_pressure is not None:
            hints["inference_failure_pressure"] = float(state.inference_failure_pressure)
        after["pressure_hints"] = hints
        deltas.append(
            StateDeltaV1(
                delta_id=stable_delta_id(
                    reducer_id=reducer_id,
                    target_projection=LLM_INFERENCE_PROJECTION_ID,
                    target_kind=LLM_INFERENCE_TARGET_KIND,
                    target_id=target_id,
                    operation=operation,
                    caused_by_event_ids=state.evidence_event_ids,
                ),
                target_projection=LLM_INFERENCE_PROJECTION_ID,
                target_kind=LLM_INFERENCE_TARGET_KIND,
                target_id=target_id,
                operation=operation,
                before=existing.model_dump(mode="json") if existing else None,
                after=after,
                caused_by_event_ids=state.evidence_event_ids,
                reducer_id=reducer_id,
            )
        )
        projection_updates.append(
            ProjectionUpdateV1(
                projection_kind=LLM_INFERENCE_TARGET_KIND,
                projection_id=LLM_INFERENCE_PROJECTION_ID,
                node_id=state.node_id,
                operation=operation,
            )
        )

    receipt = ReductionReceiptV1(
        receipt_id=stable_receipt_id(
            reducer_id=reducer_id,
            accepted_event_ids=accepted,
            rejected_event_ids=[],
            merged_event_ids=[],
            noop_event_ids=[],
        ),
        accepted_event_ids=accepted,
        state_deltas=deltas,
        projection_updates=projection_updates,
        created_at=clock,
    )
    return updated, receipt
