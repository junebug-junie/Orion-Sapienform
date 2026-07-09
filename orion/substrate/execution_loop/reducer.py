from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

from orion.schemas.execution_projection import ExecutionTrajectoryProjectionV1
from orion.schemas.grammar import GrammarEventV1
from orion.schemas.reduction_receipt import ProjectionUpdateV1, ReductionReceiptV1
from orion.schemas.state_delta import StateDeltaV1
from orion.substrate.ids import stable_delta_id, stable_receipt_id

from .constants import (
    EXECUTION_REDUCER_ID,
    EXECUTION_SOURCE_SERVICES,
    EXECUTION_TRAJECTORY_PROJECTION_ID,
)
from .grammar_extract import extract_execution_state_from_events
from .ids import parse_execution_trace_id
from .merge import merge_execution_run_state


def _utc_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    return now if now.tzinfo else now.replace(tzinfo=timezone.utc)


def reduce_execution_trace_events(
    *,
    events: list[GrammarEventV1],
    projection: ExecutionTrajectoryProjectionV1,
    now: datetime | None = None,
    reducer_id: str = EXECUTION_REDUCER_ID,
) -> tuple[ExecutionTrajectoryProjectionV1, ReductionReceiptV1]:
    clock = _utc_now(now)
    if not events:
        receipt = ReductionReceiptV1(
            receipt_id=stable_receipt_id(
                reducer_id=reducer_id,
                accepted_event_ids=[],
                rejected_event_ids=[],
                merged_event_ids=[],
                noop_event_ids=[],
            ),
            noop_event_ids=[],
            created_at=clock,
        )
        return projection, receipt

    trace_id = events[0].trace_id or ""
    if not parse_execution_trace_id(trace_id):
        noop_ids = [e.event_id for e in events]
        return projection, ReductionReceiptV1(
            receipt_id=stable_receipt_id(
                reducer_id=reducer_id,
                accepted_event_ids=[],
                rejected_event_ids=[],
                merged_event_ids=[],
                noop_event_ids=noop_ids,
            ),
            noop_event_ids=noop_ids,
            created_at=clock,
        )

    if any(e.provenance.source_service not in EXECUTION_SOURCE_SERVICES for e in events):
        noop_ids = [e.event_id for e in events]
        return projection, ReductionReceiptV1(
            receipt_id=stable_receipt_id(
                reducer_id=reducer_id,
                accepted_event_ids=[],
                rejected_event_ids=[],
                merged_event_ids=[],
                noop_event_ids=noop_ids,
            ),
            noop_event_ids=noop_ids,
            created_at=clock,
        )

    processable: list[GrammarEventV1] = []
    harness_fcc_noop_ids: list[str] = []
    for event in events:
        role = (event.atom.semantic_role or "") if event.atom else ""
        if role == "harness_fcc_step":
            harness_fcc_noop_ids.append(event.event_id)
        else:
            processable.append(event)

    if not processable:
        return projection, ReductionReceiptV1(
            receipt_id=stable_receipt_id(
                reducer_id=reducer_id,
                accepted_event_ids=[],
                rejected_event_ids=[],
                merged_event_ids=[],
                noop_event_ids=harness_fcc_noop_ids,
            ),
            noop_event_ids=harness_fcc_noop_ids,
            created_at=clock,
        )

    events = processable
    fcc_noop_ids = harness_fcc_noop_ids

    updated = deepcopy(projection)
    updated.generated_at = clock
    if updated.projection_id != EXECUTION_TRAJECTORY_PROJECTION_ID:
        updated.projection_id = EXECUTION_TRAJECTORY_PROJECTION_ID

    warnings: list[str] = []
    try:
        incoming = extract_execution_state_from_events(events, now=clock)
    except ValueError as exc:
        warnings.append(str(exc))
        noop_ids = [e.event_id for e in events]
        return projection, ReductionReceiptV1(
            receipt_id=stable_receipt_id(
                reducer_id=reducer_id,
                accepted_event_ids=[],
                rejected_event_ids=[],
                merged_event_ids=[],
                noop_event_ids=noop_ids,
            ),
            noop_event_ids=noop_ids,
            warnings=warnings,
            created_at=clock,
        )

    existing = updated.runs.get(trace_id)
    operation = "create" if existing is None else "update"
    merged = merge_execution_run_state(existing, incoming)
    updated.runs[trace_id] = merged

    event_ids = [e.event_id for e in events if e.atom]
    receipt = ReductionReceiptV1(
        receipt_id=stable_receipt_id(
            reducer_id=reducer_id,
            accepted_event_ids=event_ids,
            rejected_event_ids=[],
            merged_event_ids=[],
            noop_event_ids=fcc_noop_ids,
        ),
        accepted_event_ids=event_ids,
        noop_event_ids=fcc_noop_ids,
        state_deltas=[
            StateDeltaV1(
                delta_id=stable_delta_id(
                    reducer_id=reducer_id,
                    target_projection=EXECUTION_TRAJECTORY_PROJECTION_ID,
                    target_kind="execution_run",
                    target_id=trace_id,
                    operation=operation,
                    caused_by_event_ids=event_ids,
                ),
                target_projection=EXECUTION_TRAJECTORY_PROJECTION_ID,
                target_kind="execution_run",
                target_id=trace_id,
                operation=operation,
                before=existing.model_dump(mode="json") if existing else None,
                after=merged.model_dump(mode="json"),
                caused_by_event_ids=event_ids,
                reducer_id=reducer_id,
            )
        ],
        projection_updates=[
            ProjectionUpdateV1(
                projection_kind="execution_trajectory",
                projection_id=EXECUTION_TRAJECTORY_PROJECTION_ID,
                node_id=merged.node_id,
                operation=operation,
            )
        ],
        warnings=warnings,
        created_at=clock,
    )
    return updated, receipt
