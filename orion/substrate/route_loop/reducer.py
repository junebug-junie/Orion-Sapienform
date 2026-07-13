from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone

from orion.schemas.grammar import GrammarEventV1
from orion.schemas.reduction_receipt import ProjectionUpdateV1, ReductionReceiptV1
from orion.schemas.route_projection import RouteArbitrationProjectionV1
from orion.schemas.state_delta import StateDeltaV1
from orion.substrate.ids import stable_delta_id, stable_receipt_id

from .constants import (
    ROUTE_ARBITRATION_MAX_AGE_SEC,
    ROUTE_ARBITRATION_MAX_RUNS,
    ROUTE_ARBITRATION_PROJECTION_ID,
    ROUTE_REDUCER_ID,
    ROUTE_SOURCE_SERVICE,
)
from .grammar_extract import extract_route_state_from_events
from .ids import parse_route_trace_id
from .merge import merge_route_run_state


def _utc_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    return now if now.tzinfo else now.replace(tzinfo=timezone.utc)


def _evict_stale_runs(
    projection: RouteArbitrationProjectionV1,
    *,
    clock: datetime,
    max_runs: int | None,
    max_age_sec: float | None,
    protected_trace_id: str,
) -> None:
    """Prune `projection.runs` in place by LRU (last_updated_at).

    Mutates `projection.runs` directly. Only called on the success path,
    right after `protected_trace_id` has been written, so `protected_trace_id`
    is always excluded from eviction candidates -- this is a structural
    guarantee, not one inferred from timestamp freshness: batches processed
    together (see pipeline.py) share a single `clock`, so multiple runs can
    legitimately tie on `last_updated_at`, and a stable sort alone would be
    free to evict whichever tied run happens to sit earlier in dict order
    (e.g. a pre-existing run whose position predates this tick).
    """
    if max_age_sec is not None:
        cutoff = clock - timedelta(seconds=max_age_sec)
        stale_ids = [
            trace_id
            for trace_id, run in projection.runs.items()
            if trace_id != protected_trace_id and run.last_updated_at < cutoff
        ]
        for trace_id in stale_ids:
            del projection.runs[trace_id]

    if max_runs is not None and len(projection.runs) > max_runs:
        candidates = sorted(
            (item for item in projection.runs.items() if item[0] != protected_trace_id),
            key=lambda item: item[1].last_updated_at,
        )
        excess = len(projection.runs) - max_runs
        for trace_id, _run in candidates[:excess]:
            del projection.runs[trace_id]


def reduce_route_trace_events(
    *,
    events: list[GrammarEventV1],
    projection: RouteArbitrationProjectionV1,
    now: datetime | None = None,
    reducer_id: str = ROUTE_REDUCER_ID,
    max_runs: int | None = ROUTE_ARBITRATION_MAX_RUNS,
    max_age_sec: float | None = ROUTE_ARBITRATION_MAX_AGE_SEC,
) -> tuple[RouteArbitrationProjectionV1, ReductionReceiptV1]:
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
    if not parse_route_trace_id(trace_id):
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

    if any(e.provenance.source_service != ROUTE_SOURCE_SERVICE for e in events):
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

    updated = deepcopy(projection)
    updated.generated_at = clock
    if updated.projection_id != ROUTE_ARBITRATION_PROJECTION_ID:
        updated.projection_id = ROUTE_ARBITRATION_PROJECTION_ID

    warnings: list[str] = []
    try:
        incoming = extract_route_state_from_events(events, now=clock)
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
    merged = merge_route_run_state(existing, incoming)
    updated.runs[trace_id] = merged
    _evict_stale_runs(
        updated,
        clock=clock,
        max_runs=max_runs,
        max_age_sec=max_age_sec,
        protected_trace_id=trace_id,
    )

    event_ids = [e.event_id for e in events if e.atom]
    receipt = ReductionReceiptV1(
        receipt_id=stable_receipt_id(
            reducer_id=reducer_id,
            accepted_event_ids=event_ids,
            rejected_event_ids=[],
            merged_event_ids=[],
            noop_event_ids=[],
        ),
        accepted_event_ids=event_ids,
        noop_event_ids=[],
        state_deltas=[
            StateDeltaV1(
                delta_id=stable_delta_id(
                    reducer_id=reducer_id,
                    target_projection=ROUTE_ARBITRATION_PROJECTION_ID,
                    target_kind="route_arbitration_run",
                    target_id=trace_id,
                    operation=operation,
                    caused_by_event_ids=event_ids,
                ),
                target_projection=ROUTE_ARBITRATION_PROJECTION_ID,
                target_kind="route_arbitration_run",
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
                projection_kind="route_arbitration",
                projection_id=ROUTE_ARBITRATION_PROJECTION_ID,
                node_id=merged.node_id,
                operation=operation,
            )
        ],
        warnings=warnings,
        created_at=clock,
    )
    return updated, receipt
