from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable

from orion.schemas.grammar import GrammarEventV1
from orion.schemas.storage_write_projection import StorageWriteProjectionV1

from .constants import STORAGE_WRITE_PROJECTION_ID
from .reducer import reduce_storage_write_trace_events

ProjectionLoader = Callable[[], StorageWriteProjectionV1]
ProjectionSaver = Callable[[StorageWriteProjectionV1], None]
ReceiptSaver = Callable[[Any], None]


def process_storage_write_grammar_events(
    *,
    events: list[GrammarEventV1],
    load_projection: ProjectionLoader,
    save_projection: ProjectionSaver,
    save_receipt: ReceiptSaver,
    now: datetime | None = None,
) -> dict[str, int]:
    """Group a cursor batch by trace (one trace = one writer window) and reduce
    each in event-time order, so the rolling span slides the same way it would
    have live."""
    clock = now or datetime.now(timezone.utc)
    stats = {"events": 0, "receipts": 0, "traces": 0}
    by_trace: dict[str, list[GrammarEventV1]] = defaultdict(list)
    for event in events:
        stats["events"] += 1
        by_trace[event.trace_id or ""].append(event)

    projection = load_projection()
    for trace_id, trace_events in by_trace.items():
        if not trace_id:
            continue
        stats["traces"] += 1
        projection, receipt = reduce_storage_write_trace_events(
            events=trace_events,
            projection=projection,
            now=clock,
        )
        save_receipt(receipt)
        stats["receipts"] += 1
    save_projection(projection)
    return stats


def empty_storage_write_projection(*, now: datetime) -> StorageWriteProjectionV1:
    return StorageWriteProjectionV1(
        projection_id=STORAGE_WRITE_PROJECTION_ID,
        generated_at=now,
    )
