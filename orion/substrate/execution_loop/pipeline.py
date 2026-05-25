from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable

from orion.schemas.execution_projection import ExecutionTrajectoryProjectionV1
from orion.schemas.grammar import GrammarEventV1

from .reducer import reduce_execution_trace_events


ExecutionProjectionLoader = Callable[[], ExecutionTrajectoryProjectionV1]
ExecutionProjectionSaver = Callable[[ExecutionTrajectoryProjectionV1], None]
ReceiptSaver = Callable[[Any], None]


def process_execution_grammar_events(
    *,
    events: list[GrammarEventV1],
    load_projection: ExecutionProjectionLoader,
    save_projection: ExecutionProjectionSaver,
    save_receipt: ReceiptSaver,
    now: datetime | None = None,
) -> dict[str, int]:
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
        projection, receipt = reduce_execution_trace_events(
            events=trace_events,
            projection=projection,
            now=clock,
        )
        save_receipt(receipt)
        stats["receipts"] += 1

    save_projection(projection)
    return stats
