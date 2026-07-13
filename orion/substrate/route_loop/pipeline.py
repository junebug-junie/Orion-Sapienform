from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable

from orion.schemas.grammar import GrammarEventV1
from orion.schemas.route_projection import RouteArbitrationProjectionV1

from .constants import ROUTE_ARBITRATION_MAX_AGE_SEC, ROUTE_ARBITRATION_MAX_RUNS
from .reducer import reduce_route_trace_events

RouteProjectionLoader = Callable[[], RouteArbitrationProjectionV1]
RouteProjectionSaver = Callable[[RouteArbitrationProjectionV1], None]
ReceiptSaver = Callable[[Any], None]


def process_route_grammar_events(
    *,
    events: list[GrammarEventV1],
    load_projection: RouteProjectionLoader,
    save_projection: RouteProjectionSaver,
    save_receipt: ReceiptSaver,
    now: datetime | None = None,
    max_runs: int | None = ROUTE_ARBITRATION_MAX_RUNS,
    max_age_sec: float | None = ROUTE_ARBITRATION_MAX_AGE_SEC,
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
        projection, receipt = reduce_route_trace_events(
            events=trace_events,
            projection=projection,
            now=clock,
            max_runs=max_runs,
            max_age_sec=max_age_sec,
        )
        save_receipt(receipt)
        stats["receipts"] += 1

    save_projection(projection)
    return stats
