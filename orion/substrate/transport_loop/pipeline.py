from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable

from orion.schemas.grammar import GrammarEventV1
from orion.schemas.transport_projection import TransportBusProjectionV1

from .constants import DEFAULT_STREAM_DEPTH_CRITICAL, TRANSPORT_BUS_PROJECTION_ID
from .reducer import reduce_transport_trace_events

TransportProjectionLoader = Callable[[], TransportBusProjectionV1]
TransportProjectionSaver = Callable[[TransportBusProjectionV1], None]
ReceiptSaver = Callable[[Any], None]


def process_transport_grammar_events(
    *,
    events: list[GrammarEventV1],
    load_projection: TransportProjectionLoader,
    save_projection: TransportProjectionSaver,
    save_receipt: ReceiptSaver,
    now: datetime | None = None,
    stream_depth_critical: int = DEFAULT_STREAM_DEPTH_CRITICAL,
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
        projection, receipt = reduce_transport_trace_events(
            events=trace_events,
            projection=projection,
            now=clock,
            stream_depth_critical=stream_depth_critical,
        )
        save_receipt(receipt)
        stats["receipts"] += 1

    save_projection(projection)
    return stats


def empty_transport_projection(*, now: datetime) -> TransportBusProjectionV1:
    return TransportBusProjectionV1(
        projection_id=TRANSPORT_BUS_PROJECTION_ID,
        updated_at=now,
        buses={},
    )
