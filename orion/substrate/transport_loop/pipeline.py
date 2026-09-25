from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable

from orion.schemas.grammar import GrammarEventV1
from orion.schemas.transport_projection import TransportBusProjectionV1

from .constants import (
    DEFAULT_STREAM_DEPTH_CRITICAL,
    NON_BUS_TRANSPORT_NODE_IDS,
    NON_BUS_TRANSPORT_TARGET_IDS,
    TRANSPORT_BUS_PROJECTION_ID,
)
from .reducer import TraceEventsLoader, reduce_transport_trace_events

TransportProjectionLoader = Callable[[], TransportBusProjectionV1]
TransportProjectionSaver = Callable[[TransportBusProjectionV1], None]
ReceiptSaver = Callable[[Any], None]

logger = logging.getLogger(__name__)


def prune_non_bus_entries(projection: TransportBusProjectionV1) -> list[str]:
    """Drop persisted `buses` entries that were never buses (e.g. the
    `bus:rpc_timeout` phantom minted before NON_BUS_TRANSPORT_NODE_IDS was
    excluded at parse time). Mutates in place; returns the dropped keys.

    Runs on every load so already-persisted state self-heals on the first
    batch after deploy -- no one-off SQL patch needed.
    """
    dropped = [
        key
        for key, state in projection.buses.items()
        if state.node_id in NON_BUS_TRANSPORT_NODE_IDS
        or key in NON_BUS_TRANSPORT_TARGET_IDS
    ]
    for key in dropped:
        projection.buses.pop(key, None)
    return dropped


def process_transport_grammar_events(
    *,
    events: list[GrammarEventV1],
    load_projection: TransportProjectionLoader,
    save_projection: TransportProjectionSaver,
    save_receipt: ReceiptSaver,
    now: datetime | None = None,
    stream_depth_critical: int = DEFAULT_STREAM_DEPTH_CRITICAL,
    load_trace_events: TraceEventsLoader | None = None,
) -> dict[str, int]:
    """`load_trace_events` lets a trace cut across two cursor batches be
    reduced from its whole stored trace instead of from the piece in hand
    (see reduce_transport_trace_events). Without it, pieces are held."""
    clock = now or datetime.now(timezone.utc)
    stats = {"events": 0, "receipts": 0, "traces": 0}

    by_trace: dict[str, list[GrammarEventV1]] = defaultdict(list)
    for event in events:
        stats["events"] += 1
        by_trace[event.trace_id or ""].append(event)

    projection = load_projection()
    dropped = prune_non_bus_entries(projection)
    if dropped:
        # Runtime proof the persisted phantom self-healed after deploy.
        logger.info("transport_projection_pruned_non_bus keys=%s", dropped)
    for trace_id, trace_events in by_trace.items():
        if not trace_id:
            continue
        stats["traces"] += 1
        projection, receipt = reduce_transport_trace_events(
            events=trace_events,
            projection=projection,
            now=clock,
            stream_depth_critical=stream_depth_critical,
            load_trace_events=load_trace_events,
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
