from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable

from orion.schemas.grammar import GrammarEventV1
from orion.schemas.vision_organ_projection import VisionOrganProjectionV1

from .constants import VISION_ORGAN_PROJECTION_ID
from .reducer import reduce_vision_organ_trace_events

ProjectionLoader = Callable[[], VisionOrganProjectionV1]
ProjectionSaver = Callable[[VisionOrganProjectionV1], None]
ReceiptSaver = Callable[[Any], None]


def process_vision_organ_grammar_events(
    *,
    events: list[GrammarEventV1],
    load_projection: ProjectionLoader,
    save_projection: ProjectionSaver,
    save_receipt: ReceiptSaver,
    now: datetime | None = None,
) -> dict[str, int]:
    """Traces are reduced in first-seen order, so windows replay oldest first."""
    clock = now or datetime.now(timezone.utc)
    stats = {"events": 0, "receipts": 0, "traces": 0, "deltas": 0}
    by_trace: dict[str, list[GrammarEventV1]] = defaultdict(list)
    for event in events:
        stats["events"] += 1
        by_trace[event.trace_id or ""].append(event)

    projection = load_projection()
    for trace_id, trace_events in by_trace.items():
        if not trace_id:
            continue
        stats["traces"] += 1
        projection, receipt = reduce_vision_organ_trace_events(
            events=trace_events,
            projection=projection,
            now=clock,
        )
        save_receipt(receipt)
        stats["receipts"] += 1
        stats["deltas"] += len(receipt.state_deltas)
    save_projection(projection)
    return stats


def empty_vision_organ_projection(*, now: datetime) -> VisionOrganProjectionV1:
    return VisionOrganProjectionV1(
        projection_id=VISION_ORGAN_PROJECTION_ID,
        generated_at=now,
    )
