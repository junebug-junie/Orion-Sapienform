from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable

from orion.schemas.chat_projection import ChatSessionProjectionV1
from orion.schemas.grammar import GrammarEventV1

from .constants import CHAT_TRACE_PREFIX
from .reducer import reduce_chat_trace_events


ChatProjectionLoader = Callable[[], ChatSessionProjectionV1]
ChatProjectionSaver = Callable[[ChatSessionProjectionV1], None]
ReceiptSaver = Callable[[Any], None]


def process_chat_grammar_events(
    *,
    events: list[GrammarEventV1],
    load_projection: ChatProjectionLoader,
    save_projection: ChatProjectionSaver,
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
        if not trace_id.startswith(CHAT_TRACE_PREFIX):
            continue
        stats["traces"] += 1
        projection, receipt = reduce_chat_trace_events(
            events=trace_events,
            projection=projection,
            now=clock,
        )
        save_receipt(receipt)
        stats["receipts"] += 1

    save_projection(projection)
    return stats
