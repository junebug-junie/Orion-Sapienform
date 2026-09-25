from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable

from orion.schemas.grammar import GrammarEventV1
from orion.schemas.llm_inference_projection import LlmInferenceProjectionV1

from .constants import LLM_INFERENCE_PROJECTION_ID
from .reducer import reduce_llm_inference_trace_events

ProjectionLoader = Callable[[], LlmInferenceProjectionV1]
ProjectionSaver = Callable[[LlmInferenceProjectionV1], None]
ReceiptSaver = Callable[[Any], None]


def process_llm_inference_grammar_events(
    *,
    events: list[GrammarEventV1],
    load_projection: ProjectionLoader,
    save_projection: ProjectionSaver,
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
        projection, receipt = reduce_llm_inference_trace_events(
            events=trace_events,
            projection=projection,
            now=clock,
        )
        save_receipt(receipt)
        stats["receipts"] += 1
    save_projection(projection)
    return stats


def empty_llm_inference_projection(*, now: datetime) -> LlmInferenceProjectionV1:
    return LlmInferenceProjectionV1(
        projection_id=LLM_INFERENCE_PROJECTION_ID,
        generated_at=now,
        nodes={},
    )
