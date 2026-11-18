from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

from orion.schemas.grammar import GrammarEventV1
from orion.schemas.reduction_receipt import ProjectionUpdateV1, ReductionReceiptV1
from orion.schemas.state_delta import StateDeltaV1
from orion.schemas.storage_write_projection import (
    StorageWriteProjectionV1,
    StorageWriteWindowCountV1,
)
from orion.substrate.ids import stable_delta_id, stable_receipt_id

from .constants import (
    STORAGE_WRITE_CHANNEL,
    STORAGE_WRITE_NODE_ID,
    STORAGE_WRITE_PROJECTION_ID,
    STORAGE_WRITE_REDUCER_ID,
    STORAGE_WRITE_SOURCE_SERVICE,
    STORAGE_WRITE_TARGET_KIND,
)
from .extract import extract_storage_write_window, parse_storage_write_trace_id
from .failure_window import failure_reading, fold_window

# Families named in the delta (worst first). The projection keeps all of them.
_DELTA_FAMILY_LIMIT = 8


def _utc_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    return now if now.tzinfo else now.replace(tzinfo=timezone.utc)


def _noop(reducer_id: str, events: list[GrammarEventV1], clock: datetime, warnings: list[str] | None = None) -> ReductionReceiptV1:
    noop_ids = [e.event_id for e in events]
    return ReductionReceiptV1(
        receipt_id=stable_receipt_id(
            reducer_id=reducer_id,
            accepted_event_ids=[],
            rejected_event_ids=[],
            merged_event_ids=[],
            noop_event_ids=noop_ids,
        ),
        noop_event_ids=noop_ids,
        warnings=list(warnings or []),
        created_at=clock,
    )


def reduce_storage_write_trace_events(
    *,
    events: list[GrammarEventV1],
    projection: StorageWriteProjectionV1,
    now: datetime | None = None,
    reducer_id: str = STORAGE_WRITE_REDUCER_ID,
) -> tuple[StorageWriteProjectionV1, ReductionReceiptV1]:
    """One sql-writer window trace -> one receipt with one delta on
    ``node:substrate.storage_write``.

    The window's per-family counts are folded into a 600 s event-time span
    (``failure_window.py``) and the reading is the worst family's (or the pool's)
    floored failure share. When nothing was attempted in the whole span the
    delta carries no pressure hint at all: "not measured" is never written into
    the field as a calm 0.0, and the field channel expires on its own.
    """
    clock = _utc_now(now)
    if not events:
        return projection, _noop(reducer_id, [], clock)
    trace_id = events[0].trace_id or ""
    parsed = parse_storage_write_trace_id(trace_id)
    if not parsed:
        return projection, _noop(reducer_id, events, clock)
    if any(e.provenance.source_service != STORAGE_WRITE_SOURCE_SERVICE for e in events):
        return projection, _noop(reducer_id, events, clock)
    try:
        window = extract_storage_write_window(events, now=clock)
    except ValueError as exc:
        return projection, _noop(reducer_id, events, clock, warnings=[str(exc)])
    if not window.evidence_event_ids:
        return projection, _noop(reducer_id, events, clock, warnings=["no storage_write atoms"])

    writer, window_id = parsed
    window_key = f"{writer}:{window_id}"
    window_end = window.window_end or clock

    updated = deepcopy(projection)
    same_window = updated.last_window_id == window_key
    updated.projection_id = STORAGE_WRITE_PROJECTION_ID
    updated.generated_at = clock
    updated.writer_node = writer
    updated.last_window_id = window_key
    updated.last_window_end = window_end
    if window.window_sec is not None:
        updated.last_window_sec = window.window_sec
    if window.grammar_queue_max is not None:
        updated.grammar_queue_max = window.grammar_queue_max
    # A trace split across reducer batches arrives in parts: merge by family.
    families = dict(updated.families) if same_window else {}
    families.update(window.families)
    updated.families = families
    updated.attempted = sum(f.attempted for f in families.values())
    updated.committed = sum(f.committed for f in families.values())
    updated.failed = sum(f.failed for f in families.values())
    updated.unrouted = sum(f.unrouted for f in families.values())

    count = StorageWriteWindowCountV1(
        window_id=window_key,
        window_end=window_end,
        attempted={k: f.attempted for k, f in window.families.items() if f.attempted},
        failed={k: f.failed for k, f in window.families.items() if f.failed},
    )
    updated.recent_windows = fold_window(updated.recent_windows, count)
    reading = failure_reading(updated.recent_windows)
    updated.write_failure_pressure = reading.pressure
    updated.reading = reading.as_dict()
    updated.evidence_event_ids = list(window.evidence_event_ids)

    hints: dict[str, float] = {}
    if reading.pressure is not None:
        hints[STORAGE_WRITE_CHANNEL] = float(reading.pressure)
    worst = sorted(
        families.values(),
        key=lambda f: (-(f.failed / max(f.attempted, 1)), -f.failed, f.family),
    )[:_DELTA_FAMILY_LIMIT]
    after = {
        "node_id": STORAGE_WRITE_NODE_ID,
        "writer_node": writer,
        "window_id": window_key,
        "window_end": window_end.isoformat(),
        "window_complete": window.completed,
        "attempted": updated.attempted,
        "committed": updated.committed,
        "failed": updated.failed,
        "unrouted": updated.unrouted,
        "grammar_queue_max": updated.grammar_queue_max,
        "families": [f.model_dump(mode="json") for f in worst],
        "failure_window": reading.as_dict(),
        "pressure_hints": hints,
    }
    existing_before = projection.reading if projection.last_window_id else None
    operation = "create" if projection.last_window_id is None else "update"
    delta = StateDeltaV1(
        delta_id=stable_delta_id(
            reducer_id=reducer_id,
            target_projection=STORAGE_WRITE_PROJECTION_ID,
            target_kind=STORAGE_WRITE_TARGET_KIND,
            target_id=STORAGE_WRITE_NODE_ID,
            operation=operation,
            caused_by_event_ids=window.evidence_event_ids,
        ),
        target_projection=STORAGE_WRITE_PROJECTION_ID,
        target_kind=STORAGE_WRITE_TARGET_KIND,
        target_id=STORAGE_WRITE_NODE_ID,
        operation=operation,
        before={"failure_window": existing_before} if existing_before else None,
        after=after,
        caused_by_event_ids=list(window.evidence_event_ids),
        reducer_id=reducer_id,
    )
    accepted = list(window.evidence_event_ids)
    noop = [e.event_id for e in events if e.event_id not in set(accepted)]
    receipt = ReductionReceiptV1(
        receipt_id=stable_receipt_id(
            reducer_id=reducer_id,
            accepted_event_ids=accepted,
            rejected_event_ids=[],
            merged_event_ids=[],
            noop_event_ids=noop,
        ),
        accepted_event_ids=accepted,
        noop_event_ids=noop,
        state_deltas=[delta],
        projection_updates=[
            ProjectionUpdateV1(
                projection_kind=STORAGE_WRITE_TARGET_KIND,
                projection_id=STORAGE_WRITE_PROJECTION_ID,
                node_id=STORAGE_WRITE_NODE_ID,
                operation=operation,
            )
        ],
        created_at=clock,
    )
    return updated, receipt
