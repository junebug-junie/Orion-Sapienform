from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

from orion.schemas.chat_projection import ChatSessionProjectionV1
from orion.schemas.grammar import GrammarEventV1
from orion.schemas.reduction_receipt import ProjectionUpdateV1, ReductionReceiptV1
from orion.schemas.state_delta import StateDeltaV1
from orion.substrate.ids import stable_delta_id, stable_receipt_id

from .constants import (
    CHAT_REDUCER_ID,
    CHAT_SESSION_PROJECTION_ID,
    CHAT_SOURCE_SERVICE,
    CHAT_TRACE_PREFIX,
)
from .grammar_extract import compute_chat_pressure_hints, extract_chat_turn_state

_EVIDENCE_CAP = 50


def _utc_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    return now if now.tzinfo else now.replace(tzinfo=timezone.utc)


def reduce_chat_trace_events(
    *,
    events: list[GrammarEventV1],
    projection: ChatSessionProjectionV1,
    now: datetime | None = None,
    reducer_id: str = CHAT_REDUCER_ID,
) -> tuple[ChatSessionProjectionV1, ReductionReceiptV1]:
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
    if not trace_id.startswith(CHAT_TRACE_PREFIX):
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

    if any(e.provenance.source_service != CHAT_SOURCE_SERVICE for e in events):
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
    if updated.projection_id != CHAT_SESSION_PROJECTION_ID:
        updated.projection_id = CHAT_SESSION_PROJECTION_ID

    warnings: list[str] = []
    try:
        turn = extract_chat_turn_state(events, now=clock)
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

    turn_id = turn.turn_id
    session_id = turn.session_id
    existing = updated.turns.get(turn_id)
    operation = "create" if existing is None else "update"
    updated.turns[turn_id] = turn
    updated.total_turn_count = len(updated.turns)
    if session_id and session_id not in updated.sessions:
        updated.sessions.append(session_id)

    event_ids = [e.event_id for e in events if e.atom]
    capped_event_ids = event_ids[:_EVIDENCE_CAP]

    pressure_hints = compute_chat_pressure_hints(turn)
    after = {**turn.model_dump(mode="json"), "pressure_hints": pressure_hints}

    receipt = ReductionReceiptV1(
        receipt_id=stable_receipt_id(
            reducer_id=reducer_id,
            accepted_event_ids=event_ids,
            rejected_event_ids=[],
            merged_event_ids=[],
            noop_event_ids=[],
        ),
        accepted_event_ids=event_ids,
        state_deltas=[
            StateDeltaV1(
                delta_id=stable_delta_id(
                    reducer_id=reducer_id,
                    target_projection=CHAT_SESSION_PROJECTION_ID,
                    target_kind="chat_turn",
                    target_id=session_id,
                    operation=operation,
                    caused_by_event_ids=capped_event_ids,
                ),
                target_projection=CHAT_SESSION_PROJECTION_ID,
                target_kind="chat_turn",
                target_id=session_id,
                operation=operation,
                before=existing.model_dump(mode="json") if existing else None,
                after=after,
                caused_by_event_ids=capped_event_ids,
                reducer_id=reducer_id,
            )
        ],
        projection_updates=[
            ProjectionUpdateV1(
                projection_kind="chat_session",
                projection_id=CHAT_SESSION_PROJECTION_ID,
                node_id=turn.node_id,
                operation=operation,
            )
        ],
        warnings=warnings,
        created_at=clock,
    )
    return updated, receipt
