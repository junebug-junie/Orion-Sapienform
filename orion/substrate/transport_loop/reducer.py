from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

from orion.schemas.grammar import GrammarEventV1
from orion.schemas.reduction_receipt import ProjectionUpdateV1, ReductionReceiptV1
from orion.schemas.state_delta import StateDeltaV1
from orion.schemas.transport_projection import TransportBusProjectionV1
from orion.substrate.ids import stable_delta_id, stable_receipt_id

from .constants import (
    DEFAULT_STREAM_DEPTH_CRITICAL,
    TRANSPORT_BUS_PROJECTION_ID,
    TRANSPORT_REDUCER_ID,
    TRANSPORT_SOURCE_SERVICE,
)
from .extract import extract_transport_bus_state_from_events, parse_bus_transport_trace_id


def _utc_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    return now if now.tzinfo else now.replace(tzinfo=timezone.utc)


def reduce_transport_trace_events(
    *,
    events: list[GrammarEventV1],
    projection: TransportBusProjectionV1,
    now: datetime | None = None,
    reducer_id: str = TRANSPORT_REDUCER_ID,
    stream_depth_critical: int = DEFAULT_STREAM_DEPTH_CRITICAL,
) -> tuple[TransportBusProjectionV1, ReductionReceiptV1]:
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
    if not parse_bus_transport_trace_id(trace_id):
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

    if any(e.provenance.source_service != TRANSPORT_SOURCE_SERVICE for e in events):
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
    updated.updated_at = clock
    if updated.projection_id != TRANSPORT_BUS_PROJECTION_ID:
        updated.projection_id = TRANSPORT_BUS_PROJECTION_ID

    warnings: list[str] = []
    try:
        incoming = extract_transport_bus_state_from_events(
            events,
            now=clock,
            stream_depth_critical=stream_depth_critical,
        )
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

    existing = updated.buses.get(incoming.target_id)
    operation = "create" if existing is None else "update"
    updated.buses[incoming.target_id] = incoming

    event_ids = [
        e.event_id
        for e in events
        if e.atom and (e.atom.semantic_role or "").strip() not in {"", "trace_started", "trace_ended", "edge_emitted"}
    ]
    after_payload = incoming.model_dump(mode="json")
    after_payload["pressure_hints"] = {
        k: after_payload[k]
        for k in (
            "bus_health",
            "delivery_confidence",
            "stream_depth_pressure",
            "backpressure",
            "catalog_drift_pressure",
            "observer_failure_pressure",
            "transport_pressure",
            "contract_pressure",
            "reliability_pressure",
        )
    }

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
                    target_projection=TRANSPORT_BUS_PROJECTION_ID,
                    target_kind="transport_bus",
                    target_id=incoming.target_id,
                    operation=operation,
                    caused_by_event_ids=event_ids,
                ),
                target_projection=TRANSPORT_BUS_PROJECTION_ID,
                target_kind="transport_bus",
                target_id=incoming.target_id,
                operation=operation,
                before=existing.model_dump(mode="json") if existing else None,
                after=after_payload,
                caused_by_event_ids=event_ids,
                reducer_id=reducer_id,
            )
        ],
        projection_updates=[
            ProjectionUpdateV1(
                projection_kind="transport_bus",
                projection_id=TRANSPORT_BUS_PROJECTION_ID,
                node_id=incoming.node_id,
                operation=operation,
            )
        ],
        warnings=warnings,
        created_at=clock,
    )
    return updated, receipt
