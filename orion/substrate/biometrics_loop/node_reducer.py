from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

from orion.biometrics.node_catalog import NodeCatalog
from orion.substrate.ids import stable_delta_id, stable_receipt_id
from orion.schemas.biometrics_projection import NodeBiometricsProjectionV1
from orion.schemas.grammar import GrammarEventV1
from orion.schemas.reduction_receipt import ProjectionUpdateV1, ReductionReceiptV1
from orion.schemas.state_delta import StateDeltaV1

from .grammar_extract import extract_node_state_from_events
from .ids import parse_biometrics_trace_id

BIOMETRICS_SOURCE_SERVICE = "orion-biometrics"
DEFAULT_STALE_AFTER_SEC = 180


def _utc_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    return now if now.tzinfo else now.replace(tzinfo=timezone.utc)


def _noop_receipt(*, event_id: str, now: datetime, reducer_id: str) -> ReductionReceiptV1:
    return ReductionReceiptV1(
        receipt_id=stable_receipt_id(
            reducer_id=reducer_id,
            accepted_event_ids=[],
            rejected_event_ids=[],
            merged_event_ids=[],
            noop_event_ids=[event_id],
        ),
        noop_event_ids=[event_id],
        created_at=now,
    )


def reduce_biometrics_node_event(
    *,
    event: GrammarEventV1,
    projection: NodeBiometricsProjectionV1,
    catalog: NodeCatalog,
    stale_after_sec: int = DEFAULT_STALE_AFTER_SEC,
    reducer_id: str = "biometrics_node_reducer",
    now: datetime | None = None,
) -> tuple[NodeBiometricsProjectionV1, ReductionReceiptV1]:
    clock = _utc_now(now)

    if event.provenance.source_service != BIOMETRICS_SOURCE_SERVICE:
        return projection, _noop_receipt(
            event_id=event.event_id, now=clock, reducer_id=reducer_id
        )

    if not parse_biometrics_trace_id(event.trace_id):
        return projection, _noop_receipt(
            event_id=event.event_id, now=clock, reducer_id=reducer_id
        )

    updated = deepcopy(projection)
    updated.generated_at = clock

    extracted = extract_node_state_from_events(
        [event],
        catalog,
        stale_after_sec=stale_after_sec,
        now=clock,
    )
    node_id = extracted.node_id
    existing = updated.nodes.get(node_id)

    if existing is None:
        merged = extracted
        operation = "create"
    else:
        merged = existing.model_copy(deep=True)
        if extracted.last_seen_at and (
            merged.last_seen_at is None or extracted.last_seen_at >= merged.last_seen_at
        ):
            merged.last_seen_at = extracted.last_seen_at
        if extracted.latest_trace_id:
            merged.latest_trace_id = extracted.latest_trace_id
        if extracted.latest_payload_ref:
            merged.latest_payload_ref = extracted.latest_payload_ref
        if extracted.latest_sample_event_id:
            merged.latest_sample_event_id = extracted.latest_sample_event_id
        if extracted.latest_summary_event_id:
            merged.latest_summary_event_id = extracted.latest_summary_event_id
        if extracted.latest_induction_event_id:
            merged.latest_induction_event_id = extracted.latest_induction_event_id
        if extracted.pressure_hints:
            merged.pressure_hints = {**merged.pressure_hints, **extracted.pressure_hints}
        merged.availability_status = extracted.availability_status
        merged.role = extracted.role or merged.role
        merged.capabilities = extracted.capabilities or merged.capabilities
        merged.expected_online = extracted.expected_online
        merged.aliases = extracted.aliases or merged.aliases
        operation = "update"

    updated.nodes[node_id] = merged

    receipt = ReductionReceiptV1(
        receipt_id=stable_receipt_id(
            reducer_id=reducer_id,
            accepted_event_ids=[event.event_id],
            rejected_event_ids=[],
            merged_event_ids=[],
            noop_event_ids=[],
        ),
        accepted_event_ids=[event.event_id],
        state_deltas=[
            StateDeltaV1(
                delta_id=stable_delta_id(
                    reducer_id=reducer_id,
                    target_projection=updated.projection_id,
                    target_kind="node_biometrics",
                    target_id=node_id,
                    operation=operation,
                    caused_by_event_ids=[event.event_id],
                ),
                target_projection=updated.projection_id,
                target_kind="node_biometrics",
                target_id=node_id,
                operation=operation,
                before=existing.model_dump(mode="json") if existing else None,
                after=merged.model_dump(mode="json"),
                caused_by_event_ids=[event.event_id],
                reducer_id=reducer_id,
            )
        ],
        projection_updates=[
            ProjectionUpdateV1(
                projection_kind="node_biometrics",
                projection_id=updated.projection_id,
                node_id=node_id,
                operation=operation,
            )
        ],
        created_at=clock,
    )
    return updated, receipt
