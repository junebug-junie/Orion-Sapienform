from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4

from orion.biometrics.node_catalog import NodeCatalog
from orion.schemas.biometrics_projection import (
    ActiveNodePressureProjectionV1,
    NodeBiometricsProjectionV1,
)
from orion.schemas.grammar import GrammarEventV1
from orion.schemas.organ_emission import OrganEmissionV1

from .constants import (
    ACTIVE_NODE_PRESSURE_PROJECTION_ID,
    NODE_BIOMETRICS_PROJECTION_ID,
)
from .emission_validator import group_candidate_events_by_trace, validate_organ_emission
from .node_reducer import reduce_biometrics_node_event
from .pressure_organ import invoke_biometrics_pressure
from .pressure_reducer import reduce_node_pressure_candidates

GrammarEventFetcher = Callable[[], list[tuple[GrammarEventV1, datetime]]]
NodeBioLoader = Callable[[], NodeBiometricsProjectionV1]
NodeBioSaver = Callable[[NodeBiometricsProjectionV1], None]
PressureLoader = Callable[[], ActiveNodePressureProjectionV1]
PressureSaver = Callable[[ActiveNodePressureProjectionV1], None]
ReceiptSaver = Callable[[Any], None]
EmissionSaver = Callable[[OrganEmissionV1], None]
AcceptedPublisher = Callable[[list[GrammarEventV1]], None]


def _empty_node_bio(now: datetime) -> NodeBiometricsProjectionV1:
    return NodeBiometricsProjectionV1(
        projection_id=NODE_BIOMETRICS_PROJECTION_ID,
        generated_at=now,
        nodes={},
    )


def _empty_pressure(now: datetime) -> ActiveNodePressureProjectionV1:
    return ActiveNodePressureProjectionV1(
        projection_id=ACTIVE_NODE_PRESSURE_PROJECTION_ID,
        generated_at=now,
        nodes={},
    )


def process_biometrics_grammar_events(
    *,
    events: list[GrammarEventV1],
    catalog: NodeCatalog,
    load_node_bio: NodeBioLoader,
    save_node_bio: NodeBioSaver,
    load_pressure: PressureLoader,
    save_pressure: PressureSaver,
    save_receipt: ReceiptSaver,
    save_emission: EmissionSaver,
    publish_accepted: AcceptedPublisher | None,
    enable_node_reducer: bool = True,
    enable_organ: bool = True,
    enable_pressure_reducer: bool = True,
    stale_after_sec: int = 180,
    min_confidence: float = 0.60,
    now: datetime | None = None,
) -> dict[str, int]:
    clock = now or datetime.now(timezone.utc)
    stats = {"events": 0, "receipts": 0, "emissions": 0, "published": 0}

    node_bio = load_node_bio()
    pressure = load_pressure()

    for event in events:
        stats["events"] += 1

        if enable_node_reducer:
            node_bio, bio_receipt = reduce_biometrics_node_event(
                event=event,
                projection=node_bio,
                catalog=catalog,
                stale_after_sec=stale_after_sec,
            )
            save_node_bio(node_bio)
            save_receipt(bio_receipt)
            stats["receipts"] += 1

        if not enable_organ:
            continue

        emission = invoke_biometrics_pressure(
            trigger_event=event,
            node_bio=node_bio,
            active_pressure=pressure,
            catalog=catalog,
            stale_after_sec=stale_after_sec,
            min_confidence=min_confidence,
            now=clock,
        )
        emission = validate_organ_emission(emission)
        if emission.candidate_events:
            save_emission(emission)
            stats["emissions"] += 1

        if not enable_pressure_reducer:
            continue

        candidate_traces = (
            group_candidate_events_by_trace(emission.candidate_events)
            if emission.candidate_events
            else []
        )
        pressure, pressure_receipt = reduce_node_pressure_candidates(
            candidates=candidate_traces,
            projection=pressure,
            catalog=catalog,
            min_confidence=min_confidence,
            emission_id=emission.emission_id,
            now=clock,
        )
        save_pressure(pressure)
        save_receipt(pressure_receipt)
        stats["receipts"] += 1

        if publish_accepted and pressure_receipt.accepted_event_ids:
            accepted = [
                c
                for c in emission.candidate_events
                if c.event_id in pressure_receipt.accepted_event_ids
            ]
            if accepted:
                publish_accepted(accepted)
                stats["published"] += len(accepted)

    return stats
