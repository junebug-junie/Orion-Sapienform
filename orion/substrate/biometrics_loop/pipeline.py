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
from orion.schemas.reduction_receipt import ProjectionUpdateV1, ReductionReceiptV1
from orion.schemas.state_delta import StateDeltaV1
from orion.substrate.ids import stable_delta_id, stable_hash_id

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


PRUNE_REDUCER_ID = "biometrics_retired_node_prune"


def prune_retired_nodes(
    *,
    node_bio: NodeBiometricsProjectionV1,
    pressure: ActiveNodePressureProjectionV1,
    catalog: NodeCatalog,
    now: datetime,
    prune_node_bio: bool = True,
    prune_pressure: bool = True,
) -> tuple[NodeBiometricsProjectionV1, ActiveNodePressureProjectionV1, ReductionReceiptV1 | None]:
    """Drop retired hosts (catalog `retired_nodes:`) from both live projections.

    A host that stops existing never sends another event, so nothing else ever
    rewrites its entry: its last reading sits in the projection forever and every
    generic consumer keeps counting it. Confirmed live 2026-10-16: atlas (last report
    2026-08-20) still had 8 frozen `pressure_hints` averaged into every
    `biometrics_prediction_error()` tick as zero deltas, and a stale `availability`
    pressure re-reinforced on every sweep. The node reducer, the organ and the
    pressure reducer all refuse retired hosts, so once pruned they stay out.

    Returns the (possibly) new projections and one receipt naming what was removed,
    or None when there was nothing to prune. `prune_node_bio` / `prune_pressure`
    let a caller that will not persist one projection leave it alone, so the
    receipt never claims a removal that was not saved.
    """
    bio_gone = (
        sorted(n for n in (node_bio.nodes or {}) if catalog.is_retired(n)) if prune_node_bio else []
    )
    pressure_gone = (
        sorted(n for n in (pressure.nodes or {}) if catalog.is_retired(n)) if prune_pressure else []
    )
    if not bio_gone and not pressure_gone:
        return node_bio, pressure, None

    deltas: list[StateDeltaV1] = []
    updates: list[ProjectionUpdateV1] = []

    def _record(projection_id: str, kind: str, node_id: str, before: dict) -> None:
        deltas.append(
            StateDeltaV1(
                delta_id=stable_delta_id(
                    reducer_id=PRUNE_REDUCER_ID,
                    target_projection=projection_id,
                    target_kind=kind,
                    target_id=node_id,
                    operation="suppress",
                    caused_by_event_ids=[],
                ),
                target_projection=projection_id,
                target_kind=kind,
                target_id=node_id,
                operation="suppress",
                before=before,
                after=None,
                caused_by_event_ids=[],
                reducer_id=PRUNE_REDUCER_ID,
                explanation="retired host (config/biometrics/node_catalog.yaml retired_nodes) removed from live projection",
            )
        )
        updates.append(
            ProjectionUpdateV1(
                projection_kind=kind,
                projection_id=projection_id,
                node_id=node_id,
                operation="remove",
            )
        )

    if bio_gone:
        node_bio = node_bio.model_copy(deep=True)
        for node_id in bio_gone:
            state = node_bio.nodes.pop(node_id)
            _record(node_bio.projection_id, "node_biometrics", node_id, state.model_dump(mode="json"))
    if pressure_gone:
        pressure = pressure.model_copy(deep=True)
        for node_id in pressure_gone:
            state = pressure.nodes.pop(node_id)
            _record(pressure.projection_id, "active_node_pressure", node_id, state.model_dump(mode="json"))

    receipt = ReductionReceiptV1(
        receipt_id=stable_hash_id(
            "receipt",
            [PRUNE_REDUCER_ID, ",".join(bio_gone), ",".join(pressure_gone), now.isoformat()],
        ),
        state_deltas=deltas,
        projection_updates=updates,
        created_at=now,
    )
    return node_bio, pressure, receipt


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
    offline_frozen_hint_guard: bool = True,
) -> dict[str, int]:
    clock = now or datetime.now(timezone.utc)
    stats = {"events": 0, "receipts": 0, "emissions": 0, "published": 0}

    node_bio = load_node_bio()
    pressure = load_pressure()
    node_bio_dirty = False
    pressure_dirty = False

    # Prune only the projections this call owns and saves: the runtime's absence
    # sweep runs with enable_node_reducer=False and a no-op node-bio saver.
    pruned_bio, pruned_pressure, prune_receipt = prune_retired_nodes(
        node_bio=node_bio,
        pressure=pressure,
        catalog=catalog,
        now=clock,
        prune_node_bio=enable_node_reducer,
    )
    if prune_receipt is not None:
        node_bio_dirty = pruned_bio is not node_bio
        pressure_dirty = pruned_pressure is not pressure
        node_bio, pressure = pruned_bio, pruned_pressure
        save_receipt(prune_receipt)
        stats["receipts"] += 1

    for event in events:
        stats["events"] += 1

        if enable_node_reducer:
            node_bio, bio_receipt = reduce_biometrics_node_event(
                event=event,
                projection=node_bio,
                catalog=catalog,
                stale_after_sec=stale_after_sec,
            )
            node_bio_dirty = True
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
            offline_frozen_hint_guard=offline_frozen_hint_guard,
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
        pressure_dirty = True
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

    # Flush projections once per batch instead of once per event to avoid hot-row TOAST bloat.
    if node_bio_dirty:
        save_node_bio(node_bio)
    if pressure_dirty:
        save_pressure(pressure)

    return stats
