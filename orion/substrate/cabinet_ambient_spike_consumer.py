"""Consume ``cabinet.ambient.spike.v1`` into grammar trace + node biometrics bump.

v1 cognition consumer: one inspectable grammar trace and one
``cabinet_ambient_audio_activity`` pressure hint bump on the spiking node.
No STT, no keyword triggers.
"""

from __future__ import annotations

import hashlib
from copy import deepcopy
from datetime import datetime, timezone

from orion.biometrics.node_catalog import NodeCatalog
from orion.schemas.biometrics_projection import NodeBiometricsProjectionV1
from orion.schemas.grammar import GrammarAtomV1, GrammarEventV1, GrammarProvenanceV1
from orion.schemas.reduction_receipt import ProjectionUpdateV1, ReductionReceiptV1
from orion.schemas.state_delta import StateDeltaV1
from orion.schemas.telemetry.cabinet_ambient_spike import CabinetAmbientSpikeV1
from orion.substrate.biometrics_loop.constants import NODE_BIOMETRICS_PROJECTION_ID
from orion.substrate.ids import stable_delta_id, stable_receipt_id

REDUCER_ID = "cabinet_ambient_spike_consumer"
SOURCE_SERVICE = "orion-substrate-runtime"
SOURCE_COMPONENT = "cabinet_ambient_spike_listener"
SPIKE_TRACE_PREFIX = "cabinet.ambient.spike:"
HINT_KEY = "cabinet_ambient_audio_activity"


def _utc_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    return now if now.tzinfo else now.replace(tzinfo=timezone.utc)


def _hash_id(*parts: object, prefix: str) -> str:
    raw = "|".join(str(p) for p in parts)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def spike_trace_id(spike: CabinetAmbientSpikeV1) -> str:
    return f"{SPIKE_TRACE_PREFIX}{spike.node}:{spike.spike_id}"


def build_cabinet_ambient_spike_grammar_events(
    spike: CabinetAmbientSpikeV1,
) -> list[GrammarEventV1]:
    """Build a minimal grammar trace for one sustained ambient spike."""
    clock = _utc_now(spike.timestamp)
    trace_id = spike_trace_id(spike)
    provenance = GrammarProvenanceV1(
        source_service=SOURCE_SERVICE,
        source_component=SOURCE_COMPONENT,
        source_trace_id=trace_id,
    )

    root = GrammarEventV1(
        event_id=_hash_id(trace_id, "trace_started", prefix="gev"),
        event_kind="trace_started",
        trace_id=trace_id,
        emitted_at=clock,
        observed_at=clock,
        layer="organ_signal",
        dimensions=["physiology", "telemetry", "node", "cabinet", "spike"],
        provenance=provenance,
    )
    root_id = root.event_id

    atom = GrammarAtomV1(
        atom_id=f"{trace_id}:cabinet_ambient_audio_spike_signal",
        trace_id=trace_id,
        atom_type="signal",
        semantic_role="cabinet_ambient_audio_spike_signal",
        layer="organ_signal",
        dimensions=["physiology", "telemetry", "node", "cabinet", "spike"],
        summary=f"{spike.node} cabinet ambient audio spike observed",
        confidence=0.95,
        salience=float(spike.activity),
        source_event_id=spike.spike_id,
        payload_ref=f"cabinet.ambient.spike:{spike.spike_id}",
        text_value=spike.node,
    )
    atom_event = GrammarEventV1(
        event_id=_hash_id(trace_id, "atom", atom.atom_id, prefix="gev"),
        event_kind="atom_emitted",
        trace_id=trace_id,
        parent_event_id=root_id,
        root_event_id=root_id,
        emitted_at=clock,
        observed_at=clock,
        layer="organ_signal",
        dimensions=["physiology", "telemetry", "node", "cabinet", "spike"],
        atom=atom,
        provenance=provenance,
    )

    end = GrammarEventV1(
        event_id=_hash_id(trace_id, "trace_ended", prefix="gev"),
        event_kind="trace_ended",
        trace_id=trace_id,
        parent_event_id=root_id,
        root_event_id=root_id,
        emitted_at=clock,
        observed_at=clock,
        layer="organ_signal",
        dimensions=["physiology", "telemetry", "node", "cabinet", "spike"],
        provenance=provenance,
    )

    return [root, atom_event, end]


def _merged_activity_hint(existing: float | None, spike_activity: float) -> float:
    prior = float(existing or 0.0)
    return max(prior, float(spike_activity))


def apply_cabinet_ambient_spike_bump(
    *,
    projection: NodeBiometricsProjectionV1,
    spike: CabinetAmbientSpikeV1,
    catalog: NodeCatalog,
    grammar_events: list[GrammarEventV1],
    now: datetime | None = None,
) -> tuple[NodeBiometricsProjectionV1, ReductionReceiptV1]:
    """Bump node biometrics pressure hint and emit a field-digester receipt."""
    clock = _utc_now(now)
    node_id = catalog.resolve(spike.node).node_id
    atom_event = next(
        (event for event in grammar_events if event.atom is not None),
        None,
    )
    if atom_event is None:
        raise ValueError("grammar_events must include an atom_emitted event")

    updated = deepcopy(projection)
    updated.generated_at = clock
    existing = updated.nodes.get(node_id)

    if existing is None:
        from orion.schemas.biometrics_projection import NodeBiometricsStateV1

        merged = NodeBiometricsStateV1(
            node_id=node_id,
            pressure_hints={HINT_KEY: float(spike.activity)},
            last_seen_at=clock,
            latest_trace_id=spike_trace_id(spike),
        )
        operation = "create"
    else:
        merged = existing.model_copy(deep=True)
        merged.last_seen_at = clock
        merged.latest_trace_id = spike_trace_id(spike)
        hints = dict(merged.pressure_hints or {})
        hints[HINT_KEY] = _merged_activity_hint(hints.get(HINT_KEY), spike.activity)
        merged.pressure_hints = hints
        operation = "update"

    updated.nodes[node_id] = merged

    receipt = ReductionReceiptV1(
        receipt_id=stable_receipt_id(
            reducer_id=REDUCER_ID,
            accepted_event_ids=[atom_event.event_id],
            rejected_event_ids=[],
            merged_event_ids=[],
            noop_event_ids=[],
        ),
        accepted_event_ids=[atom_event.event_id],
        state_deltas=[
            StateDeltaV1(
                delta_id=stable_delta_id(
                    reducer_id=REDUCER_ID,
                    target_projection=NODE_BIOMETRICS_PROJECTION_ID,
                    target_kind="node_biometrics",
                    target_id=node_id,
                    operation=operation,
                    caused_by_event_ids=[atom_event.event_id],
                ),
                target_projection=NODE_BIOMETRICS_PROJECTION_ID,
                target_kind="node_biometrics",
                target_id=node_id,
                operation=operation,
                before=existing.model_dump(mode="json") if existing else None,
                after=merged.model_dump(mode="json"),
                caused_by_event_ids=[atom_event.event_id],
                reducer_id=REDUCER_ID,
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
